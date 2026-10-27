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
// Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek.
// DeepGEMM portions are licensed under MIT; see DEEPGEMM_NOTICE.txt in this directory.

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
#define TMEM_NCOLS 396
#define TMEM_TMEM_ACC_OFFSET 0
#define TMEM_TMEM_SF_Q_OFFSET 384
#define TMEM_TMEM_SF_KV_OFFSET 388
#define NUM_Q_PIPE_STAGES 3
#define NUM_KV_PIPE_STAGES 10
#define NUM_TMEM_PIPE_STAGES 3
#define SMEM_NEGINF_SCRATCH_OFF 218240
#define SMEM_NEGINF_SCRATCH_STAGE_BYTES 4096
#define SMEM_NEGINF_SCRATCH_STRIDE 4096
#define SMEM_LOCAL_WORK_OFF 201728
#define SMEM_LOCAL_WORK_STAGE_BYTES 4096
#define SMEM_LOCAL_WORK_STRIDE 4096
#define SMEM_LOCAL_COST_OFF 205824
#define SMEM_LOCAL_COST_STAGE_BYTES 4096
#define SMEM_LOCAL_COST_STRIDE 4096
#define SMEM_LOCAL_BASE_OFF 209920
#define SMEM_LOCAL_BASE_STAGE_BYTES 4096
#define SMEM_LOCAL_BASE_STRIDE 4096
#define SMEM_LOCAL_SPLITS_OFF 214016
#define SMEM_LOCAL_SPLITS_STAGE_BYTES 4096
#define SMEM_LOCAL_SPLITS_STRIDE 4096
#define SMEM_SCAN_WARP_SUMS_OFF 218112
#define SMEM_SCAN_WARP_SUMS_STAGE_BYTES 96
#define SMEM_SCAN_WARP_SUMS_STRIDE 96
#define SMEM_SCAN_CARRY_OFF 218208
#define SMEM_SCAN_CARRY_STAGE_BYTES 8
#define SMEM_SCAN_CARRY_STRIDE 8
#define SMEM_LOCAL_HEADER_OFF 218216
#define SMEM_LOCAL_HEADER_STAGE_BYTES 12
#define SMEM_LOCAL_HEADER_STRIDE 12
#define SMEM_SMEM_Q_FP4_OFF 0
#define SMEM_SMEM_Q_FP4_STAGE_BYTES 8192
#define SMEM_SMEM_Q_FP4_STRIDE 8192
#define SMEM_SMEM_KV_FP4_OFF 24576
#define SMEM_SMEM_KV_FP4_STAGE_BYTES 16384
#define SMEM_SMEM_KV_FP4_STRIDE 16384
#define SMEM_SMEM_SF_Q_OFF 188416
#define SMEM_SMEM_SF_Q_STAGE_BYTES 512
#define SMEM_SMEM_SF_Q_STRIDE 512
#define SMEM_SMEM_SF_KV_OFF 189952
#define SMEM_SMEM_SF_KV_STAGE_BYTES 1024
#define SMEM_SMEM_SF_KV_STRIDE 1024
#define SMEM_SMEM_WEIGHTS_OFF 200192
#define SMEM_SMEM_WEIGHTS_STAGE_BYTES 512
#define SMEM_SMEM_WEIGHTS_STRIDE 512
#define SMEM_TOTAL 222720
#ifndef SM_COUNT
#error "SM_COUNT is a downstream specialization of this program; define it on the compile line"
#endif
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


__device__ __forceinline__ void tcgen05_mma_mxf4_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}



__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


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


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_leader(
    int taddr, uint64_t s_desc, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %2, 0;\n\t"
        "@leader tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n\t"
        "}\n"
        :: "r"(taddr), "l"(s_desc), "r"(leader) : "memory");
}



__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit_leader(
    int mbar_addr, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %1, 0;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(leader) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_deepgemm_dense_mqa_76983cd8926bee02337c(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap Weights, const __grid_constant__ CUtensorMap SF_Q, const __grid_constant__ CUtensorMap SF_KV, float* __restrict__ Logits, int* __restrict__ cu_seq_len_k_start, int* __restrict__ cu_seq_len_k_end, int seq_len, int seq_len_kv, int stride_logits, int num_q_blocks, unsigned int* __restrict__ ScheduleMeta)
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

    const int mbar_base = smem + 222336;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 24)
    #define kv_full_addr (mbar_base + 48)
    #define kv_empty_addr (mbar_base + 128)
    #define umma_full_addr (mbar_base + 208)
    #define umma_empty_addr (mbar_base + 232)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* neginf_scratch = reinterpret_cast<float*>(smem_raw + 218240);
    const int neginf_scratch_addr = smem + 218240;
    unsigned int* local_work = reinterpret_cast<unsigned int*>(smem_raw + 201728);
    const int local_work_addr = smem + 201728;
    unsigned int* local_cost = reinterpret_cast<unsigned int*>(smem_raw + 205824);
    const int local_cost_addr = smem + 205824;
    unsigned int* local_base = reinterpret_cast<unsigned int*>(smem_raw + 209920);
    const int local_base_addr = smem + 209920;
    unsigned int* local_splits = reinterpret_cast<unsigned int*>(smem_raw + 214016);
    const int local_splits_addr = smem + 214016;
    unsigned long long* scan_warp_sums = reinterpret_cast<unsigned long long*>(smem_raw + 218112);
    const int scan_warp_sums_addr = smem + 218112;
    unsigned long long* scan_carry = reinterpret_cast<unsigned long long*>(smem_raw + 218208);
    const int scan_carry_addr = smem + 218208;
    unsigned int* local_header = reinterpret_cast<unsigned int*>(smem_raw + 218216);
    const int local_header_addr = smem + 218216;
    uint8_t* smem_q_fp4 = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int smem_q_fp4_addr = smem + 0;
    uint8_t* smem_kv_fp4 = reinterpret_cast<uint8_t*>(smem_raw + 24576);
    const int smem_kv_fp4_addr = smem + 24576;
    uint8_t* smem_sf_q = reinterpret_cast<uint8_t*>(smem_raw + 188416);
    const int smem_sf_q_addr = smem + 188416;
    uint8_t* smem_sf_kv = reinterpret_cast<uint8_t*>(smem_raw + 189952);
    const int smem_sf_kv_addr = smem + 189952;
    uint8_t* smem_weights = reinterpret_cast<uint8_t*>(smem_raw + 200192);
    const int smem_weights_addr = smem + 200192;
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SF_Q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SF_KV))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Weights))) : "memory"); }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int pre_win[8];
    unsigned int pre_q = tid;
    if (pre_q < (unsigned int)num_q_blocks) {
        #pragma unroll
        for (int qi = 0; qi < 4; qi++) {
            unsigned int row = ((pre_q * 4 + (unsigned int)qi) < ((unsigned int)seq_len - 1) ? (pre_q * 4 + (unsigned int)qi) : ((unsigned int)seq_len - 1));
            pre_win[qi] = (unsigned int)cu_seq_len_k_start[row];
            pre_win[4 + qi] = (unsigned int)cu_seq_len_k_end[row];
        }
    }
    if (tid == 352) {
        unsigned int pf_kv_len = (unsigned int)seq_len_kv;
        unsigned int pf_start = (((unsigned int)cu_seq_len_k_start[0]) < (pf_kv_len) ? ((unsigned int)cu_seq_len_k_start[0]) : (pf_kv_len));
        unsigned int pf_end = (((unsigned int)cu_seq_len_k_end[0]) < (pf_kv_len) ? ((unsigned int)cu_seq_len_k_end[0]) : (pf_kv_len));
        unsigned int pf_base = pf_start / 4 * 4;
        unsigned int pf_splits = 0;
        if (pf_end > pf_base) {
            pf_splits = (pf_end - pf_base + 255) / 256;
        }
        if (pf_splits != 0) {
            unsigned int pf_cost = pf_splits + 1;
            unsigned int pf_total = (unsigned int)num_q_blocks * pf_cost;
            unsigned int quotient = pf_total / (unsigned int)SM_COUNT;
            unsigned int remainder = pf_total % (unsigned int)SM_COUNT;
            unsigned int _min_0 = (((unsigned int)bid) < (remainder) ? ((unsigned int)bid) : (remainder));
            unsigned int pf_target = (unsigned int)bid * quotient + _min_0;
            if (pf_target < pf_total) {
                unsigned int pf_block = pf_target / pf_cost;
                unsigned int _max_1 = ((pf_target - pf_block * pf_cost) > (1) ? (pf_target - pf_block * pf_cost) : (1));
                unsigned int pf_split = ((_max_1 - 1) < (pf_splits - 1) ? (_max_1 - 1) : (pf_splits - 1));
                int pf_kv_row = (int)(pf_base + pf_split * 256);
                unsigned int pf_q_first = 1;
                if (bid > 0) {
                    unsigned int _min_2 = (((unsigned int)(bid - 1)) < (remainder) ? ((unsigned int)(bid - 1)) : (remainder));
                    unsigned int pf_prev = (unsigned int)(bid - 1) * quotient + _min_2;
                    if (pf_prev / pf_cost == pf_block) {
                        pf_q_first = 0;
                    }
                }
                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&KV))), "r"((int)(0)), "r"((int)(pf_kv_row)) : "memory");
                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&SF_KV))), "r"((int)(0)), "r"((int)(pf_kv_row / 4)) : "memory");
                if (pf_q_first != 0) {
                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&Q))), "r"((int)(0)), "r"((int)(pf_block * 128)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&SF_Q))), "r"((int)(0)), "r"((int)(pf_block * 32)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&Weights))), "r"((int)(0)), "r"((int)(pf_block * 4)) : "memory");
                }
            }
        }
    }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 32 barriers)
    // Mbarriers at smem_raw[222336..222592)

    if (warp == 9) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 3 barriers, init_count=1
            mbarrier_init(smem + 222336, 1);
            mbarrier_init(smem + 222344, 1);
            mbarrier_init(smem + 222352, 1);
            // q_empty: 3 barriers, init_count=257
            mbarrier_init(smem + 222360, 257);
            mbarrier_init(smem + 222368, 257);
            mbarrier_init(smem + 222376, 257);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 10 barriers, init_count=1
            mbarrier_init(smem + 222384, 1);
            mbarrier_init(smem + 222392, 1);
            mbarrier_init(smem + 222400, 1);
            mbarrier_init(smem + 222408, 1);
            mbarrier_init(smem + 222416, 1);
            mbarrier_init(smem + 222424, 1);
            mbarrier_init(smem + 222432, 1);
            mbarrier_init(smem + 222440, 1);
            mbarrier_init(smem + 222448, 1);
            mbarrier_init(smem + 222456, 1);
            // kv_empty: 10 barriers, init_count=1
            mbarrier_init(smem + 222464, 1);
            mbarrier_init(smem + 222472, 1);
            mbarrier_init(smem + 222480, 1);
            mbarrier_init(smem + 222488, 1);
            mbarrier_init(smem + 222496, 1);
            mbarrier_init(smem + 222504, 1);
            mbarrier_init(smem + 222512, 1);
            mbarrier_init(smem + 222520, 1);
            mbarrier_init(smem + 222528, 1);
            mbarrier_init(smem + 222536, 1);
            // --- pipeline 'tmem_pipe' ---
            // umma_full: 3 barriers, init_count=1
            mbarrier_init(smem + 222544, 1);
            mbarrier_init(smem + 222552, 1);
            mbarrier_init(smem + 222560, 1);
            // umma_empty: 3 barriers, init_count=128
            mbarrier_init(smem + 222568, 128);
            mbarrier_init(smem + 222576, 128);
            mbarrier_init(smem + 222584, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 396 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 222592);
    if (warp == 10) {
        int _tmem_hold = smem + 222592;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    const int tmem_tmem_sf_q = taddr + 384;
    const int tmem_tmem_sf_kv = taddr + 388;
    unsigned int scan_span_offset = (3 * SM_COUNT + 1) / 2 * 2;
    if (tid == 0) {
        scan_carry[0] = 0;
    }
    unsigned int qidx_1 = tid;
    unsigned long long value_1 = 0;
    if (qidx_1 < (unsigned int)num_q_blocks) {
        unsigned int start = 4294967295;
        unsigned int end = 0;
        #pragma unroll
        for (int qi_1 = 0; qi_1 < 4; qi_1++) {
            unsigned int _min_1 = ((pre_win[qi_1]) < ((unsigned int)seq_len_kv) ? (pre_win[qi_1]) : ((unsigned int)seq_len_kv));
            unsigned int _min_0_1 = ((start) < (_min_1) ? (start) : (_min_1));
            start = _min_0_1;
            unsigned int _min_3 = ((pre_win[4 + qi_1]) < ((unsigned int)seq_len_kv) ? (pre_win[4 + qi_1]) : ((unsigned int)seq_len_kv));
            unsigned int _max_2 = ((end) > (_min_3) ? (end) : (_min_3));
            end = _max_2;
        }
        unsigned int base = start / 4 * 4;
        unsigned int splits = (end - base + 255) / 256;
        unsigned int cost = splits;
        if (splits != 0) {
            cost += 1;
        }
        local_base[qidx_1] = base;
        local_splits[qidx_1] = splits;
        if (bid == 0) {
            ScheduleMeta[scan_span_offset + 2 * qidx_1] = base;
            ScheduleMeta[scan_span_offset + 2 * qidx_1 + 1] = splits;
        }
        value_1 = ((unsigned long long)splits << 32) + (unsigned long long)cost;
    }
    __syncthreads();
    unsigned long long previous_carry = scan_carry[0];
    unsigned long long lane_sum = value_1;
    unsigned long long synced = __shfl_up_sync(0xFFFFFFFF, lane_sum, 1, 32);
    if (lane >= 1) {
        lane_sum += synced;
    }
    unsigned long long synced_0 = __shfl_up_sync(0xFFFFFFFF, lane_sum, 2, 32);
    if (lane >= 2) {
        lane_sum += synced_0;
    }
    unsigned long long synced_1 = __shfl_up_sync(0xFFFFFFFF, lane_sum, 4, 32);
    if (lane >= 4) {
        lane_sum += synced_1;
    }
    unsigned long long synced_2 = __shfl_up_sync(0xFFFFFFFF, lane_sum, 8, 32);
    if (lane >= 8) {
        lane_sum += synced_2;
    }
    unsigned long long synced_3 = __shfl_up_sync(0xFFFFFFFF, lane_sum, 16, 32);
    if (lane >= 16) {
        lane_sum += synced_3;
    }
    if (lane == 31) {
        scan_warp_sums[warp] = lane_sum;
    }
    __syncthreads();
    unsigned long long warp_total = 0;
    if (lane < 12) {
        warp_total = scan_warp_sums[lane];
    }
    unsigned long long warp_sum = warp_total;
    unsigned long long synced_w = __shfl_up_sync(0xFFFFFFFF, warp_sum, 1, 32);
    if (lane >= 1) {
        warp_sum += synced_w;
    }
    unsigned long long synced_w_4 = __shfl_up_sync(0xFFFFFFFF, warp_sum, 2, 32);
    if (lane >= 2) {
        warp_sum += synced_w_4;
    }
    unsigned long long synced_w_5 = __shfl_up_sync(0xFFFFFFFF, warp_sum, 4, 32);
    if (lane >= 4) {
        warp_sum += synced_w_5;
    }
    unsigned long long synced_w_6 = __shfl_up_sync(0xFFFFFFFF, warp_sum, 8, 32);
    if (lane >= 8) {
        warp_sum += synced_w_6;
    }
    unsigned long long synced_w_7 = __shfl_up_sync(0xFFFFFFFF, warp_sum, 16, 32);
    if (lane >= 16) {
        warp_sum += synced_w_7;
    }
    unsigned long long preceding = __shfl_sync(0xFFFFFFFF, warp_sum - warp_total, warp);
    unsigned long long scanned_1 = lane_sum - value_1 + preceding + value_1 + previous_carry;
    if (qidx_1 < (unsigned int)num_q_blocks) {
        local_work[qidx_1] = (unsigned int)(scanned_1 >> 32);
        local_cost[qidx_1] = (unsigned int)scanned_1;
    }
    if (tid == 383) {
        scan_carry[0] = scanned_1;
    }
    __syncthreads();
    #pragma unroll 1
    for (unsigned int block_base = 384; block_base < (unsigned int)num_q_blocks; block_base += 384) {
        unsigned int qidx_1_0 = block_base + (unsigned int)tid;
        unsigned long long value_1_1 = 0;
        if (qidx_1_0 < (unsigned int)num_q_blocks) {
            unsigned int start_1 = 4294967295;
            unsigned int end_1 = 0;
            #pragma unroll
            for (int qi_2 = 0; qi_2 < 4; qi_2++) {
                unsigned int row_1 = ((qidx_1_0 * 4 + (unsigned int)qi_2) < ((unsigned int)seq_len - 1) ? (qidx_1_0 * 4 + (unsigned int)qi_2) : ((unsigned int)seq_len - 1));
                unsigned int _min_5 = (((unsigned int)cu_seq_len_k_start[row_1]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[row_1]) : ((unsigned int)seq_len_kv));
                unsigned int _min_4 = ((start_1) < (_min_5) ? (start_1) : (_min_5));
                start_1 = _min_4;
                unsigned int _min_7 = (((unsigned int)cu_seq_len_k_end[row_1]) < ((unsigned int)seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[row_1]) : ((unsigned int)seq_len_kv));
                unsigned int _max_6 = ((end_1) > (_min_7) ? (end_1) : (_min_7));
                end_1 = _max_6;
            }
            unsigned int base_1 = start_1 / 4 * 4;
            unsigned int splits_1 = (end_1 - base_1 + 255) / 256;
            unsigned int cost_1 = splits_1;
            if (splits_1 != 0) {
                cost_1 += 1;
            }
            local_base[qidx_1_0] = base_1;
            local_splits[qidx_1_0] = splits_1;
            if (bid == 0) {
                ScheduleMeta[scan_span_offset + 2 * qidx_1_0] = base_1;
                ScheduleMeta[scan_span_offset + 2 * qidx_1_0 + 1] = splits_1;
            }
            value_1_1 = ((unsigned long long)splits_1 << 32) + (unsigned long long)cost_1;
        }
        __syncthreads();
        unsigned long long previous_carry_2 = scan_carry[0];
        unsigned long long lane_sum_3 = value_1_1;
        unsigned long long synced_4 = __shfl_up_sync(0xFFFFFFFF, lane_sum_3, 1, 32);
        if (lane >= 1) {
            lane_sum_3 += synced_4;
        }
        unsigned long long synced_5 = __shfl_up_sync(0xFFFFFFFF, lane_sum_3, 2, 32);
        if (lane >= 2) {
            lane_sum_3 += synced_5;
        }
        unsigned long long synced_6 = __shfl_up_sync(0xFFFFFFFF, lane_sum_3, 4, 32);
        if (lane >= 4) {
            lane_sum_3 += synced_6;
        }
        unsigned long long synced_7 = __shfl_up_sync(0xFFFFFFFF, lane_sum_3, 8, 32);
        if (lane >= 8) {
            lane_sum_3 += synced_7;
        }
        unsigned long long synced_8 = __shfl_up_sync(0xFFFFFFFF, lane_sum_3, 16, 32);
        if (lane >= 16) {
            lane_sum_3 += synced_8;
        }
        if (lane == 31) {
            scan_warp_sums[warp] = lane_sum_3;
        }
        __syncthreads();
        unsigned long long warp_total_9 = 0;
        if (lane < 12) {
            warp_total_9 = scan_warp_sums[lane];
        }
        unsigned long long warp_sum_10 = warp_total_9;
        unsigned long long synced_w_11 = __shfl_up_sync(0xFFFFFFFF, warp_sum_10, 1, 32);
        if (lane >= 1) {
            warp_sum_10 += synced_w_11;
        }
        unsigned long long synced_w_12 = __shfl_up_sync(0xFFFFFFFF, warp_sum_10, 2, 32);
        if (lane >= 2) {
            warp_sum_10 += synced_w_12;
        }
        unsigned long long synced_w_13 = __shfl_up_sync(0xFFFFFFFF, warp_sum_10, 4, 32);
        if (lane >= 4) {
            warp_sum_10 += synced_w_13;
        }
        unsigned long long synced_w_14 = __shfl_up_sync(0xFFFFFFFF, warp_sum_10, 8, 32);
        if (lane >= 8) {
            warp_sum_10 += synced_w_14;
        }
        unsigned long long synced_w_15 = __shfl_up_sync(0xFFFFFFFF, warp_sum_10, 16, 32);
        if (lane >= 16) {
            warp_sum_10 += synced_w_15;
        }
        unsigned long long preceding_16 = __shfl_sync(0xFFFFFFFF, warp_sum_10 - warp_total_9, warp);
        unsigned long long scanned_1_17 = lane_sum_3 - value_1_1 + preceding_16 + value_1_1 + previous_carry_2;
        if (qidx_1_0 < (unsigned int)num_q_blocks) {
            local_work[qidx_1_0] = (unsigned int)(scanned_1_17 >> 32);
            local_cost[qidx_1_0] = (unsigned int)scanned_1_17;
        }
        if (tid == 383) {
            scan_carry[0] = scanned_1_17;
        }
        __syncthreads();
    }
    if (tid == 0) {
        unsigned int scan_blocks = (unsigned int)num_q_blocks;
        unsigned int total_work = local_work[scan_blocks - 1];
        unsigned int total_cost = local_cost[scan_blocks - 1];
        unsigned int quotient_1 = total_cost / (unsigned int)SM_COUNT;
        unsigned int remainder_1 = total_cost % (unsigned int)SM_COUNT;
        unsigned int _min_8 = (((unsigned int)bid) < (remainder_1) ? ((unsigned int)bid) : (remainder_1));
        unsigned int target = (unsigned int)bid * quotient_1 + _min_8;
        unsigned int block = scan_blocks;
        unsigned int split = 0;
        unsigned int coordinate = total_work;
        if (target != total_cost) {
            block = 0;
            unsigned int probe = block + 1024;
            if (probe <= scan_blocks) {
                if (target >= local_cost[probe - 1]) {
                    block = probe;
                }
            }
            unsigned int probe_0 = block + 512;
            if (probe_0 <= scan_blocks) {
                if (target >= local_cost[probe_0 - 1]) {
                    block = probe_0;
                }
            }
            unsigned int probe_1 = block + 256;
            if (probe_1 <= scan_blocks) {
                if (target >= local_cost[probe_1 - 1]) {
                    block = probe_1;
                }
            }
            unsigned int probe_2 = block + 128;
            if (probe_2 <= scan_blocks) {
                if (target >= local_cost[probe_2 - 1]) {
                    block = probe_2;
                }
            }
            unsigned int probe_3 = block + 64;
            if (probe_3 <= scan_blocks) {
                if (target >= local_cost[probe_3 - 1]) {
                    block = probe_3;
                }
            }
            unsigned int probe_4 = block + 32;
            if (probe_4 <= scan_blocks) {
                if (target >= local_cost[probe_4 - 1]) {
                    block = probe_4;
                }
            }
            unsigned int probe_5 = block + 16;
            if (probe_5 <= scan_blocks) {
                if (target >= local_cost[probe_5 - 1]) {
                    block = probe_5;
                }
            }
            unsigned int probe_6 = block + 8;
            if (probe_6 <= scan_blocks) {
                if (target >= local_cost[probe_6 - 1]) {
                    block = probe_6;
                }
            }
            unsigned int probe_7 = block + 4;
            if (probe_7 <= scan_blocks) {
                if (target >= local_cost[probe_7 - 1]) {
                    block = probe_7;
                }
            }
            unsigned int probe_8 = block + 2;
            if (probe_8 <= scan_blocks) {
                if (target >= local_cost[probe_8 - 1]) {
                    block = probe_8;
                }
            }
            unsigned int probe_9 = block + 1;
            if (probe_9 <= scan_blocks) {
                if (target >= local_cost[probe_9 - 1]) {
                    block = probe_9;
                }
            }
            unsigned int cost_before = 0;
            unsigned int work_before = 0;
            if (block > 0) {
                cost_before = local_cost[block - 1];
                work_before = local_work[block - 1];
            }
            unsigned int _max_10 = ((target - cost_before) > (1) ? (target - cost_before) : (1));
            unsigned int _min_9 = ((_max_10 - 1) < (local_work[block] - work_before - 1) ? (_max_10 - 1) : (local_work[block] - work_before - 1));
            split = _min_9;
            coordinate = work_before + split;
        }
        unsigned int _min_11 = (((unsigned int)(bid + 1)) < (remainder_1) ? ((unsigned int)(bid + 1)) : (remainder_1));
        unsigned int target_0 = (unsigned int)(bid + 1) * quotient_1 + _min_11;
        unsigned int block_1 = scan_blocks;
        unsigned int split_2 = 0;
        unsigned int coordinate_3 = total_work;
        if (target_0 != total_cost) {
            block_1 = 0;
            unsigned int probe_10 = block_1 + 1024;
            if (probe_10 <= scan_blocks) {
                if (target_0 >= local_cost[probe_10 - 1]) {
                    block_1 = probe_10;
                }
            }
            unsigned int probe_0_1 = block_1 + 512;
            if (probe_0_1 <= scan_blocks) {
                if (target_0 >= local_cost[probe_0_1 - 1]) {
                    block_1 = probe_0_1;
                }
            }
            unsigned int probe_1_1 = block_1 + 256;
            if (probe_1_1 <= scan_blocks) {
                if (target_0 >= local_cost[probe_1_1 - 1]) {
                    block_1 = probe_1_1;
                }
            }
            unsigned int probe_2_1 = block_1 + 128;
            if (probe_2_1 <= scan_blocks) {
                if (target_0 >= local_cost[probe_2_1 - 1]) {
                    block_1 = probe_2_1;
                }
            }
            unsigned int probe_3_1 = block_1 + 64;
            if (probe_3_1 <= scan_blocks) {
                if (target_0 >= local_cost[probe_3_1 - 1]) {
                    block_1 = probe_3_1;
                }
            }
            unsigned int probe_4_1 = block_1 + 32;
            if (probe_4_1 <= scan_blocks) {
                if (target_0 >= local_cost[probe_4_1 - 1]) {
                    block_1 = probe_4_1;
                }
            }
            unsigned int probe_5_1 = block_1 + 16;
            if (probe_5_1 <= scan_blocks) {
                if (target_0 >= local_cost[probe_5_1 - 1]) {
                    block_1 = probe_5_1;
                }
            }
            unsigned int probe_6_1 = block_1 + 8;
            if (probe_6_1 <= scan_blocks) {
                if (target_0 >= local_cost[probe_6_1 - 1]) {
                    block_1 = probe_6_1;
                }
            }
            unsigned int probe_7_1 = block_1 + 4;
            if (probe_7_1 <= scan_blocks) {
                if (target_0 >= local_cost[probe_7_1 - 1]) {
                    block_1 = probe_7_1;
                }
            }
            unsigned int probe_8_1 = block_1 + 2;
            if (probe_8_1 <= scan_blocks) {
                if (target_0 >= local_cost[probe_8_1 - 1]) {
                    block_1 = probe_8_1;
                }
            }
            unsigned int probe_9_1 = block_1 + 1;
            if (probe_9_1 <= scan_blocks) {
                if (target_0 >= local_cost[probe_9_1 - 1]) {
                    block_1 = probe_9_1;
                }
            }
            unsigned int cost_before_1 = 0;
            unsigned int work_before_1 = 0;
            if (block_1 > 0) {
                cost_before_1 = local_cost[block_1 - 1];
                work_before_1 = local_work[block_1 - 1];
            }
            unsigned int _max_13 = ((target_0 - cost_before_1) > (1) ? (target_0 - cost_before_1) : (1));
            unsigned int _min_12 = ((_max_13 - 1) < (local_work[block_1] - work_before_1 - 1) ? (_max_13 - 1) : (local_work[block_1] - work_before_1 - 1));
            split_2 = _min_12;
            coordinate_3 = work_before_1 + split_2;
        }
        local_header[0] = block;
        local_header[1] = split;
        local_header[2] = coordinate_3 - coordinate;
        ScheduleMeta[2 * bid] = block;
        ScheduleMeta[2 * bid + 1] = split;
        ScheduleMeta[2 * SM_COUNT + bid] = coordinate_3 - coordinate;
    }
    __syncthreads();
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: math ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // math_main
            unsigned int wg_idx = make_warp_uniform(warp / 4);
            unsigned int math_q_stage = 0;
            unsigned int math_tmem_stage = wg_idx;
            unsigned int first_q = bid;
            unsigned int q_step = num_bids;
            unsigned int split_offset = 0;
            unsigned int remaining = 0;
            first_q = local_header[0];
            split_offset = local_header[1];
            remaining = local_header[2];
            q_step = 1;
            unsigned int first_q_0 = first_q;
            unsigned int q_step_1 = q_step;
            unsigned int split_offset_2 = split_offset;
            unsigned int remaining_3 = remaining;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_umma_full = 0;
            #pragma unroll 1
            for (unsigned int q_block_idx = first_q_0; q_block_idx < num_q_blocks; q_block_idx += q_step_1) {
                if (remaining_3 == 0) {
                    break;
                }
                int q_start = q_block_idx * 4;
                int kv_start = 0;
                unsigned int num_kv_blocks = 0;
                kv_start = local_base[q_block_idx] + split_offset_2 * 256;
                unsigned int span_splits = local_splits[q_block_idx];
                if (split_offset_2 < span_splits) {
                    unsigned int _min_0_2 = ((span_splits - split_offset_2) < (remaining_3) ? (span_splits - split_offset_2) : (remaining_3));
                    num_kv_blocks = _min_0_2;
                }
                remaining_3 -= num_kv_blocks;
                split_offset_2 = 0;
                if (num_kv_blocks > 0) {
                    mbarrier_wait(q_full_addr + (math_q_stage) * 8, _phase_q_full);
                    float weights_reg[128];
                    int wsmem = smem_weights_addr + math_q_stage * 512;
                    #pragma unroll
                    for (int wi = 0; wi < 32; wi++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[wi * 4])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(wi * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(wi * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(wi * 4) + 3]))
                            : "r"(wsmem + wi * 16));
                    }
                    int seq_start[4];
                    int seq_end[4];
                    #pragma unroll
                    for (int qi_bound = 0; qi_bound < 4; qi_bound++) {
                        int bound_row = ((q_start + qi_bound < seq_len) ? q_start + qi_bound : seq_len - 1);
                        int raw_start = cu_seq_len_k_start[bound_row];
                        int raw_end = cu_seq_len_k_end[bound_row];
                        seq_start[qi_bound] = ((raw_start < seq_len_kv) ? raw_start : seq_len_kv);
                        seq_end[qi_bound] = ((raw_end < seq_len_kv) ? raw_end : seq_len_kv);
                    }
                    #pragma unroll 1
                    for (unsigned int kv_iter = 0; kv_iter < num_kv_blocks; kv_iter++) {
                        mbarrier_wait(umma_full_addr + (math_tmem_stage) * 8, _phase_umma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int kv_pos = (unsigned int)kv_start + kv_iter * 256 + wg_idx * 128 + (unsigned int)(warp % 4 * 32 + lane);
                        float _tmem_load_0[32];
                        tmem_ld_x16(&_tmem_load_0[0], taddr + math_tmem_stage * 128 + (unsigned int)(warp % 4 * 32 << 16));
                        tmem_ld_x16(&_tmem_load_0[16], taddr + math_tmem_stage * 128 + (unsigned int)(warp % 4 * 32 << 16) + 16);
                        float _tmem_load_1[32];
                        tmem_ld_x16(&_tmem_load_1[0], taddr + math_tmem_stage * 128 + 32 + (unsigned int)(warp % 4 * 32 << 16));
                        tmem_ld_x16(&_tmem_load_1[16], taddr + math_tmem_stage * 128 + 32 + (unsigned int)(warp % 4 * 32 << 16) + 16);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float _tmem_load_2[32];
                        tmem_ld_x16(&_tmem_load_2[0], taddr + math_tmem_stage * 128 + 64 + (unsigned int)(warp % 4 * 32 << 16));
                        tmem_ld_x16(&_tmem_load_2[16], taddr + math_tmem_stage * 128 + 64 + (unsigned int)(warp % 4 * 32 << 16) + 16);
                        float _tmem_load_3[32];
                        tmem_ld_x16(&_tmem_load_3[0], taddr + math_tmem_stage * 128 + 96 + (unsigned int)(warp % 4 * 32 << 16));
                        tmem_ld_x16(&_tmem_load_3[16], taddr + math_tmem_stage * 128 + 96 + (unsigned int)(warp % 4 * 32 << 16) + 16);
                        float2 sum0 = make_float2(0.0f, 0.0f);
                        float2 sum1 = make_float2(0.0f, 0.0f);
                        #pragma unroll
                        for (int j = 0; j < 32; j += 4) {
                            asm("{\n\t.reg .f32 _ab0, _ab1;\n\t.reg .b64 _pr, _pa, _pb;\n\t"
                                "abs.f32 _ab0, %1;\n\tabs.f32 _ab1, %2;\n\t"
                                "mov.b64 _pr, {%1, %2};\n\tmov.b64 _pa, {_ab0, _ab1};\n\t"
                                "add.rn.f32x2 _pr, _pr, _pa;\n\t"
                                "mov.b64 _pb, {%3, %4};\n\t"
                                "fma.rn.f32x2 %0, _pr, _pb, %0;\n\t}"
                                : "+l"(*(unsigned long long*)&sum0) : "f"(_tmem_load_0[j]), "f"(_tmem_load_0[j + 1]), "f"(weights_reg[j]), "f"(weights_reg[j + 1]));
                            asm("{\n\t.reg .f32 _ab0, _ab1;\n\t.reg .b64 _pr, _pa, _pb;\n\t"
                                "abs.f32 _ab0, %1;\n\tabs.f32 _ab1, %2;\n\t"
                                "mov.b64 _pr, {%1, %2};\n\tmov.b64 _pa, {_ab0, _ab1};\n\t"
                                "add.rn.f32x2 _pr, _pr, _pa;\n\t"
                                "mov.b64 _pb, {%3, %4};\n\t"
                                "fma.rn.f32x2 %0, _pr, _pb, %0;\n\t}"
                                : "+l"(*(unsigned long long*)&sum1) : "f"(_tmem_load_0[j + 2]), "f"(_tmem_load_0[j + 3]), "f"(weights_reg[j + 2]), "f"(weights_reg[j + 3]));
                        }
                        float2 total = add_f32x2_noftz(sum0, sum1);
                        float relu_wsum = (total.x + total.y) * 0.5f;
                        float weighted_sum = relu_wsum;
                        int q_row = q_start;
                        unsigned long long q_offset = (unsigned long long)q_row * (unsigned long long)stride_logits;
                        unsigned long long out_elem = q_offset + (unsigned long long)kv_pos;
                        weighted_sum = ((kv_pos >= seq_start[0] && kv_pos < seq_end[0]) ? weighted_sum : -CUDART_INF_F);
                        *(reinterpret_cast<float*>(Logits + out_elem) + (0)) = weighted_sum;
                        float2 sum0_0 = make_float2(0.0f, 0.0f);
                        float2 sum1_1 = make_float2(0.0f, 0.0f);
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 32; j_1 += 4) {
                            asm("{\n\t.reg .f32 _ab0, _ab1;\n\t.reg .b64 _pr, _pa, _pb;\n\t"
                                "abs.f32 _ab0, %1;\n\tabs.f32 _ab1, %2;\n\t"
                                "mov.b64 _pr, {%1, %2};\n\tmov.b64 _pa, {_ab0, _ab1};\n\t"
                                "add.rn.f32x2 _pr, _pr, _pa;\n\t"
                                "mov.b64 _pb, {%3, %4};\n\t"
                                "fma.rn.f32x2 %0, _pr, _pb, %0;\n\t}"
                                : "+l"(*(unsigned long long*)&sum0_0) : "f"(_tmem_load_1[j_1]), "f"(_tmem_load_1[j_1 + 1]), "f"(weights_reg[32 + j_1]), "f"(weights_reg[32 + j_1 + 1]));
                            asm("{\n\t.reg .f32 _ab0, _ab1;\n\t.reg .b64 _pr, _pa, _pb;\n\t"
                                "abs.f32 _ab0, %1;\n\tabs.f32 _ab1, %2;\n\t"
                                "mov.b64 _pr, {%1, %2};\n\tmov.b64 _pa, {_ab0, _ab1};\n\t"
                                "add.rn.f32x2 _pr, _pr, _pa;\n\t"
                                "mov.b64 _pb, {%3, %4};\n\t"
                                "fma.rn.f32x2 %0, _pr, _pb, %0;\n\t}"
                                : "+l"(*(unsigned long long*)&sum1_1) : "f"(_tmem_load_1[j_1 + 2]), "f"(_tmem_load_1[j_1 + 3]), "f"(weights_reg[32 + j_1 + 2]), "f"(weights_reg[32 + j_1 + 3]));
                        }
                        float2 total_2 = add_f32x2_noftz(sum0_0, sum1_1);
                        float relu_wsum_3 = (total_2.x + total_2.y) * 0.5f;
                        float weighted_sum_4 = relu_wsum_3;
                        int q_row_5 = q_start + 1;
                        unsigned long long q_offset_6 = (unsigned long long)q_row_5 * (unsigned long long)stride_logits;
                        unsigned long long out_elem_7 = q_offset_6 + (unsigned long long)kv_pos;
                        weighted_sum_4 = ((kv_pos >= seq_start[1] && kv_pos < seq_end[1]) ? weighted_sum_4 : -CUDART_INF_F);
                        *(reinterpret_cast<float*>(Logits + out_elem_7) + (0)) = weighted_sum_4;
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(umma_empty_addr + (math_tmem_stage) * 8);
                        float2 sum0_8 = make_float2(0.0f, 0.0f);
                        float2 sum1_9 = make_float2(0.0f, 0.0f);
                        #pragma unroll
                        for (int j_2 = 0; j_2 < 32; j_2 += 4) {
                            asm("{\n\t.reg .f32 _ab0, _ab1;\n\t.reg .b64 _pr, _pa, _pb;\n\t"
                                "abs.f32 _ab0, %1;\n\tabs.f32 _ab1, %2;\n\t"
                                "mov.b64 _pr, {%1, %2};\n\tmov.b64 _pa, {_ab0, _ab1};\n\t"
                                "add.rn.f32x2 _pr, _pr, _pa;\n\t"
                                "mov.b64 _pb, {%3, %4};\n\t"
                                "fma.rn.f32x2 %0, _pr, _pb, %0;\n\t}"
                                : "+l"(*(unsigned long long*)&sum0_8) : "f"(_tmem_load_2[j_2]), "f"(_tmem_load_2[j_2 + 1]), "f"(weights_reg[64 + j_2]), "f"(weights_reg[64 + j_2 + 1]));
                            asm("{\n\t.reg .f32 _ab0, _ab1;\n\t.reg .b64 _pr, _pa, _pb;\n\t"
                                "abs.f32 _ab0, %1;\n\tabs.f32 _ab1, %2;\n\t"
                                "mov.b64 _pr, {%1, %2};\n\tmov.b64 _pa, {_ab0, _ab1};\n\t"
                                "add.rn.f32x2 _pr, _pr, _pa;\n\t"
                                "mov.b64 _pb, {%3, %4};\n\t"
                                "fma.rn.f32x2 %0, _pr, _pb, %0;\n\t}"
                                : "+l"(*(unsigned long long*)&sum1_9) : "f"(_tmem_load_2[j_2 + 2]), "f"(_tmem_load_2[j_2 + 3]), "f"(weights_reg[64 + j_2 + 2]), "f"(weights_reg[64 + j_2 + 3]));
                        }
                        float2 total_10 = add_f32x2_noftz(sum0_8, sum1_9);
                        float relu_wsum_11 = (total_10.x + total_10.y) * 0.5f;
                        float weighted_sum_12 = relu_wsum_11;
                        int q_row_13 = q_start + 2;
                        unsigned long long q_offset_14 = (unsigned long long)q_row_13 * (unsigned long long)stride_logits;
                        unsigned long long out_elem_15 = q_offset_14 + (unsigned long long)kv_pos;
                        weighted_sum_12 = ((kv_pos >= seq_start[2] && kv_pos < seq_end[2]) ? weighted_sum_12 : -CUDART_INF_F);
                        *(reinterpret_cast<float*>(Logits + out_elem_15) + (0)) = weighted_sum_12;
                        float2 sum0_16 = make_float2(0.0f, 0.0f);
                        float2 sum1_17 = make_float2(0.0f, 0.0f);
                        #pragma unroll
                        for (int j_3 = 0; j_3 < 32; j_3 += 4) {
                            asm("{\n\t.reg .f32 _ab0, _ab1;\n\t.reg .b64 _pr, _pa, _pb;\n\t"
                                "abs.f32 _ab0, %1;\n\tabs.f32 _ab1, %2;\n\t"
                                "mov.b64 _pr, {%1, %2};\n\tmov.b64 _pa, {_ab0, _ab1};\n\t"
                                "add.rn.f32x2 _pr, _pr, _pa;\n\t"
                                "mov.b64 _pb, {%3, %4};\n\t"
                                "fma.rn.f32x2 %0, _pr, _pb, %0;\n\t}"
                                : "+l"(*(unsigned long long*)&sum0_16) : "f"(_tmem_load_3[j_3]), "f"(_tmem_load_3[j_3 + 1]), "f"(weights_reg[96 + j_3]), "f"(weights_reg[96 + j_3 + 1]));
                            asm("{\n\t.reg .f32 _ab0, _ab1;\n\t.reg .b64 _pr, _pa, _pb;\n\t"
                                "abs.f32 _ab0, %1;\n\tabs.f32 _ab1, %2;\n\t"
                                "mov.b64 _pr, {%1, %2};\n\tmov.b64 _pa, {_ab0, _ab1};\n\t"
                                "add.rn.f32x2 _pr, _pr, _pa;\n\t"
                                "mov.b64 _pb, {%3, %4};\n\t"
                                "fma.rn.f32x2 %0, _pr, _pb, %0;\n\t}"
                                : "+l"(*(unsigned long long*)&sum1_17) : "f"(_tmem_load_3[j_3 + 2]), "f"(_tmem_load_3[j_3 + 3]), "f"(weights_reg[96 + j_3 + 2]), "f"(weights_reg[96 + j_3 + 3]));
                        }
                        float2 total_18 = add_f32x2_noftz(sum0_16, sum1_17);
                        float relu_wsum_19 = (total_18.x + total_18.y) * 0.5f;
                        float weighted_sum_20 = relu_wsum_19;
                        int q_row_21 = q_start + 3;
                        unsigned long long q_offset_22 = (unsigned long long)q_row_21 * (unsigned long long)stride_logits;
                        unsigned long long out_elem_23 = q_offset_22 + (unsigned long long)kv_pos;
                        weighted_sum_20 = ((kv_pos >= seq_start[3] && kv_pos < seq_end[3]) ? weighted_sum_20 : -CUDART_INF_F);
                        *(reinterpret_cast<float*>(Logits + out_elem_23) + (0)) = weighted_sum_20;
                        math_tmem_stage += 1;
                        if (math_tmem_stage == 3) { math_tmem_stage = 0; _phase_umma_full ^= 1; }
                        math_tmem_stage += 1;
                        if (math_tmem_stage == 3) { math_tmem_stage = 0; _phase_umma_full ^= 1; }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(q_empty_addr + (math_q_stage) * 8);
                    math_q_stage += 1;
                    if (math_q_stage == 3) { math_q_stage = 0; _phase_q_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_q ----
    if (warp == 8) {
        { // load_q_main
            unsigned int load_q_stage = 0;
            unsigned int _phase_q_empty = 1;
            if (elect_sync()) {
                unsigned int first_q_1 = bid;
                unsigned int q_step_2 = num_bids;
                unsigned int split_offset_1 = 0;
                unsigned int remaining_1 = 0;
                first_q_1 = local_header[0];
                split_offset_1 = local_header[1];
                remaining_1 = local_header[2];
                q_step_2 = 1;
                unsigned int first_q_0_1 = first_q_1;
                unsigned int q_step_1_1 = q_step_2;
                unsigned int split_offset_2_1 = split_offset_1;
                unsigned int remaining_3_1 = remaining_1;
                #pragma unroll 1
                for (unsigned int q_block_idx_1 = first_q_0_1; q_block_idx_1 < num_q_blocks; q_block_idx_1 += q_step_1_1) {
                    if (remaining_3_1 == 0) {
                        break;
                    }
                    int q_start_1 = q_block_idx_1 * 4;
                    int kv_start_1 = 0;
                    unsigned int num_kv_blocks_1 = 0;
                    kv_start_1 = local_base[q_block_idx_1] + split_offset_2_1 * 256;
                    unsigned int span_splits_1 = local_splits[q_block_idx_1];
                    if (split_offset_2_1 < span_splits_1) {
                        unsigned int _min_0_3 = ((span_splits_1 - split_offset_2_1) < (remaining_3_1) ? (span_splits_1 - split_offset_2_1) : (remaining_3_1));
                        num_kv_blocks_1 = _min_0_3;
                    }
                    remaining_3_1 -= num_kv_blocks_1;
                    split_offset_2_1 = 0;
                    if (num_kv_blocks_1 > 0) {
                        mbarrier_wait(q_empty_addr + (load_q_stage) * 8, _phase_q_empty);
                        tma_2d_gmem2smem(smem_q_fp4_addr + load_q_stage * 8192, (&Q), 0, q_block_idx_1 * 128, q_full_addr + (load_q_stage) * 8);
                        tma_2d_gmem2smem(smem_sf_q_addr + load_q_stage * 512, (&SF_Q), 0, q_block_idx_1 * 32, q_full_addr + (load_q_stage) * 8);
                        tma_2d_gmem2smem(smem_weights_addr + load_q_stage * 512, (&Weights), 0, q_block_idx_1 * 4, q_full_addr + (load_q_stage) * 8);
                        mbarrier_arrive_expect_tx(q_full_addr + (load_q_stage) * 8, 9216);
                        load_q_stage += 1;
                        if (load_q_stage == 3) { load_q_stage = 0; _phase_q_empty ^= 1; }
                    }
                }
            }
            __syncwarp();
        }
    }
    // ---- Role: load_kv ----
    if (warp == 9) {
        { // load_kv_main
            unsigned int load_kv_stage = 0;
            unsigned int first_q_2 = bid;
            unsigned int q_step_3 = num_bids;
            unsigned int split_offset_3 = 0;
            unsigned int remaining_2 = 0;
            first_q_2 = local_header[0];
            split_offset_3 = local_header[1];
            remaining_2 = local_header[2];
            q_step_3 = 1;
            unsigned int first_q_0_2 = first_q_2;
            unsigned int q_step_1_2 = q_step_3;
            unsigned int split_offset_2_2 = split_offset_3;
            unsigned int remaining_3_2 = remaining_2;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int q_block_idx_2 = first_q_0_2; q_block_idx_2 < num_q_blocks; q_block_idx_2 += q_step_1_2) {
                if (remaining_3_2 == 0) {
                    break;
                }
                int q_start_2 = q_block_idx_2 * 4;
                int kv_start_2 = 0;
                unsigned int num_kv_blocks_2 = 0;
                kv_start_2 = local_base[q_block_idx_2] + split_offset_2_2 * 256;
                unsigned int span_splits_2 = local_splits[q_block_idx_2];
                if (split_offset_2_2 < span_splits_2) {
                    unsigned int _min_0_4 = ((span_splits_2 - split_offset_2_2) < (remaining_3_2) ? (span_splits_2 - split_offset_2_2) : (remaining_3_2));
                    num_kv_blocks_2 = _min_0_4;
                }
                remaining_3_2 -= num_kv_blocks_2;
                split_offset_2_2 = 0;
                if (num_kv_blocks_2 > 0) {
                    #pragma unroll 1
                    for (unsigned int kv_iter_1 = 0; kv_iter_1 < num_kv_blocks_2; kv_iter_1++) {
                        int kv_row = (unsigned int)kv_start_2 + kv_iter_1 * 256;
                        if (elect_sync()) {
                            mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                            tma_2d_gmem2smem(smem_kv_fp4_addr + load_kv_stage * 16384, (&KV), 0, kv_row, kv_full_addr + (load_kv_stage) * 8);
                            tma_2d_gmem2smem(smem_sf_kv_addr + load_kv_stage * 1024, (&SF_KV), 0, kv_row / 4, kv_full_addr + (load_kv_stage) * 8);
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 17408);
                            load_kv_stage += 1;
                            if (load_kv_stage == 10) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                        }
                        __syncwarp();
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 10) {
        { // mma_main
            unsigned int mma_q_stage = 0;
            unsigned int mma_kv_stage = 0;
            unsigned int mma_tmem_stage = 0;
            bool _elect_sync_0 = elect_sync();
            unsigned int first_q_3 = bid;
            unsigned int q_step_4 = num_bids;
            unsigned int split_offset_4 = 0;
            unsigned int remaining_4 = 0;
            first_q_3 = local_header[0];
            split_offset_4 = local_header[1];
            remaining_4 = local_header[2];
            q_step_4 = 1;
            unsigned int first_q_0_3 = first_q_3;
            unsigned int q_step_1_3 = q_step_4;
            unsigned int split_offset_2_3 = split_offset_4;
            unsigned int remaining_3_3 = remaining_4;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_umma_empty = 1;
            #pragma unroll 1
            for (unsigned int q_block_idx_3 = first_q_0_3; q_block_idx_3 < num_q_blocks; q_block_idx_3 += q_step_1_3) {
                if (remaining_3_3 == 0) {
                    break;
                }
                int q_start_3 = q_block_idx_3 * 4;
                int kv_start_3 = 0;
                unsigned int num_kv_blocks_3 = 0;
                kv_start_3 = local_base[q_block_idx_3] + split_offset_2_3 * 256;
                unsigned int span_splits_3 = local_splits[q_block_idx_3];
                if (split_offset_2_3 < span_splits_3) {
                    unsigned int _min_0_5 = ((span_splits_3 - split_offset_2_3) < (remaining_3_3) ? (span_splits_3 - split_offset_2_3) : (remaining_3_3));
                    num_kv_blocks_3 = _min_0_5;
                }
                remaining_3_3 -= num_kv_blocks_3;
                split_offset_2_3 = 0;
                if (num_kv_blocks_3 > 0) {
                    mbarrier_wait(q_full_addr + (mma_q_stage) * 8, _phase_q_full_1);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    unsigned int _sf_v[4];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[0])) : "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)(((lane >> 3 ^ 0) * 32 + lane) * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[1])) : "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)(((lane >> 3 ^ 1) * 32 + lane) * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[2])) : "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)(((lane >> 3 ^ 2) * 32 + lane) * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[3])) : "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)(((lane >> 3 ^ 3) * 32 + lane) * 4)));
                    __syncwarp();
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 0)) * 4)), "r"(_sf_v[0]));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 1)) * 4)), "r"(_sf_v[1]));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 2)) * 4)), "r"(_sf_v[2]));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_q_addr + mma_q_stage * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 3)) * 4)), "r"(_sf_v[3]));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    tcgen05_cp_32x128b_warpx4_leader(tmem_tmem_sf_q, make_sf_cp_desc_lo_sbo128((((smem_sf_q_addr) >> 4) + (mma_q_stage) * 32)), (uint32_t)(_elect_sync_0));
                    __syncwarp();
                    #pragma unroll 1
                    for (unsigned int kv_iter_2 = 0; kv_iter_2 < num_kv_blocks_3; kv_iter_2++) {
                        mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                        int sfkv_smem = smem_sf_kv_addr + mma_kv_stage * 1024;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        unsigned int _sf_v_0[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[0])) : "r"(sfkv_smem + ((lane >> 3 ^ 0) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[1])) : "r"(sfkv_smem + ((lane >> 3 ^ 1) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[2])) : "r"(sfkv_smem + ((lane >> 3 ^ 2) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[3])) : "r"(sfkv_smem + ((lane >> 3 ^ 3) * 32 + lane) * 4));
                        __syncwarp();
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + (lane * 4 + (lane >> 3 ^ 0)) * 4), "r"(_sf_v_0[0]));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + (lane * 4 + (lane >> 3 ^ 1)) * 4), "r"(_sf_v_0[1]));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + (lane * 4 + (lane >> 3 ^ 2)) * 4), "r"(_sf_v_0[2]));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + (lane * 4 + (lane >> 3 ^ 3)) * 4), "r"(_sf_v_0[3]));
                        unsigned int _sf_v_1[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[0])) : "r"(sfkv_smem + 512 + ((lane >> 3 ^ 0) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[1])) : "r"(sfkv_smem + 512 + ((lane >> 3 ^ 1) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[2])) : "r"(sfkv_smem + 512 + ((lane >> 3 ^ 2) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[3])) : "r"(sfkv_smem + 512 + ((lane >> 3 ^ 3) * 32 + lane) * 4));
                        __syncwarp();
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + 512 + (lane * 4 + (lane >> 3 ^ 0)) * 4), "r"(_sf_v_1[0]));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + 512 + (lane * 4 + (lane >> 3 ^ 1)) * 4), "r"(_sf_v_1[1]));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + 512 + (lane * 4 + (lane >> 3 ^ 2)) * 4), "r"(_sf_v_1[2]));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfkv_smem + 512 + (lane * 4 + (lane >> 3 ^ 3)) * 4), "r"(_sf_v_1[3]));
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (_elect_sync_0) {
                            tcgen05_cp_32x128b_warpx4(tmem_tmem_sf_kv, make_sf_cp_desc_lo_sbo128((((smem_sf_kv_addr) >> 4) + (mma_kv_stage) * 64)));
                            tcgen05_cp_32x128b_warpx4((tmem_tmem_sf_kv + 4), make_sf_cp_desc_lo_sbo128((((smem_sf_kv_addr) >> 4) + (mma_kv_stage) * 64 + 32)));
                            mbarrier_wait(umma_empty_addr + (mma_tmem_stage) * 8, _phase_umma_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _smem_desc_lo_1 = (((smem_kv_fp4_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 1024;
                            int _smem_desc_lo_2 = (((smem_q_fp4_addr) >> 4) & 0x3FFF) + (mma_q_stage) * 512;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mma_tmem_stage * 128)), a_desc + 0, b_desc + 0,
                                    0x8a00480U, tmem_tmem_sf_kv + 0, tmem_tmem_sf_q + 0, 0);
                                tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mma_tmem_stage * 128)), a_desc + 2, b_desc + 2,
                                    0x48a004a0U, tmem_tmem_sf_kv + 0, tmem_tmem_sf_q + 0, 1);
                            }
                            tcgen05_commit(umma_full_addr + (mma_tmem_stage) * 8);
                            mma_tmem_stage += 1;
                            if (mma_tmem_stage == 3) { mma_tmem_stage = 0; _phase_umma_empty ^= 1; }
                            mbarrier_wait(umma_empty_addr + (mma_tmem_stage) * 8, _phase_umma_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _smem_desc_lo_3 = (((smem_kv_fp4_addr + 8192) >> 4) & 0x3FFF) + (mma_kv_stage) * 1024;
                            int _smem_desc_lo_4 = (((smem_q_fp4_addr) >> 4) & 0x3FFF) + (mma_q_stage) * 512;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mma_tmem_stage * 128)), a_desc + 0, b_desc + 0,
                                    0x8a00480U, tmem_tmem_sf_kv + 4 + 0, tmem_tmem_sf_q + 0, 0);
                                tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mma_tmem_stage * 128)), a_desc + 2, b_desc + 2,
                                    0x48a004a0U, tmem_tmem_sf_kv + 4 + 0, tmem_tmem_sf_q + 0, 1);
                            }
                            tcgen05_commit(umma_full_addr + (mma_tmem_stage) * 8);
                            mma_tmem_stage += 1;
                            if (mma_tmem_stage == 3) { mma_tmem_stage = 0; _phase_umma_empty ^= 1; }
                        }
                        __syncwarp();
                        elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                        mma_kv_stage += 1;
                        if (mma_kv_stage == 10) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                    }
                    tcgen05_commit_leader(q_empty_addr + (mma_q_stage) * 8, (uint32_t)(_elect_sync_0));
                    __syncwarp();
                    mma_q_stage += 1;
                    if (mma_q_stage == 3) { mma_q_stage = 0; _phase_q_full_1 ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 11) {
        { // idle_main
            float neg_inf[4];
            #pragma unroll
            for (int component = 0; component < 4; component++) {
                neg_inf[component] = -CUDART_INF_F;
            }
            for (int scratch_j = lane; scratch_j < 1024; scratch_j += 32) {
                neginf_scratch[scratch_j] = -CUDART_INF_F;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            unsigned int clean_tiles_per_row = (stride_logits + 4095) / 4096;
            unsigned int clean_tasks = (unsigned int)num_q_blocks * clean_tiles_per_row;
            #pragma unroll 1
            for (unsigned int clean_task = bid; clean_task < clean_tasks; clean_task += SM_COUNT) {
                unsigned int q_block_idx_4 = clean_task / clean_tiles_per_row;
                int clean_begin = clean_task % clean_tiles_per_row * 4096;
                int clean_end_raw = clean_begin + 4096;
                int clean_end = ((clean_end_raw) < (stride_logits) ? (clean_end_raw) : (stride_logits));
                int q_start_4 = q_block_idx_4 * 4;
                int ks0 = cu_seq_len_k_start[q_start_4];
                int ke0 = cu_seq_len_k_end[q_start_4];
                int kv_start_acc = ((ks0 < seq_len_kv) ? ks0 : seq_len_kv);
                int kv_end_acc = ((ke0 < seq_len_kv) ? ke0 : seq_len_kv);
                int row_idx = ((q_start_4 + 1 < seq_len) ? q_start_4 + 1 : seq_len - 1);
                int ks = cu_seq_len_k_start[row_idx];
                int ke = cu_seq_len_k_end[row_idx];
                int sk = ((ks < seq_len_kv) ? ks : seq_len_kv);
                int se = ((ke < seq_len_kv) ? ke : seq_len_kv);
                kv_start_acc = ((sk < kv_start_acc) ? sk : kv_start_acc);
                kv_end_acc = ((se > kv_end_acc) ? se : kv_end_acc);
                int row_idx_0 = ((q_start_4 + 2 < seq_len) ? q_start_4 + 2 : seq_len - 1);
                int ks_1 = cu_seq_len_k_start[row_idx_0];
                int ke_2 = cu_seq_len_k_end[row_idx_0];
                int sk_3 = ((ks_1 < seq_len_kv) ? ks_1 : seq_len_kv);
                int se_4 = ((ke_2 < seq_len_kv) ? ke_2 : seq_len_kv);
                kv_start_acc = ((sk_3 < kv_start_acc) ? sk_3 : kv_start_acc);
                kv_end_acc = ((se_4 > kv_end_acc) ? se_4 : kv_end_acc);
                int row_idx_5 = ((q_start_4 + 3 < seq_len) ? q_start_4 + 3 : seq_len - 1);
                int ks_6 = cu_seq_len_k_start[row_idx_5];
                int ke_7 = cu_seq_len_k_end[row_idx_5];
                int sk_8 = ((ks_6 < seq_len_kv) ? ks_6 : seq_len_kv);
                int se_9 = ((ke_7 < seq_len_kv) ? ke_7 : seq_len_kv);
                kv_start_acc = ((sk_8 < kv_start_acc) ? sk_8 : kv_start_acc);
                kv_end_acc = ((se_9 > kv_end_acc) ? se_9 : kv_end_acc);
                int kv_start_4 = kv_start_acc / 4 * 4;
                unsigned int num_kv_blocks_4 = (kv_end_acc - kv_start_4 + 256 - 1) / 256;
                int raw_end_1 = (unsigned int)kv_start_4 + num_kv_blocks_4 * 256;
                int coverage_end = ((raw_end_1 < stride_logits) ? raw_end_1 : stride_logits);
                #pragma unroll 1
                for (int qi_3 = 0; qi_3 < 4; qi_3++) {
                    unsigned long long row_base = (unsigned long long)(q_start_4 + qi_3) * (unsigned long long)stride_logits;
                    int _min_0_6 = ((kv_start_4) < (clean_end) ? (kv_start_4) : (clean_end));
                    int aligned_start = (clean_begin + 3) / 4 * 4;
                    int aligned_end = _min_0_6 / 4 * 4;
                    if (aligned_start >= aligned_end) {
                        for (int j_4 = clean_begin + lane; j_4 < _min_0_6; j_4 += 32) {
                            *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_4)) + (0)) = -CUDART_INF_F;
                        }
                        __syncwarp();
                    } else {
                        for (int j_5 = clean_begin + lane; j_5 < aligned_start; j_5 += 32) {
                            *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_5)) + (0)) = -CUDART_INF_F;
                        }
                        for (int j_6 = aligned_end + lane; j_6 < _min_0_6; j_6 += 32) {
                            *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_6)) + (0)) = -CUDART_INF_F;
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            for (int j_7 = aligned_start; j_7 < aligned_end; j_7 += 1024) {
                                int bulk_elems = ((aligned_end - j_7) < (1024) ? (aligned_end - j_7) : (1024));
                                {
                                    void* _cpbulk_dst_0 = reinterpret_cast<void*>(Logits + (row_base + (unsigned long long)j_7));
                                    asm volatile(
                                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                        :: "l"(_cpbulk_dst_0), "r"(neginf_scratch_addr), "r"((uint32_t)(bulk_elems * 4))
                                        : "memory");
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        __syncwarp();
                    }
                    int _max_0 = ((coverage_end) > (clean_begin) ? (coverage_end) : (clean_begin));
                    int aligned_start_0 = (_max_0 + 3) / 4 * 4;
                    int aligned_end_1 = clean_end / 4 * 4;
                    if (aligned_start_0 >= aligned_end_1) {
                        for (int j_8 = _max_0 + lane; j_8 < clean_end; j_8 += 32) {
                            *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_8)) + (0)) = -CUDART_INF_F;
                        }
                        __syncwarp();
                    } else {
                        for (int j_9 = _max_0 + lane; j_9 < aligned_start_0; j_9 += 32) {
                            *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_9)) + (0)) = -CUDART_INF_F;
                        }
                        for (int j_10 = aligned_end_1 + lane; j_10 < clean_end; j_10 += 32) {
                            *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_10)) + (0)) = -CUDART_INF_F;
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            for (int j_11 = aligned_start_0; j_11 < aligned_end_1; j_11 += 1024) {
                                int bulk_elems_1 = ((aligned_end_1 - j_11) < (1024) ? (aligned_end_1 - j_11) : (1024));
                                {
                                    void* _cpbulk_dst_1 = reinterpret_cast<void*>(Logits + (row_base + (unsigned long long)j_11));
                                    asm volatile(
                                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                        :: "l"(_cpbulk_dst_1), "r"(neginf_scratch_addr), "r"((uint32_t)(bulk_elems_1 * 4))
                                        : "memory");
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        __syncwarp();
                    }
                }
            }
            if (elect_sync()) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            __syncwarp();
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 10) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
