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
#define TMEM_NCOLS 512
#define TMEM_TMEM_OFFSET 0
#define NUM_KV_PIPE_STAGES 4
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_KV_OFF 66560
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 66560
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_V_KEPT_OFF 66560
#define SMEM_SMEM_V_KEPT_STAGE_BYTES 16384
#define SMEM_SMEM_V_KEPT_STRIDE 16384
#define SMEM_SMEM_SUM_OFF 197632
#define SMEM_SMEM_SUM_STAGE_BYTES 256
#define SMEM_SMEM_SUM_STRIDE 256
#define SMEM_SMEM_ALPHA_OFF 198448
#define SMEM_SMEM_ALPHA_STAGE_BYTES 256
#define SMEM_SMEM_ALPHA_STRIDE 256
#define SMEM_SMEM_INDEX_FLAGS_OFF 198400
#define SMEM_SMEM_INDEX_FLAGS_STAGE_BYTES 32
#define SMEM_SMEM_INDEX_FLAGS_STRIDE 32
#define SMEM_SMEM_MERGED_FLAGS_OFF 198432
#define SMEM_SMEM_MERGED_FLAGS_STAGE_BYTES 8
#define SMEM_SMEM_MERGED_FLAGS_STRIDE 8
#define SMEM_SMEM_INDICES_OFF 197888
#define SMEM_SMEM_INDICES_STAGE_BYTES 512
#define SMEM_SMEM_INDICES_STRIDE 512
#define SMEM_TOTAL 198784
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

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
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


__device__ __forceinline__ void tma_gather4_gmem2smem(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr) {
    // Non-multicast, single-CTA gather4 in the .shared::cta window (the
    // cuda::ptx form): the destination and the barrier are the executing
    // CTA's. A CTA pair's gather takes the .shared::cluster window
    // (tma_gather4_gmem2smem_cta2), its barrier being the pair leader's.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsv4_750835d54fe97cfc7ad4(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_swa_kv, __nv_bfloat16* __restrict__ O, int* __restrict__ swa_indices, int* __restrict__ compressed_indices, int* __restrict__ sparse_topk_lens, int* __restrict__ seq_lens, int* __restrict__ cum_seq_lens_q, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int swa_index_stride, int compressed_index_stride, int sparse_topk_lens_offset, int sparse_topk, int has_sinks, int ragged_query, int max_q_len, int batch_size)
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
    #define kv_full_addr (mbar_base + 8)
    #define kv_empty_addr (mbar_base + 40)
    #define s0_full_addr (mbar_base + 72)
    #define s1_full_addr (mbar_base + 80)
    #define p0_full_addr (mbar_base + 88)
    #define p1_full_addr (mbar_base + 96)
    #define sum_ready_addr (mbar_base + 104)
    #define o_done_addr (mbar_base + 112)
    #define tmem_dealloc_addr (mbar_base + 120)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_kv_addr = smem + 66560;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_v_addr = smem + 66560;
    __nv_bfloat16* smem_v_kept = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_v_kept_addr = smem + 66560;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + 197632);
    const int smem_sum_addr = smem + 197632;
    float* smem_alpha = reinterpret_cast<float*>(smem_raw + 198448);
    const int smem_alpha_addr = smem + 198448;
    int* smem_index_flags = reinterpret_cast<int*>(smem_raw + 198400);
    const int smem_index_flags_addr = smem + 198400;
    int* smem_merged_flags = reinterpret_cast<int*>(smem_raw + 198432);
    const int smem_merged_flags_addr = smem + 198432;
    int* smem_indices = reinterpret_cast<int*>(smem_raw + 197888);
    const int smem_indices_addr = smem + 197888;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_swa_kv))) : "memory");

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        // q_full: 1 barriers, init_count=1
        // --- pipeline 'kv_pipe' ---
        // kv_full: 4 barriers, init_count=4
        // kv_empty: 4 barriers, init_count=1
        // s0_full: 1 barriers, init_count=1
        // s1_full: 1 barriers, init_count=1
        // p0_full: 1 barriers, init_count=64
        // p1_full: 1 barriers, init_count=64
        // sum_ready: 1 barriers, init_count=64
        // o_done: 1 barriers, init_count=1
        // tmem_dealloc: 1 barriers, init_count=128
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 128;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(15), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(14), "r"((uint32_t)(64)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(11), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(5), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(1), "r"((uint32_t)(1)));
        if (lane < 16) {
            mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
    if (warp == 3) {
        int _tmem_hold = smem + 128;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem = taddr;
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
    } else if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;");
    }

    // ---- Role: softmax ----
    if (warp <= 1) {
        { // softmax_main
            float softmax_scale_log2 = bmm1_scale[0] * 1.442695f;
            int query_idx = blockIdx.x >> 2;
            int _max_0 = ((sparse_topk_lens[query_idx] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_idx] + sparse_topk_lens_offset) : (0));
            int active_topk = ((_max_0) < (sparse_topk) ? (_max_0) : (sparse_topk));
            int query_batch = query_idx / max_q_len;
            int query_offset = query_idx - query_batch * max_q_len;
            int query_length = max_q_len;
            if (ragged_query != 0) {
                query_batch = 0;
                #pragma unroll 2
                for (int chunk = 0; chunk < (batch_size + 31) / 32; chunk++) {
                    int lane_entry = chunk * 32 + lane + 1;
                    int lane_load = ((lane_entry) < (batch_size) ? (lane_entry) : (batch_size));
                    unsigned int started = __ballot_sync(0xFFFFFFFF, lane_entry <= batch_size && query_idx >= cum_seq_lens_q[lane_load]);
                    query_batch = query_batch + (int)(unsigned int)__popc(started);
                }
                int query_begin = cum_seq_lens_q[query_batch];
                query_length = cum_seq_lens_q[query_batch + 1] - query_begin;
                query_offset = query_idx - query_begin;
            }
            int visible = seq_lens[query_batch] - query_length + query_offset + 1;
            if (visible < 0) {
                visible = 0;
            }
            if (visible > 128) {
                visible = 128;
            }
            int swa_visible = visible;
            if (active_topk > swa_visible) {
                active_topk = swa_visible;
            }
            const int warp_in_compute = warp;
            const int tmem_row_origin = warp_in_compute * 32;
            const int logical_row_origin = warp_in_compute * 16;
            const int my_row = logical_row_origin + lane % 16;
            const int col_half = lane / 16;
            int score_addr = taddr + (unsigned int)(tmem_row_origin << 16);
            int p_addr = taddr + 128 + (unsigned int)(tmem_row_origin << 16);
            float2 _f2_0 = make_float2(softmax_scale_log2, softmax_scale_log2);
            int row_bound1 = active_topk - 64;
            unsigned int _phase_s0_full_0 = 0;
            mbarrier_wait(s0_full_addr, _phase_s0_full_0);
            _phase_s0_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 32;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                : "r"(score_addr));
            int causal0 = smem_merged_flags[0];
            int valid0 = ((active_topk < causal0) ? active_topk : causal0);
            valid0 = valid0 - col_half * 32;
            uint32_t _thresh_mask_1;
            {
                int _lim_2 = valid0;
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
            if (!(_thresh_mask_1 & (1u << 0))) _tmem_load_0[0] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 1))) _tmem_load_0[1] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 2))) _tmem_load_0[2] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 3))) _tmem_load_0[3] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 4))) _tmem_load_0[4] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 5))) _tmem_load_0[5] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 6))) _tmem_load_0[6] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 7))) _tmem_load_0[7] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 8))) _tmem_load_0[8] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 9))) _tmem_load_0[9] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 10))) _tmem_load_0[10] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 11))) _tmem_load_0[11] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 12))) _tmem_load_0[12] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 13))) _tmem_load_0[13] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 14))) _tmem_load_0[14] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 15))) _tmem_load_0[15] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 16))) _tmem_load_0[16] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 17))) _tmem_load_0[17] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 18))) _tmem_load_0[18] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 19))) _tmem_load_0[19] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 20))) _tmem_load_0[20] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 21))) _tmem_load_0[21] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 22))) _tmem_load_0[22] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 23))) _tmem_load_0[23] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 24))) _tmem_load_0[24] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 25))) _tmem_load_0[25] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 26))) _tmem_load_0[26] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 27))) _tmem_load_0[27] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 28))) _tmem_load_0[28] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 29))) _tmem_load_0[29] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 30))) _tmem_load_0[30] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 31))) _tmem_load_0[31] = -CUDART_INF_F;
            float2 _reg_reduce_max2_3 = {-CAKE_INF, -CAKE_INF};
            _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_0[0], _tmem_load_0[1]));
            _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_0[2], _tmem_load_0[3]));
            _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_0[4], _tmem_load_0[5]));
            _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_0[6], _tmem_load_0[7]));
            _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_0[8], _tmem_load_0[9]));
            _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_0[10], _tmem_load_0[11]));
            _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_0[12], _tmem_load_0[13]));
            _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_0[14], _tmem_load_0[15]));
            _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_0[16], _tmem_load_0[17]));
            _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_0[18], _tmem_load_0[19]));
            _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_0[20], _tmem_load_0[21]));
            _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_0[22], _tmem_load_0[23]));
            _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_0[24], _tmem_load_0[25]));
            _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_0[26], _tmem_load_0[27]));
            _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_0[28], _tmem_load_0[29]));
            _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_0[30], _tmem_load_0[31]));
            float m0 = row_max_reduce(_reg_reduce_max2_3);
            float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, m0, 16);
            m0 = max_noftz(m0, _shfl_xor_sync_4);
            if (has_sinks != 0 && my_row < num_heads) {
                float sink_unscaled = sinks[my_row] * 1.442695f / softmax_scale_log2;
                m0 = max_noftz(m0, sink_unscaled);
            }
            bool empty0 = m0 == -CUDART_INF_F;
            float safe0 = ((empty0) ? 0.0f : m0);
            float max0_scaled = safe0 * softmax_scale_log2;
            float2 _f2_1 = make_float2(-max0_scaled, -max0_scaled);
            float2 sum0 = make_float2(0.0f, 0.0f);
            float2 sum1 = make_float2(0.0f, 0.0f);
            float2 sum2 = make_float2(0.0f, 0.0f);
            float2 sum3 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int i = 0; i < 32; i += 8) {
                float2 _f2_2 = make_float2(_tmem_load_0[i], _tmem_load_0[i + 1]);
                float2 _f2_3 = make_float2(_tmem_load_0[i + 2], _tmem_load_0[i + 3]);
                float2 _f2_4 = make_float2(_tmem_load_0[i + 4], _tmem_load_0[i + 5]);
                float2 _f2_5 = make_float2(_tmem_load_0[i + 6], _tmem_load_0[i + 7]);
                float2 affine0 = fma_f32x2_rn_ftz(_f2_2, _f2_0, _f2_1);
                float2 affine1 = fma_f32x2_rn_ftz(_f2_3, _f2_0, _f2_1);
                float2 affine2 = fma_f32x2_rn_ftz(_f2_4, _f2_0, _f2_1);
                float2 affine3 = fma_f32x2_rn_ftz(_f2_5, _f2_0, _f2_1);
                float exp0 = approx_exp2(affine0.x);
                float exp1 = approx_exp2(affine0.y);
                float exp2 = approx_exp2(affine1.x);
                float exp3 = approx_exp2(affine1.y);
                float exp4 = approx_exp2(affine2.x);
                float exp5 = approx_exp2(affine2.y);
                float exp6 = approx_exp2(affine3.x);
                float exp7 = approx_exp2(affine3.y);
                _tmem_load_0[i] = exp0;
                _tmem_load_0[i + 1] = exp1;
                _tmem_load_0[i + 2] = exp2;
                _tmem_load_0[i + 3] = exp3;
                _tmem_load_0[i + 4] = exp4;
                _tmem_load_0[i + 5] = exp5;
                _tmem_load_0[i + 6] = exp6;
                _tmem_load_0[i + 7] = exp7;
                sum0 = add_f32x2(sum0, make_float2(exp0, exp1));
                sum1 = add_f32x2(sum1, make_float2(exp2, exp3));
                sum2 = add_f32x2(sum2, make_float2(exp4, exp5));
                sum3 = add_f32x2(sum3, make_float2(exp6, exp7));
            }
            float2 sum01 = add_f32x2(sum0, sum1);
            float2 sum23 = add_f32x2(sum2, sum3);
            float2 l0_pair = add_f32x2(sum01, sum23);
            float l0 = l0_pair.x + l0_pair.y;
            unsigned int packed_p0[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                packed_p0[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                " [%0], 16, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p0[15])));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(p0_full_addr);
            unsigned int _phase_s1_full_0 = 0;
            mbarrier_wait(s1_full_addr, _phase_s1_full_0);
            _phase_s1_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_1[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 32;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                : "r"(score_addr + 64));
            int causal1 = smem_merged_flags[1];
            int valid1 = ((row_bound1 < causal1) ? row_bound1 : causal1);
            valid1 = valid1 - col_half * 32;
            uint32_t _thresh_mask_5;
            {
                int _lim_6 = valid1;
                if (_lim_6 <= 0) { _thresh_mask_5 = 0u; }
                else if (_lim_6 >= 32) { _thresh_mask_5 = 0xFFFFFFFFu; }
                else {
                    asm volatile("{"
                        ".reg .u32 t;\n\t"
                        "shl.b32 t, 1, %1;\n\t"
                        "add.u32 %0, t, -1;\n\t"
                        "}" : "=r"(_thresh_mask_5) : "r"(_lim_6));
                }
            }
            if (!(_thresh_mask_5 & (1u << 0))) _tmem_load_1[0] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 1))) _tmem_load_1[1] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 2))) _tmem_load_1[2] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 3))) _tmem_load_1[3] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 4))) _tmem_load_1[4] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 5))) _tmem_load_1[5] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 6))) _tmem_load_1[6] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 7))) _tmem_load_1[7] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 8))) _tmem_load_1[8] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 9))) _tmem_load_1[9] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 10))) _tmem_load_1[10] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 11))) _tmem_load_1[11] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 12))) _tmem_load_1[12] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 13))) _tmem_load_1[13] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 14))) _tmem_load_1[14] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 15))) _tmem_load_1[15] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 16))) _tmem_load_1[16] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 17))) _tmem_load_1[17] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 18))) _tmem_load_1[18] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 19))) _tmem_load_1[19] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 20))) _tmem_load_1[20] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 21))) _tmem_load_1[21] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 22))) _tmem_load_1[22] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 23))) _tmem_load_1[23] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 24))) _tmem_load_1[24] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 25))) _tmem_load_1[25] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 26))) _tmem_load_1[26] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 27))) _tmem_load_1[27] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 28))) _tmem_load_1[28] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 29))) _tmem_load_1[29] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 30))) _tmem_load_1[30] = -CUDART_INF_F;
            if (!(_thresh_mask_5 & (1u << 31))) _tmem_load_1[31] = -CUDART_INF_F;
            float2 _reg_reduce_max2_7 = {-CAKE_INF, -CAKE_INF};
            _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(_tmem_load_1[0], _tmem_load_1[1]));
            _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(_tmem_load_1[2], _tmem_load_1[3]));
            _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(_tmem_load_1[4], _tmem_load_1[5]));
            _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(_tmem_load_1[6], _tmem_load_1[7]));
            _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(_tmem_load_1[8], _tmem_load_1[9]));
            _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(_tmem_load_1[10], _tmem_load_1[11]));
            _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(_tmem_load_1[12], _tmem_load_1[13]));
            _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(_tmem_load_1[14], _tmem_load_1[15]));
            _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(_tmem_load_1[16], _tmem_load_1[17]));
            _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(_tmem_load_1[18], _tmem_load_1[19]));
            _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(_tmem_load_1[20], _tmem_load_1[21]));
            _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(_tmem_load_1[22], _tmem_load_1[23]));
            _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(_tmem_load_1[24], _tmem_load_1[25]));
            _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(_tmem_load_1[26], _tmem_load_1[27]));
            _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(_tmem_load_1[28], _tmem_load_1[29]));
            _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(_tmem_load_1[30], _tmem_load_1[31]));
            float m1 = row_max_reduce(_reg_reduce_max2_7);
            float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, m1, 16);
            m1 = max_noftz(m1, _shfl_xor_sync_8);
            float row_max;
            asm("max.f32 %0, %1, %2;" : "=f"(row_max) : "f"(m0), "f"(m1));
            bool empty_row = row_max == -CUDART_INF_F;
            float safe_max = ((empty_row) ? 0.0f : row_max);
            float max_scaled = safe_max * softmax_scale_log2;
            float alpha0_raw = approx_exp2(max0_scaled - max_scaled);
            float alpha0 = ((empty0) ? 0.0f : alpha0_raw);
            float2 _f2_6 = make_float2(-max_scaled, -max_scaled);
            sum0 = make_float2(0.0f, 0.0f);
            sum1 = make_float2(0.0f, 0.0f);
            sum2 = make_float2(0.0f, 0.0f);
            sum3 = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int i_1 = 0; i_1 < 32; i_1 += 8) {
                float2 _f2_7 = make_float2(_tmem_load_1[i_1], _tmem_load_1[i_1 + 1]);
                float2 _f2_8 = make_float2(_tmem_load_1[i_1 + 2], _tmem_load_1[i_1 + 3]);
                float2 _f2_9 = make_float2(_tmem_load_1[i_1 + 4], _tmem_load_1[i_1 + 5]);
                float2 _f2_10 = make_float2(_tmem_load_1[i_1 + 6], _tmem_load_1[i_1 + 7]);
                float2 affine0_1 = fma_f32x2_rn_ftz(_f2_7, _f2_0, _f2_6);
                float2 affine1_1 = fma_f32x2_rn_ftz(_f2_8, _f2_0, _f2_6);
                float2 affine2_1 = fma_f32x2_rn_ftz(_f2_9, _f2_0, _f2_6);
                float2 affine3_1 = fma_f32x2_rn_ftz(_f2_10, _f2_0, _f2_6);
                float exp0_1 = approx_exp2(affine0_1.x);
                float exp1_1 = approx_exp2(affine0_1.y);
                float exp2_1 = approx_exp2(affine1_1.x);
                float exp3_1 = approx_exp2(affine1_1.y);
                float exp4_1 = approx_exp2(affine2_1.x);
                float exp5_1 = approx_exp2(affine2_1.y);
                float exp6_1 = approx_exp2(affine3_1.x);
                float exp7_1 = approx_exp2(affine3_1.y);
                _tmem_load_1[i_1] = exp0_1;
                _tmem_load_1[i_1 + 1] = exp1_1;
                _tmem_load_1[i_1 + 2] = exp2_1;
                _tmem_load_1[i_1 + 3] = exp3_1;
                _tmem_load_1[i_1 + 4] = exp4_1;
                _tmem_load_1[i_1 + 5] = exp5_1;
                _tmem_load_1[i_1 + 6] = exp6_1;
                _tmem_load_1[i_1 + 7] = exp7_1;
                sum0 = add_f32x2(sum0, make_float2(exp0_1, exp1_1));
                sum1 = add_f32x2(sum1, make_float2(exp2_1, exp3_1));
                sum2 = add_f32x2(sum2, make_float2(exp4_1, exp5_1));
                sum3 = add_f32x2(sum3, make_float2(exp6_1, exp7_1));
            }
            sum01 = add_f32x2(sum0, sum1);
            sum23 = add_f32x2(sum2, sum3);
            float2 l1_pair = add_f32x2(sum01, sum23);
            float l1 = l1_pair.x + l1_pair.y;
            float row_sum = __fmaf_rn(l0, alpha0, l1);
            float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 16);
            row_sum = row_sum + _shfl_xor_sync_9;
            if (has_sinks != 0 && my_row < num_heads) {
                float _exp2_10 = approx_exp2(__fmaf_rn(sinks[my_row], 1.442695f, -max_scaled));
                row_sum = row_sum + _exp2_10;
            }
            unsigned int packed_p1[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                packed_p1[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                " [%0], 16, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                :: "r"(p_addr + 32), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p1[15])));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(p1_full_addr);
            if (col_half == 0) {
                smem_sum[my_row] = row_sum;
                smem_alpha[my_row] = alpha0;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(sum_ready_addr);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 5) {
        { // epilogue_main
            float output_scale = bmm2_scale[0];
            int work_idx = blockIdx.x;
            int query_idx_1 = work_idx >> 2;
            int v_chunk = work_idx & 3;
            const int warp_in_role = warp - 4;
            const int tmem_row_origin_1 = warp_in_role * 32;
            const int logical_row_origin_1 = warp_in_role * 16;
            const int my_row_1 = logical_row_origin_1 + lane % 16;
            const int col_half_1 = lane / 16;
            const int row_addr = tmem_row_origin_1 << 16;
            unsigned int _phase_o_done_0 = 0;
            mbarrier_wait(o_done_addr, _phase_o_done_0);
            _phase_o_done_0 ^= 1;
            unsigned int _phase_sum_ready_0 = 0;
            mbarrier_wait(sum_ready_addr, _phase_sum_ready_0);
            _phase_sum_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float row_sum_total = smem_sum[my_row_1];
            float alpha0_1 = smem_alpha[my_row_1];
            float _rcp_0 = approx_rcp(row_sum_total);
            float inv_sum = ((row_sum_total == 0.0f) ? 0.0f : _rcp_0);
            int output_base = (query_idx_1 * num_heads + my_row_1) * 512 + v_chunk * 128;
            float _tmem_load_2[64];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                : "r"(taddr + 256 + (unsigned int)row_addr));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[63]))
                : "r"(taddr + 256 + (unsigned int)row_addr + 32));
            float _tmem_load_3[64];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                : "r"(taddr + 384 + (unsigned int)row_addr));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[63]))
                : "r"(taddr + 384 + (unsigned int)row_addr + 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
            float2 _f2_11 = make_float2(alpha0_1, alpha0_1);
            #pragma unroll
            for (int i_2 = 0; i_2 < 64; i_2 += 2) {
                float2 merged2 = fma_f32x2_rn_noftz(make_float2(_tmem_load_2[i_2], _tmem_load_2[i_2 + 1]), _f2_11, make_float2(_tmem_load_3[i_2], _tmem_load_3[i_2 + 1]));
                _tmem_load_3[i_2] = merged2.x;
                _tmem_load_3[i_2 + 1] = merged2.y;
            }
            int col_base = col_half_1 * 64;
            if (my_row_1 < num_heads) {
                #pragma unroll
                for (int offset = 0; offset < 64; offset += 8) {
                    float _scaled_0[8];
                    const float2 _scale2_1 = {inv_sum * output_scale, inv_sum * output_scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((_tmem_load_3 + offset))[_ls], _scale2_1);
                    unsigned int _packed_0[4];
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
                    reinterpret_cast<int4*>(O + (output_base + col_base + offset))[0] = reinterpret_cast<int4*>(_packed_0)[0];
                }
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 2) {
        { // mma_warp_main
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            mbarrier_wait(kv_full_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512);
            int _mma_b_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (0) * 2048);
            {
                uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, 1);
                }
            }
            int _mma_a_lo_1 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (1) * 512);
            int _mma_b_lo_1 = make_warp_uniform((((smem_kv_addr + 8192) >> 4) & 0x3FFF) + (0) * 2048);
            {
                uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 68158608, 1);
                }
            }
            int _mma_a_lo_2 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (2) * 512);
            int _mma_b_lo_2 = make_warp_uniform((((smem_kv_addr + 16384) >> 4) & 0x3FFF) + (0) * 2048);
            {
                uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 68158608, 1);
                }
            }
            int _mma_a_lo_3 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (3) * 512);
            int _mma_b_lo_3 = make_warp_uniform((((smem_kv_addr + 24576) >> 4) & 0x3FFF) + (0) * 2048);
            {
                uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 68158608, 1);
                }
            }
            elect_commit(kv_empty_addr);
            mbarrier_wait(kv_full_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_4 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (4) * 512);
            int _mma_b_lo_4 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (1) * 2048);
            {
                uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 68158608, 1);
                }
            }
            int _mma_a_lo_5 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (5) * 512);
            int _mma_b_lo_5 = make_warp_uniform((((smem_kv_addr + 8192) >> 4) & 0x3FFF) + (1) * 2048);
            {
                uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_5);
                uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_5);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 68158608, 1);
                }
            }
            int _mma_a_lo_6 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (6) * 512);
            int _mma_b_lo_6 = make_warp_uniform((((smem_kv_addr + 16384) >> 4) & 0x3FFF) + (1) * 2048);
            {
                uint64_t _mma_ss_a_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_6);
                uint64_t _mma_ss_b_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_6);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 68158608, 1);
                }
            }
            int _mma_a_lo_7 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (7) * 512);
            int _mma_b_lo_7 = make_warp_uniform((((smem_kv_addr + 24576) >> 4) & 0x3FFF) + (1) * 2048);
            {
                uint64_t _mma_ss_a_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_7);
                uint64_t _mma_ss_b_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_7);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 68158608, 1);
                }
            }
            elect_commit(s0_full_addr);
            elect_commit(kv_empty_addr + 8);
            mbarrier_wait(kv_full_addr + 16, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_8 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512);
            int _mma_b_lo_8 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (2) * 2048);
            {
                uint64_t _mma_ss_a_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_8);
                uint64_t _mma_ss_b_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_8);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_8, _mma_ss_b_desc_8, 68158608, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_8, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_8, _mma_ss_b_desc_8, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_8, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_8, _mma_ss_b_desc_8, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_8, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_8, _mma_ss_b_desc_8, 68158608, 1);
                }
            }
            int _mma_a_lo_9 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (1) * 512);
            int _mma_b_lo_9 = make_warp_uniform((((smem_kv_addr + 8192) >> 4) & 0x3FFF) + (2) * 2048);
            {
                uint64_t _mma_ss_a_desc_9 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_9);
                uint64_t _mma_ss_b_desc_9 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_9);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_9, _mma_ss_b_desc_9, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_9, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_9, _mma_ss_b_desc_9, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_9, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_9, _mma_ss_b_desc_9, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_9, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_9, _mma_ss_b_desc_9, 68158608, 1);
                }
            }
            int _mma_a_lo_10 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (2) * 512);
            int _mma_b_lo_10 = make_warp_uniform((((smem_kv_addr + 16384) >> 4) & 0x3FFF) + (2) * 2048);
            {
                uint64_t _mma_ss_a_desc_10 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_10);
                uint64_t _mma_ss_b_desc_10 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_10);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_10, _mma_ss_b_desc_10, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_10, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_10, _mma_ss_b_desc_10, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_10, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_10, _mma_ss_b_desc_10, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_10, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_10, _mma_ss_b_desc_10, 68158608, 1);
                }
            }
            int _mma_a_lo_11 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (3) * 512);
            int _mma_b_lo_11 = make_warp_uniform((((smem_kv_addr + 24576) >> 4) & 0x3FFF) + (2) * 2048);
            {
                uint64_t _mma_ss_a_desc_11 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_11);
                uint64_t _mma_ss_b_desc_11 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_11);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_11, _mma_ss_b_desc_11, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_11, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_11, _mma_ss_b_desc_11, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_11, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_11, _mma_ss_b_desc_11, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_11, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_11, _mma_ss_b_desc_11, 68158608, 1);
                }
            }
            elect_commit(kv_empty_addr + 16);
            mbarrier_wait(kv_full_addr + 24, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_12 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (4) * 512);
            int _mma_b_lo_12 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (3) * 2048);
            {
                uint64_t _mma_ss_a_desc_12 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_12);
                uint64_t _mma_ss_b_desc_12 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_12);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_12, _mma_ss_b_desc_12, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_12, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_12, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_12, _mma_ss_b_desc_12, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_12, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_12, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_12, _mma_ss_b_desc_12, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_12, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_12, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_12, _mma_ss_b_desc_12, 68158608, 1);
                }
            }
            int _mma_a_lo_13 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (5) * 512);
            int _mma_b_lo_13 = make_warp_uniform((((smem_kv_addr + 8192) >> 4) & 0x3FFF) + (3) * 2048);
            {
                uint64_t _mma_ss_a_desc_13 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_13);
                uint64_t _mma_ss_b_desc_13 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_13);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_13, _mma_ss_b_desc_13, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_13, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_13, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_13, _mma_ss_b_desc_13, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_13, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_13, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_13, _mma_ss_b_desc_13, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_13, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_13, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_13, _mma_ss_b_desc_13, 68158608, 1);
                }
            }
            int _mma_a_lo_14 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (6) * 512);
            int _mma_b_lo_14 = make_warp_uniform((((smem_kv_addr + 16384) >> 4) & 0x3FFF) + (3) * 2048);
            {
                uint64_t _mma_ss_a_desc_14 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_14);
                uint64_t _mma_ss_b_desc_14 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_14);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_14, _mma_ss_b_desc_14, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_14, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_14, _mma_ss_b_desc_14, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_14, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_14, _mma_ss_b_desc_14, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_14, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_14, _mma_ss_b_desc_14, 68158608, 1);
                }
            }
            int _mma_a_lo_15 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (7) * 512);
            int _mma_b_lo_15 = make_warp_uniform((((smem_kv_addr + 24576) >> 4) & 0x3FFF) + (3) * 2048);
            {
                uint64_t _mma_ss_a_desc_15 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_15);
                uint64_t _mma_ss_b_desc_15 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_15);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_15, _mma_ss_b_desc_15, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_15, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_15, _mma_ss_b_desc_15, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_15, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_15, _mma_ss_b_desc_15, 68158608, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_15, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_tmem + (64)), _mma_ss_a_desc_15, _mma_ss_b_desc_15, 68158608, 1);
                }
            }
            elect_commit(s1_full_addr);
            elect_commit(kv_empty_addr + 24);
            const int v_slot = blockIdx.x & 3;
            unsigned int _phase_p0_full_0 = 0;
            mbarrier_wait(p0_full_addr, _phase_p0_full_0);
            _phase_p0_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_b_lo_16 = make_warp_uniform(((((smem_v_kept_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_slot) * 1024);
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
                    "mov.b32 id, 69272720;\n\t"
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
                    :: "r"((tmem_tmem + (256))), "r"(_mma_b_lo_16), "r"(tmem_tmem + 128), "r"(0) : "memory");
            unsigned int _phase_p1_full_0 = 0;
            mbarrier_wait(p1_full_addr, _phase_p1_full_0);
            _phase_p1_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_b_lo_17 = make_warp_uniform(((((smem_v_kept_addr) >> 4) & 0x3FFF) | 0x2000000) + (4 + v_slot) * 1024);
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
                    "mov.b32 id, 69272720;\n\t"
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
                    :: "r"((tmem_tmem + (384))), "r"(_mma_b_lo_17), "r"(tmem_tmem + 160), "r"(0) : "memory");
            elect_commit(o_done_addr);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
    // ---- Role: empty ----
    if (warp == 3 || warp == 6 || warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: load_warp ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
        { // load_warp_main
            const int load_warp_rank = warp - 8;
            int work_idx_1 = blockIdx.x;
            int query_idx_2 = work_idx_1 >> 2;
            int v_chunk_1 = work_idx_1 & 3;
            int* row_ptr = swa_indices + (query_idx_2 * swa_index_stride);
            int index_offset = load_warp_rank * 16 + lane % 16 + lane / 16 * 64;
            int sparse_row = -1;
            if (index_offset < sparse_topk) {
                sparse_row = row_ptr[index_offset];
            }
            if (load_warp_rank == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 65536);
                    #pragma unroll
                    for (int q_stage = 0; q_stage < 8; q_stage++) {
                        tma_4d_gmem2smem(smem_q_addr + (unsigned int)(q_stage * 8192), (&tmap_q), 0, 0, q_stage, query_idx_2, q_full_addr);
                    }
                }
            }
            smem_indices[index_offset] = sparse_row;
            unsigned int invalid_indices = __ballot_sync(0xFFFFFFFF, sparse_row < 0);
            unsigned int invalid_lo = invalid_indices & 65535;
            unsigned int invalid_hi = invalid_indices >> 16;
            int valid_prefix_lo = 16;
            if (invalid_lo != 0) {
                int _find_first_set_0 = __ffs(invalid_lo);
                valid_prefix_lo = _find_first_set_0 - 1;
            }
            int valid_prefix_hi = 16;
            if (invalid_hi != 0) {
                int _find_first_set_1 = __ffs(invalid_hi);
                valid_prefix_hi = _find_first_set_1 - 1;
            }
            if (elect_sync()) {
                smem_index_flags[load_warp_rank] = valid_prefix_lo;
                smem_index_flags[4 + load_warp_rank] = valid_prefix_hi;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if ((load_warp_rank & 1) != 0) {
                if (elect_sync()) {
                    int run_half = load_warp_rank / 2;
                    int run_base = run_half * 4;
                    int run0 = smem_index_flags[run_base];
                    int run1 = smem_index_flags[run_base + 1];
                    int run2 = smem_index_flags[run_base + 2];
                    int run3 = smem_index_flags[run_base + 3];
                    int prefix23 = ((run2 < 16) ? run2 : 16 + run3);
                    int prefix123 = ((run1 < 16) ? run1 : 16 + prefix23);
                    int merged_prefix = ((run0 < 16) ? run0 : 16 + prefix123);
                    smem_merged_flags[run_half] = merged_prefix;
                }
            }
            #pragma unroll
            for (int k_stage = 0; k_stage < 4; k_stage++) {
                int k_half = k_stage / 2;
                int k_dim_base = k_stage % 2 * 256;
                mbarrier_wait(kv_empty_addr + (k_stage) * 8, 1);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (k_stage) * 8, 8192);
                }
                #pragma unroll
                for (int local_group = 0; local_group < 4; local_group++) {
                    int dst_k = smem_kv_addr + (unsigned int)(k_stage * 32768);
                    int group = load_warp_rank * 4 + local_group;
                    int key_base = k_half * 64 + group * 4;
                    int raw_rows[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 3]))
                        : "r"(smem_indices_addr + (unsigned int)(key_base * 4)));
                    int raw0 = raw_rows[0];
                    int raw1 = raw_rows[1];
                    int raw2 = raw_rows[2];
                    int raw3 = raw_rows[3];
                    int row0 = ((raw0 >= 0) ? raw0 : 0);
                    int row1 = ((raw1 >= 0) ? raw1 : 0);
                    int row2 = ((raw2 >= 0) ? raw2 : 0);
                    int row3 = ((raw3 >= 0) ? raw3 : 0);
                    if (elect_sync()) {
                        tma_gather4_gmem2smem(dst_k + group * 512, (&tmap_swa_kv), k_dim_base, row0, row1, row2, row3, kv_full_addr + (k_stage) * 8);
                        tma_gather4_gmem2smem(dst_k + 8192 + group * 512, (&tmap_swa_kv), k_dim_base + 64, row0, row1, row2, row3, kv_full_addr + (k_stage) * 8);
                        tma_gather4_gmem2smem(dst_k + 16384 + group * 512, (&tmap_swa_kv), k_dim_base + 128, row0, row1, row2, row3, kv_full_addr + (k_stage) * 8);
                        tma_gather4_gmem2smem(dst_k + 24576 + group * 512, (&tmap_swa_kv), k_dim_base + 192, row0, row1, row2, row3, kv_full_addr + (k_stage) * 8);
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
