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


__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
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
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xFFFFFFFF;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_sage_block_sparse_attention_0c9c3d675b3afcbad4f5(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, const __grid_constant__ CUtensorMap v, __nv_bfloat16* out, float* lse, float* q_scale, float* k_scale, float* v_scale, int* block_index, int* block_nums, int* block_sizes, int seqlen_q, int seqlen_k, int num_heads, int num_kv_heads, int q_blocks, int k_blocks, int k_scale_groups, int capacity, int block_sparse_num, int has_block_nums, int block_sizes_mode, int v_scale_batch_stride, int total_tiles, float softmax_scale_log2, int return_lse)
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
    #define kv_full_addr (mbar_base + 32)
    #define kv_empty_addr (mbar_base + 80)
    #define s_full_addr (mbar_base + 128)
    #define p_full_addr (mbar_base + 144)
    #define corr_sig_addr (mbar_base + 160)
    #define corr_done_addr (mbar_base + 176)
    #define o_full_addr (mbar_base + 192)
    #define tile_done_addr (mbar_base + 200)
    #define meta_full_addr (mbar_base + 208)
    #define meta_empty_addr (mbar_base + 224)
    #define step_full_addr (mbar_base + 240)
    #define step_empty_addr (mbar_base + 304)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    uint8_t* q_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    uint8_t* kv_smem = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int kv_smem_addr = smem + 17408;
    uint8_t* v_smem = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int v_smem_addr = smem + 17408;
    float* scale_smem = reinterpret_cast<float*>(smem_raw + 115712);
    const int scale_smem_addr = smem + 115712;
    float* pf_sink_f = reinterpret_cast<float*>(smem_raw + 118784);
    const int pf_sink_f_addr = smem + 118784;
    int* pf_sink_i = reinterpret_cast<int*>(smem_raw + 118912);
    const int pf_sink_i_addr = smem + 118912;
    float* meta_f = reinterpret_cast<float*>(smem_raw + 119040);
    const int meta_f_addr = smem + 119040;
    int* meta_i = reinterpret_cast<int*>(smem_raw + 119040);
    const int meta_i_addr = smem + 119040;
    float* ring_f = reinterpret_cast<float*>(smem_raw + 119840);
    const int ring_f_addr = smem + 119840;
    int* ring_i = reinterpret_cast<int*>(smem_raw + 119840);
    const int ring_i_addr = smem + 119840;

    // Mbarrier init (46 barriers at smem_raw[0..368))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 6 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // kv_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 144, 256);
            mbarrier_init(smem + 152, 256);
            // corr_sig: 2 barriers, init_count=128
            mbarrier_init(smem + 160, 128);
            mbarrier_init(smem + 168, 128);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            mbarrier_init(smem + 184, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // tile_done: 1 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            // meta_full: 2 barriers, init_count=32
            mbarrier_init(smem + 208, 32);
            mbarrier_init(smem + 216, 32);
            // meta_empty: 2 barriers, init_count=384
            mbarrier_init(smem + 224, 384);
            mbarrier_init(smem + 232, 384);
            // step_full: 8 barriers, init_count=32
            mbarrier_init(smem + 240, 32);
            mbarrier_init(smem + 248, 32);
            mbarrier_init(smem + 256, 32);
            mbarrier_init(smem + 264, 32);
            mbarrier_init(smem + 272, 32);
            mbarrier_init(smem + 280, 32);
            mbarrier_init(smem + 288, 32);
            mbarrier_init(smem + 296, 32);
            // step_empty: 8 barriers, init_count=128
            mbarrier_init(smem + 304, 128);
            mbarrier_init(smem + 312, 128);
            mbarrier_init(smem + 320, 128);
            mbarrier_init(smem + 328, 128);
            mbarrier_init(smem + 336, 128);
            mbarrier_init(smem + 344, 128);
            mbarrier_init(smem + 352, 128);
            mbarrier_init(smem + 360, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 368);
    if (warp == 0) {
        int _tmem_hold = smem + 368;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_scores = taddr;
    const unsigned int tmem_probs_0 = taddr + 64;
    const unsigned int tmem_probs_1 = taddr + 192;
    const unsigned int tmem_output_0 = taddr + 256;
    const unsigned int tmem_output_1 = taddr + 384;

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            int group = make_warp_uniform(warp / 4);
            int tmem_s_off = make_warp_uniform(group * 128);
            int tmem_p_off = make_warp_uniform(group * 128 + 64);
            const int warp_in_group = warp % 4;
            const int tmem_row_origin = warp_in_group * 32;
            int my_row = warp_in_group * 16 + lane % 16;
            int col_half = lane / 16;
            int slot = group * 128 + col_half * 64 + my_row;
            int count = (total_tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
            int my_tiles = count;
            int cons_n = group * 4;
            unsigned int ring_stage = 0;
            int ring_base = 0;
            unsigned int _phase_step_full = 0;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_corr_done = 0;
            #pragma unroll 1
            for (int tile_iter = 0; tile_iter < my_tiles; tile_iter++) {
                unsigned int meta_stage = (unsigned int)(tile_iter % 2);
                int meta_off = tile_iter % 2 * 100;
                int _stage_meta_full = (int)((unsigned int)tile_iter % 2);
                int _parity_meta_full = (int)((unsigned int)tile_iter / 2 % 2);
                mbarrier_wait(meta_full_addr + (_stage_meta_full) * 8, _parity_meta_full);
                int batch = meta_i[meta_off + 88];
                int kv_head = meta_i[meta_off + 88 + 1];
                int k_scale_base = meta_i[meta_off + 88 + 2];
                int q_valid = meta_i[meta_off + 88 + 3];
                int selected_count = meta_i[meta_off + 88 + 4];
                int pair_count = meta_i[meta_off + 88 + 5];
                int index_base = meta_i[meta_off + 88 + 6];
                int row_valid = ((my_row < q_valid) ? 1 : 0);
                float q_log2_scale = 0.0f;
                if (row_valid != 0) {
                    q_log2_scale = meta_f[meta_off + my_row] * softmax_scale_log2;
                }
                if (my_tiles > tile_iter + 2) {
                    mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)tile_iter % 2)) * 8);
                }
                float row_max = -CUDART_INF_F;
                float row_sum = 0.0f;
                #pragma unroll 1
                for (int pair_index = group; pair_index < pair_count; pair_index += 2) {
                    int ring_e = cons_n % 4;
                    ring_stage = (unsigned int)(group * 4 + ring_e);
                    ring_base = (group * 4 + ring_e) * 12;
                    mbarrier_wait(step_full_addr + (cons_n) * 8, _phase_step_full);
                    int m_block = ring_i[ring_base + col_half];
                    int raw_valid = ring_i[ring_base + 2 + col_half];
                    int m_entry = pair_index * 2 + col_half;
                    int m_valid = 0;
                    if (m_entry < selected_count && row_valid != 0) {
                        m_valid = raw_valid;
                    }
                    float my_scales[4];
                    int ring_k = ring_base + 4 + col_half * 4;
                    #pragma unroll
                    for (int scale_group = 0; scale_group < 4; scale_group++) {
                        float qk_scale = q_log2_scale * ring_f[ring_k + scale_group];
                        qk_scale = max_noftz(qk_scale, 1.1754944e-38f);
                        my_scales[scale_group] = qk_scale;
                    }
                    mbarrier_arrive(step_empty_addr + (cons_n) * 8);
                    cons_n += 1;
                    if (cons_n == (group * 4 + 4)) { cons_n = group * 4; _phase_step_full ^= 1; }
                    mbarrier_wait(s_full_addr + (group) * 8, _phase_s_full);
                    _phase_s_full ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int score_addr = taddr + (unsigned int)tmem_s_off + (unsigned int)(tmem_row_origin << 16);
                    float score_values[64];
                    float red_max_0 = -CUDART_INF_F;
                    float red_max_1 = -CUDART_INF_F;
                    float red_max_2 = -CUDART_INF_F;
                    float red_max_3 = -CUDART_INF_F;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.16x32bx2.x16.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, [%17], 64;"
                        : "=f"(score_values[0]), "=f"(score_values[1]), "=f"(score_values[2]), "=f"(score_values[3]), "=f"(score_values[4]), "=f"(score_values[5]), "=f"(score_values[6]), "=f"(score_values[7]), "=f"(score_values[8]), "=f"(score_values[9]), "=f"(score_values[10]), "=f"(score_values[11]), "=f"(score_values[12]), "=f"(score_values[13]), "=f"(score_values[14]), "=f"(score_values[15]), "=f"(red_max_0)
                        : "r"(score_addr));
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.16x32bx2.x16.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, [%17], 64;"
                        : "=f"(score_values[16]), "=f"(score_values[17]), "=f"(score_values[18]), "=f"(score_values[19]), "=f"(score_values[20]), "=f"(score_values[21]), "=f"(score_values[22]), "=f"(score_values[23]), "=f"(score_values[24]), "=f"(score_values[25]), "=f"(score_values[26]), "=f"(score_values[27]), "=f"(score_values[28]), "=f"(score_values[29]), "=f"(score_values[30]), "=f"(score_values[31]), "=f"(red_max_1)
                        : "r"(score_addr + 16));
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.16x32bx2.x16.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, [%17], 64;"
                        : "=f"(score_values[32]), "=f"(score_values[33]), "=f"(score_values[34]), "=f"(score_values[35]), "=f"(score_values[36]), "=f"(score_values[37]), "=f"(score_values[38]), "=f"(score_values[39]), "=f"(score_values[40]), "=f"(score_values[41]), "=f"(score_values[42]), "=f"(score_values[43]), "=f"(score_values[44]), "=f"(score_values[45]), "=f"(score_values[46]), "=f"(score_values[47]), "=f"(red_max_2)
                        : "r"(score_addr + 32));
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.16x32bx2.x16.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, [%17], 64;"
                        : "=f"(score_values[48]), "=f"(score_values[49]), "=f"(score_values[50]), "=f"(score_values[51]), "=f"(score_values[52]), "=f"(score_values[53]), "=f"(score_values[54]), "=f"(score_values[55]), "=f"(score_values[56]), "=f"(score_values[57]), "=f"(score_values[58]), "=f"(score_values[59]), "=f"(score_values[60]), "=f"(score_values[61]), "=f"(score_values[62]), "=f"(score_values[63]), "=f"(red_max_3)
                        : "r"(score_addr + 48));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float chunk_max[4];
                    chunk_max[0] = red_max_0 * my_scales[0];
                    chunk_max[1] = red_max_1 * my_scales[1];
                    chunk_max[2] = red_max_2 * my_scales[2];
                    chunk_max[3] = red_max_3 * my_scales[3];
                    if (m_valid < 64) {
                        uint32_t _thresh_mask_0;
                        {
                            int _lim_1 = m_valid;
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
                            int _lim_3 = m_valid - 32;
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
                        #pragma unroll
                        for (int scale_group_1 = 0; scale_group_1 < 4; scale_group_1++) {
                            float _max3_5;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_5) : "f"(score_values[scale_group_1 * 16]), "f"(score_values[scale_group_1 * 16 + 1]), "f"(score_values[scale_group_1 * 16 + 2]));
                            float _max3_6;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_6) : "f"(score_values[scale_group_1 * 16 + 3]), "f"(score_values[scale_group_1 * 16 + 4]), "f"(score_values[scale_group_1 * 16 + 5]));
                            float _max3_7;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_7) : "f"(score_values[scale_group_1 * 16 + 6]), "f"(score_values[scale_group_1 * 16 + 7]), "f"(score_values[scale_group_1 * 16 + 8]));
                            float _max3_4;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_4) : "f"(_max3_5), "f"(_max3_6), "f"(_max3_7));
                            float _max3_9;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_9) : "f"(score_values[scale_group_1 * 16 + 9]), "f"(score_values[scale_group_1 * 16 + 10]), "f"(score_values[scale_group_1 * 16 + 11]));
                            float _max3_10;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_10) : "f"(score_values[scale_group_1 * 16 + 12]), "f"(score_values[scale_group_1 * 16 + 13]), "f"(score_values[scale_group_1 * 16 + 14]));
                            float _max3_8;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_8) : "f"(_max3_9), "f"(_max3_10), "f"(score_values[scale_group_1 * 16 + 15]));
                            chunk_max[scale_group_1] = max_noftz(_max3_4, _max3_8) * my_scales[scale_group_1];
                        }
                    }
                    float _max3_11;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_11) : "f"(chunk_max[0]), "f"(chunk_max[1]), "f"(chunk_max[2]));
                    float tile_max;
                    asm("max.f32 %0, %1, %2;" : "=f"(tile_max) : "f"(_max3_11), "f"(chunk_max[3]));
                    if (m_valid <= 0) {
                        tile_max = -CUDART_INF_F;
                    }
                    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, tile_max, 16);
                    tile_max = max_noftz(tile_max, _shfl_xor_sync_12);
                    float new_max;
                    asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(tile_max), "f"(row_max));
                    float acc_scale = 1.0f;
                    if (new_max > -CUDART_INF_F) {
                        if (row_max == -CUDART_INF_F) {
                            row_max = new_max;
                        } else if (new_max - row_max > 0.8073549f) {
                            float _exp2_13 = approx_exp2(row_max - new_max);
                            acc_scale = _exp2_13;
                            row_max = new_max;
                        }
                    }
                    scale_smem[slot] = acc_scale;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(corr_sig_addr + (group) * 8);
                    float score_bias = ((row_max > -CUDART_INF_F) ? 8.0f - row_max : 0.0f);
                    #pragma unroll
                    for (int scale_group_2 = 0; scale_group_2 < 4; scale_group_2++) {
                        const float2 _fma_b2_14 = {my_scales[scale_group_2], my_scales[scale_group_2]};
                        const float2 _fma_c2_15 = {score_bias, score_bias};
                        #pragma unroll
                        for (int _lf = 0; _lf < 8; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((score_values + scale_group_2 * 16))[_lf], _fma_b2_14, _fma_c2_15);
                    }
                    unsigned int p_addr = taddr + (unsigned int)tmem_p_off + (unsigned int)(tmem_row_origin << 16);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        score_values[_le] = approx_exp2(score_values[_le]);
                    }
                    float2 _reg_reduce_sum2_16 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&score_values[0], &_reg_reduce_sum2_16);
                    softmax_block_sum(&score_values[32], &_reg_reduce_sum2_16);
                    float block_half = _reg_reduce_sum2_16.x + _reg_reduce_sum2_16.y;
                    float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, block_half, 16);
                    float step_sum = block_half + _shfl_xor_sync_17;
                    unsigned int _fp8_0[16];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[0]), "f"(score_values[1]),
                                               "f"(score_values[2]), "f"(score_values[3]));
                        _fp8_0[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[4]), "f"(score_values[5]),
                                               "f"(score_values[6]), "f"(score_values[7]));
                        _fp8_0[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[8]), "f"(score_values[9]),
                                               "f"(score_values[10]), "f"(score_values[11]));
                        _fp8_0[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[12]), "f"(score_values[13]),
                                               "f"(score_values[14]), "f"(score_values[15]));
                        _fp8_0[3] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[16]), "f"(score_values[17]),
                                               "f"(score_values[18]), "f"(score_values[19]));
                        _fp8_0[4] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[20]), "f"(score_values[21]),
                                               "f"(score_values[22]), "f"(score_values[23]));
                        _fp8_0[5] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[24]), "f"(score_values[25]),
                                               "f"(score_values[26]), "f"(score_values[27]));
                        _fp8_0[6] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[28]), "f"(score_values[29]),
                                               "f"(score_values[30]), "f"(score_values[31]));
                        _fp8_0[7] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[32]), "f"(score_values[33]),
                                               "f"(score_values[34]), "f"(score_values[35]));
                        _fp8_0[8] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[36]), "f"(score_values[37]),
                                               "f"(score_values[38]), "f"(score_values[39]));
                        _fp8_0[9] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[40]), "f"(score_values[41]),
                                               "f"(score_values[42]), "f"(score_values[43]));
                        _fp8_0[10] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[44]), "f"(score_values[45]),
                                               "f"(score_values[46]), "f"(score_values[47]));
                        _fp8_0[11] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[48]), "f"(score_values[49]),
                                               "f"(score_values[50]), "f"(score_values[51]));
                        _fp8_0[12] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[52]), "f"(score_values[53]),
                                               "f"(score_values[54]), "f"(score_values[55]));
                        _fp8_0[13] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[56]), "f"(score_values[57]),
                                               "f"(score_values[58]), "f"(score_values[59]));
                        _fp8_0[14] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(score_values[60]), "f"(score_values[61]),
                                               "f"(score_values[62]), "f"(score_values[63]));
                        _fp8_0[15] = _packed;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 16, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + (group) * 8);
                    mbarrier_wait(corr_done_addr + (group) * 8, _phase_corr_done);
                    _phase_corr_done ^= 1;
                    row_sum = __fmaf_rn(row_sum, acc_scale, step_sum);
                }
                scale_smem[256 + slot] = row_sum;
                scale_smem[512 + slot] = row_max;
                mbarrier_arrive(corr_sig_addr + (group) * 8);
                mbarrier_wait(corr_done_addr + (group) * 8, _phase_corr_done);
                _phase_corr_done ^= 1;
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
        { // correction_main
            const int warp_in_role = warp - 8;
            const int tmem_row_origin_1 = warp_in_role * 32;
            int my_row_1 = warp_in_role * 16 + lane % 16;
            int col_half_1 = lane / 16;
            int row_addr = tmem_row_origin_1 << 16;
            int slot_0 = col_half_1 * 64 + my_row_1;
            int slot_1 = 128 + col_half_1 * 64 + my_row_1;
            int count_1 = (total_tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
            int my_tiles_1 = count_1;
            int corr_tiles_done = 0;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_corr_sig_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
            for (int tile_iter_1 = 0; tile_iter_1 < my_tiles_1; tile_iter_1++) {
                int tile_idx = blockIdx.x + tile_iter_1 * gridDim.x;
                int cm_off = tile_iter_1 % 2 * 100;
                int _stage_meta_full_1 = (int)((unsigned int)corr_tiles_done % 2);
                int _parity_meta_full_1 = (int)((unsigned int)corr_tiles_done / 2 % 2);
                mbarrier_wait(meta_full_addr + (_stage_meta_full_1) * 8, _parity_meta_full_1);
                int batch_1 = meta_i[cm_off + 88];
                int kv_head_1 = meta_i[cm_off + 88 + 1];
                int q_valid_1 = meta_i[cm_off + 88 + 3];
                int pair_count_1 = meta_i[cm_off + 88 + 5];
                int head = meta_i[cm_off + 88 + 7];
                int bh = meta_i[cm_off + 88 + 8];
                int query_base = meta_i[cm_off + 88 + 9];
                if (my_tiles_1 > tile_iter_1 + 2) {
                    mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)corr_tiles_done % 2)) * 8);
                }
                mbarrier_arrive(p_full_addr);
                if (pair_count_1 > 1) {
                    mbarrier_arrive(p_full_addr + 8);
                }
                #pragma unroll 1
                for (int step = 0; step < pair_count_1; step += 2) {
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    if (step >= 2) {
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float acc_scale_1 = scale_smem[slot_0];
                        int _warp_vote_0 = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                        if (_warp_vote_0 != 0) {
                            float tmem_load[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[31]))
                                : "r"(taddr + (unsigned int)256 + (unsigned int)row_addr));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[32])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[33])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[34])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[35])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[36])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[37])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[38])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[39])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[40])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[41])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[42])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[43])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[44])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[45])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[46])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[47])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[48])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[49])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[50])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[51])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[52])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[53])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[54])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[55])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[56])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[57])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[58])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[59])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[60])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[61])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[62])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[63]))
                                : "r"(taddr + (unsigned int)256 + (unsigned int)row_addr + 32));
                            const float2 _scale2_1 = {acc_scale_1, acc_scale_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 32; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(tmem_load)[_ls], _scale2_1);
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                                " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                                :: "r"(taddr + (unsigned int)256 + (unsigned int)row_addr), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[0])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[1])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[2])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[3])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[4])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[5])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[6])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[7])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[8])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[9])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[10])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[11])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[12])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[13])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[14])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[15])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[16])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[17])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[18])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[19])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[20])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[21])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[22])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[23])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[24])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[25])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[26])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[27])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[28])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[29])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[30])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[31])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[32])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[33])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[34])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[35])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[36])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[37])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[38])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[39])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[40])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[41])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[42])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[43])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[44])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[45])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[46])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[47])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[48])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[49])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[50])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[51])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[52])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[53])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[54])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[55])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[56])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[57])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[58])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[59])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[60])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[61])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[62])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load[63])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(p_full_addr);
                    }
                    mbarrier_arrive(corr_done_addr);
                    if (pair_count_1 > step + 1) {
                        mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                        _phase_corr_sig_1 ^= 1;
                        if (step >= 2) {
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            float acc_scale_2 = scale_smem[slot_1];
                            int _warp_vote_2 = __any_sync(0xFFFFFFFF, acc_scale_2 < 1.0f);
                            if (_warp_vote_2 != 0) {
                                float tmem_load_1[64];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[31]))
                                    : "r"(taddr + (unsigned int)384 + (unsigned int)row_addr));
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[63]))
                                    : "r"(taddr + (unsigned int)384 + (unsigned int)row_addr + 32));
                                const float2 _scale2_3 = {acc_scale_2, acc_scale_2};
                                #pragma unroll
                                for (int _ls = 0; _ls < 32; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(tmem_load_1)[_ls], _scale2_3);
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                                    " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                                    :: "r"(taddr + (unsigned int)384 + (unsigned int)row_addr), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[15])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[19])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[23])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[27])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[31])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[32])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[33])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[34])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[35])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[36])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[37])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[38])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[39])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[40])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[41])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[42])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[43])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[44])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[45])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[46])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[47])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[48])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[49])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[50])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[51])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[52])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[53])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[54])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[55])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[56])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[57])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[58])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[59])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[60])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[61])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[62])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1[63])));
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(p_full_addr + 8);
                        }
                        mbarrier_arrive(corr_done_addr + 8);
                    }
                }
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                _phase_corr_sig_1 ^= 1;
                float max_0 = scale_smem[512 + slot_0];
                float sum_0 = scale_smem[256 + slot_0];
                float max_1 = scale_smem[512 + slot_1];
                float sum_1 = scale_smem[256 + slot_1];
                mbarrier_arrive(corr_done_addr);
                mbarrier_arrive(corr_done_addr + 8);
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float row_max_1;
                asm("max.f32 %0, %1, %2;" : "=f"(row_max_1) : "f"(max_0), "f"(max_1));
                float _exp2_4 = approx_exp2(max_0 - row_max_1);
                float merge_0 = ((max_0 > -CUDART_INF_F) ? _exp2_4 : 0.0f);
                float _exp2_5 = approx_exp2(max_1 - row_max_1);
                float merge_1 = ((max_1 > -CUDART_INF_F) ? _exp2_5 : 0.0f);
                float total_sum = __fmaf_rn(sum_0, merge_0, sum_1 * merge_1);
                int has_mass = ((total_sum > 0.0f && total_sum == total_sum) ? 1 : 0);
                float _rcp_6 = approx_rcp(total_sum);
                float inv_sum = ((has_mass != 0) ? _rcp_6 : 0.0f);
                int col_base = col_half_1 * 64;
                int v_scale_base = batch_1 * v_scale_batch_stride + kv_head_1 * 128 + col_base;
                int query = query_base + my_row_1;
                int output_row = ((batch_1 * seqlen_q + query) * num_heads + head) * 128;
                const int chunk_cols_0 = 0;
                float tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[31]))
                    : "r"(taddr + 256 + (unsigned int)row_addr + (unsigned int)chunk_cols_0));
                const float2 _scale2_7 = {merge_0, merge_0};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(tmem_load_2)[_ls], _scale2_7);
                if (pair_count_1 > 1) {
                    float tmem_load_1_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[31]))
                        : "r"(taddr + 256 + 128 + (unsigned int)row_addr + (unsigned int)chunk_cols_0));
                    #pragma unroll
                    for (int item = 0; item < 32; item++) {
                        tmem_load_2[item] = __fmaf_rn(tmem_load_1_1[item], merge_1, tmem_load_2[item]);
                    }
                }
                if (has_mass != 0) {
                    #pragma unroll
                    for (int quad = 0; quad < 8; quad++) {
                        float _vec_load_0[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(v_scale + (v_scale_base + chunk_cols_0 + quad * 4) + 0);
                            _vec_load_0[0 + 0] = _v4.x;
                            _vec_load_0[0 + 1] = _v4.y;
                            _vec_load_0[0 + 2] = _v4.z;
                            _vec_load_0[0 + 3] = _v4.w;
                        }
                        #pragma unroll
                        for (int item_1 = 0; item_1 < 4; item_1++) {
                            tmem_load_2[quad * 4 + item_1] = tmem_load_2[quad * 4 + item_1] * _vec_load_0[item_1];
                        }
                    }
                } else {
                    if (!(0 & (1u << 0))) tmem_load_2[0] = 0.0f;
                    if (!(0 & (1u << 1))) tmem_load_2[1] = 0.0f;
                    if (!(0 & (1u << 2))) tmem_load_2[2] = 0.0f;
                    if (!(0 & (1u << 3))) tmem_load_2[3] = 0.0f;
                    if (!(0 & (1u << 4))) tmem_load_2[4] = 0.0f;
                    if (!(0 & (1u << 5))) tmem_load_2[5] = 0.0f;
                    if (!(0 & (1u << 6))) tmem_load_2[6] = 0.0f;
                    if (!(0 & (1u << 7))) tmem_load_2[7] = 0.0f;
                    if (!(0 & (1u << 8))) tmem_load_2[8] = 0.0f;
                    if (!(0 & (1u << 9))) tmem_load_2[9] = 0.0f;
                    if (!(0 & (1u << 10))) tmem_load_2[10] = 0.0f;
                    if (!(0 & (1u << 11))) tmem_load_2[11] = 0.0f;
                    if (!(0 & (1u << 12))) tmem_load_2[12] = 0.0f;
                    if (!(0 & (1u << 13))) tmem_load_2[13] = 0.0f;
                    if (!(0 & (1u << 14))) tmem_load_2[14] = 0.0f;
                    if (!(0 & (1u << 15))) tmem_load_2[15] = 0.0f;
                    if (!(0 & (1u << 16))) tmem_load_2[16] = 0.0f;
                    if (!(0 & (1u << 17))) tmem_load_2[17] = 0.0f;
                    if (!(0 & (1u << 18))) tmem_load_2[18] = 0.0f;
                    if (!(0 & (1u << 19))) tmem_load_2[19] = 0.0f;
                    if (!(0 & (1u << 20))) tmem_load_2[20] = 0.0f;
                    if (!(0 & (1u << 21))) tmem_load_2[21] = 0.0f;
                    if (!(0 & (1u << 22))) tmem_load_2[22] = 0.0f;
                    if (!(0 & (1u << 23))) tmem_load_2[23] = 0.0f;
                    if (!(0 & (1u << 24))) tmem_load_2[24] = 0.0f;
                    if (!(0 & (1u << 25))) tmem_load_2[25] = 0.0f;
                    if (!(0 & (1u << 26))) tmem_load_2[26] = 0.0f;
                    if (!(0 & (1u << 27))) tmem_load_2[27] = 0.0f;
                    if (!(0 & (1u << 28))) tmem_load_2[28] = 0.0f;
                    if (!(0 & (1u << 29))) tmem_load_2[29] = 0.0f;
                    if (!(0 & (1u << 30))) tmem_load_2[30] = 0.0f;
                    if (!(0 & (1u << 31))) tmem_load_2[31] = 0.0f;
                }
                if (my_row_1 < q_valid_1) {
                    #pragma unroll
                    for (int offset = 0; offset < 32; offset += 8) {
                        float _scaled_0[8];
                        const float2 _scale2_9 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((tmem_load_2 + offset))[_ls], _scale2_9);
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
                        reinterpret_cast<int4*>(out + (output_row + col_base + chunk_cols_0 + offset))[0] = reinterpret_cast<int4*>(_packed_0)[0];
                    }
                }
                const int chunk_cols_1 = 32;
                float tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[31]))
                    : "r"(taddr + 256 + (unsigned int)row_addr + (unsigned int)chunk_cols_1));
                const float2 _scale2_10 = {merge_0, merge_0};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(tmem_load_0)[_ls], _scale2_10);
                if (pair_count_1 > 1) {
                    float tmem_load_1_2[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[31]))
                        : "r"(taddr + 256 + 128 + (unsigned int)row_addr + (unsigned int)chunk_cols_1));
                    #pragma unroll
                    for (int item_2 = 0; item_2 < 32; item_2++) {
                        tmem_load_0[item_2] = __fmaf_rn(tmem_load_1_2[item_2], merge_1, tmem_load_0[item_2]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(tile_done_addr);
                if (has_mass != 0) {
                    #pragma unroll
                    for (int quad_1 = 0; quad_1 < 8; quad_1++) {
                        float _vec_load_1[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(v_scale + (v_scale_base + chunk_cols_1 + quad_1 * 4) + 0);
                            _vec_load_1[0 + 0] = _v4.x;
                            _vec_load_1[0 + 1] = _v4.y;
                            _vec_load_1[0 + 2] = _v4.z;
                            _vec_load_1[0 + 3] = _v4.w;
                        }
                        #pragma unroll
                        for (int item_3 = 0; item_3 < 4; item_3++) {
                            tmem_load_0[quad_1 * 4 + item_3] = tmem_load_0[quad_1 * 4 + item_3] * _vec_load_1[item_3];
                        }
                    }
                } else {
                    if (!(0 & (1u << 0))) tmem_load_0[0] = 0.0f;
                    if (!(0 & (1u << 1))) tmem_load_0[1] = 0.0f;
                    if (!(0 & (1u << 2))) tmem_load_0[2] = 0.0f;
                    if (!(0 & (1u << 3))) tmem_load_0[3] = 0.0f;
                    if (!(0 & (1u << 4))) tmem_load_0[4] = 0.0f;
                    if (!(0 & (1u << 5))) tmem_load_0[5] = 0.0f;
                    if (!(0 & (1u << 6))) tmem_load_0[6] = 0.0f;
                    if (!(0 & (1u << 7))) tmem_load_0[7] = 0.0f;
                    if (!(0 & (1u << 8))) tmem_load_0[8] = 0.0f;
                    if (!(0 & (1u << 9))) tmem_load_0[9] = 0.0f;
                    if (!(0 & (1u << 10))) tmem_load_0[10] = 0.0f;
                    if (!(0 & (1u << 11))) tmem_load_0[11] = 0.0f;
                    if (!(0 & (1u << 12))) tmem_load_0[12] = 0.0f;
                    if (!(0 & (1u << 13))) tmem_load_0[13] = 0.0f;
                    if (!(0 & (1u << 14))) tmem_load_0[14] = 0.0f;
                    if (!(0 & (1u << 15))) tmem_load_0[15] = 0.0f;
                    if (!(0 & (1u << 16))) tmem_load_0[16] = 0.0f;
                    if (!(0 & (1u << 17))) tmem_load_0[17] = 0.0f;
                    if (!(0 & (1u << 18))) tmem_load_0[18] = 0.0f;
                    if (!(0 & (1u << 19))) tmem_load_0[19] = 0.0f;
                    if (!(0 & (1u << 20))) tmem_load_0[20] = 0.0f;
                    if (!(0 & (1u << 21))) tmem_load_0[21] = 0.0f;
                    if (!(0 & (1u << 22))) tmem_load_0[22] = 0.0f;
                    if (!(0 & (1u << 23))) tmem_load_0[23] = 0.0f;
                    if (!(0 & (1u << 24))) tmem_load_0[24] = 0.0f;
                    if (!(0 & (1u << 25))) tmem_load_0[25] = 0.0f;
                    if (!(0 & (1u << 26))) tmem_load_0[26] = 0.0f;
                    if (!(0 & (1u << 27))) tmem_load_0[27] = 0.0f;
                    if (!(0 & (1u << 28))) tmem_load_0[28] = 0.0f;
                    if (!(0 & (1u << 29))) tmem_load_0[29] = 0.0f;
                    if (!(0 & (1u << 30))) tmem_load_0[30] = 0.0f;
                    if (!(0 & (1u << 31))) tmem_load_0[31] = 0.0f;
                }
                if (my_row_1 < q_valid_1) {
                    #pragma unroll
                    for (int offset_1 = 0; offset_1 < 32; offset_1 += 8) {
                        float _scaled_1[8];
                        const float2 _scale2_12 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            reinterpret_cast<float2*>(_scaled_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((tmem_load_0 + offset_1))[_ls], _scale2_12);
                        unsigned int _packed_1[4];
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
                        reinterpret_cast<int4*>(out + (output_row + col_base + chunk_cols_1 + offset_1))[0] = reinterpret_cast<int4*>(_packed_1)[0];
                    }
                }
                if (my_row_1 < q_valid_1 && col_half_1 == 0 && return_lse != 0) {
                    int stat_idx = bh * seqlen_q + query;
                    float _log2_13;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_13) : "f"(total_sum));
                    lse[stat_idx] = ((has_mass != 0) ? (row_max_1 + _log2_13 - 8.0f) * 0.6931472f : -CUDART_INF_F);
                }
                corr_tiles_done += 1;
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            unsigned int kv_stage = 0;
            unsigned int q_stage = 0;
            int count_2 = (total_tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
            int my_tiles_2 = count_2;
            int first_tile = 0;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_kv_full = 0;
            if (my_tiles_2 > 0) {
                int q_block = blockIdx.x % q_blocks;
                int bh_1 = blockIdx.x / q_blocks;
                int head_1 = bh_1 % num_heads;
                int batch_2 = bh_1 / num_heads;
                int kv_head_2 = head_1 / (num_heads / num_kv_heads);
                int query_base_1 = q_block * 64;
                int q_valid_2 = seqlen_q - query_base_1;
                if (q_valid_2 > 64) {
                    q_valid_2 = 64;
                }
                if (q_valid_2 < 0) {
                    q_valid_2 = 0;
                }
                int row_slot = bh_1 * q_blocks + q_block;
                int selected_count_1 = block_sparse_num;
                if (has_block_nums != 0) {
                    selected_count_1 = block_nums[row_slot];
                }
                if (selected_count_1 < 0) {
                    selected_count_1 = 0;
                }
                if (selected_count_1 > capacity) {
                    selected_count_1 = capacity;
                }
                int pair_count_2 = (selected_count_1 + 1) / 2;
                if (pair_count_2 < 1) {
                    pair_count_2 = 1;
                }
                int index_base_1 = row_slot * capacity;
                mbarrier_wait(q_full_addr + (q_stage) * 8, _phase_q_full);
                unsigned int k_stage = kv_stage;
                mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage) * 512);
                int mma_b_lo = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                    }
                }
                elect_commit(s_full_addr);
                elect_commit(kv_empty_addr + (kv_stage) * 8);
                kv_stage += 1;
                if (kv_stage == 6) { kv_stage = 0; _phase_kv_full ^= 1; }
                if (pair_count_2 > 1) {
                    unsigned int k_stage_0 = kv_stage;
                    mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_1 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage) * 512);
                    int mma_b_lo_1 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage_0) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_1);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_scores + (128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_scores + (128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_scores + (128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_scores + (128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                        }
                    }
                    elect_commit(s_full_addr + 8);
                    elect_commit(kv_empty_addr + (kv_stage) * 8);
                    kv_stage += 1;
                    if (kv_stage == 6) { kv_stage = 0; _phase_kv_full ^= 1; }
                }
            }
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            unsigned int _phase_tile_done_0 = 0;
            #pragma unroll 1
            for (int tile_iter_2 = 0; tile_iter_2 < my_tiles_2; tile_iter_2++) {
                int tile_idx_1 = blockIdx.x + tile_iter_2 * gridDim.x;
                int q_block_1 = tile_idx_1 % q_blocks;
                int bh_2 = tile_idx_1 / q_blocks;
                int head_2 = bh_2 % num_heads;
                int batch_3 = bh_2 / num_heads;
                int kv_head_3 = head_2 / (num_heads / num_kv_heads);
                int query_base_2 = q_block_1 * 64;
                int q_valid_3 = seqlen_q - query_base_2;
                if (q_valid_3 > 64) {
                    q_valid_3 = 64;
                }
                if (q_valid_3 < 0) {
                    q_valid_3 = 0;
                }
                int row_slot_1 = bh_2 * q_blocks + q_block_1;
                int selected_count_2 = block_sparse_num;
                if (has_block_nums != 0) {
                    selected_count_2 = block_nums[row_slot_1];
                }
                if (selected_count_2 < 0) {
                    selected_count_2 = 0;
                }
                if (selected_count_2 > capacity) {
                    selected_count_2 = capacity;
                }
                int pair_count_3 = (selected_count_2 + 1) / 2;
                if (pair_count_3 < 1) {
                    pair_count_3 = 1;
                }
                int index_base_2 = row_slot_1 * capacity;
                int first_pv_0 = 1;
                int first_pv_1 = 1;
                #pragma unroll 1
                for (int step_1 = 0; step_1 < pair_count_3; step_1 += 2) {
                    unsigned int v_stage = kv_stage;
                    mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_b_lo_2 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024);
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
                    "mov.b32 id, 69271568;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(mma_b_lo_2), "r"(tmem_probs_0), "r"(((first_pv_0) ? 0 : 1)) : "memory");
                    elect_commit(kv_empty_addr + (kv_stage) * 8);
                    kv_stage += 1;
                    if (kv_stage == 6) { kv_stage = 0; _phase_kv_full ^= 1; }
                    first_pv_0 = 0;
                    if (pair_count_3 > step_1 + 2) {
                        unsigned int k_stage_1 = kv_stage;
                        mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int mma_a_lo_2 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage) * 512);
                        int mma_b_lo_0 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage_1) * 1024);
                        {
                            uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_2);
                            uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_0);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4(tmem_scores, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4(tmem_scores, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4(tmem_scores, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4(tmem_scores, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                            }
                        }
                        elect_commit(s_full_addr);
                        elect_commit(kv_empty_addr + (kv_stage) * 8);
                        kv_stage += 1;
                        if (kv_stage == 6) { kv_stage = 0; _phase_kv_full ^= 1; }
                    }
                    if (pair_count_3 > step_1 + 1) {
                        unsigned int v_stage_0 = kv_stage;
                        mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                        mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                        _phase_p_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int mma_b_lo_2_1 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_0) * 1024);
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
                    "mov.b32 id, 69271568;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(mma_b_lo_2_1), "r"(tmem_probs_1), "r"(((first_pv_1) ? 0 : 1)) : "memory");
                        elect_commit(kv_empty_addr + (kv_stage) * 8);
                        kv_stage += 1;
                        if (kv_stage == 6) { kv_stage = 0; _phase_kv_full ^= 1; }
                        first_pv_1 = 0;
                        if (pair_count_3 > step_1 + 3) {
                            unsigned int k_stage_2 = kv_stage;
                            mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int mma_a_lo_3 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage) * 512);
                            int mma_b_lo_3 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage_2) * 1024);
                            {
                                uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_3);
                                uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_3);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_scores + (128)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 0);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_scores + (128)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_scores + (128)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_scores + (128)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                                }
                            }
                            elect_commit(s_full_addr + 8);
                            elect_commit(kv_empty_addr + (kv_stage) * 8);
                            kv_stage += 1;
                            if (kv_stage == 6) { kv_stage = 0; _phase_kv_full ^= 1; }
                        }
                    }
                }
                if (my_tiles_2 > tile_iter_2 + 1) {
                    elect_commit(q_empty_addr + (q_stage) * 8);
                }
                q_stage += 1;
                if (q_stage == 2) { q_stage = 0; _phase_q_full ^= 1; }
                elect_commit(o_full_addr);
                if (my_tiles_2 > tile_iter_2 + 1) {
                    int next_tile = tile_iter_2 + 1;
                    int next_idx = tile_idx_1 + gridDim.x;
                    int q_block_0 = next_idx % q_blocks;
                    int bh_0 = next_idx / q_blocks;
                    int head_0 = bh_0 % num_heads;
                    int batch_0 = bh_0 / num_heads;
                    int kv_head_0 = head_0 / (num_heads / num_kv_heads);
                    int query_base_0 = q_block_0 * 64;
                    int q_valid_0 = seqlen_q - query_base_0;
                    if (q_valid_0 > 64) {
                        q_valid_0 = 64;
                    }
                    if (q_valid_0 < 0) {
                        q_valid_0 = 0;
                    }
                    int row_slot_0 = bh_0 * q_blocks + q_block_0;
                    int selected_count_0 = block_sparse_num;
                    if (has_block_nums != 0) {
                        selected_count_0 = block_nums[row_slot_0];
                    }
                    if (selected_count_0 < 0) {
                        selected_count_0 = 0;
                    }
                    if (selected_count_0 > capacity) {
                        selected_count_0 = capacity;
                    }
                    int pair_count_0 = (selected_count_0 + 1) / 2;
                    if (pair_count_0 < 1) {
                        pair_count_0 = 1;
                    }
                    int index_base_0 = row_slot_0 * capacity;
                    mbarrier_wait(q_full_addr + (q_stage) * 8, _phase_q_full);
                    unsigned int k_stage_3 = kv_stage;
                    mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_4 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage) * 512);
                    int mma_b_lo_4 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage_3) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_4);
                        uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 69206032, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 69206032, 1);
                        }
                    }
                    elect_commit(s_full_addr);
                    elect_commit(kv_empty_addr + (kv_stage) * 8);
                    kv_stage += 1;
                    if (kv_stage == 6) { kv_stage = 0; _phase_kv_full ^= 1; }
                    if (pair_count_0 > 1) {
                        unsigned int k_stage_0_1 = kv_stage;
                        mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int mma_a_lo_1_1 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage) * 512);
                        int mma_b_lo_1_1 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage_0_1) * 1024);
                        {
                            uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_1_1);
                            uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_1_1);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_scores + (128)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 69206032, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_scores + (128)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_scores + (128)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_scores + (128)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 69206032, 1);
                            }
                        }
                        elect_commit(s_full_addr + 8);
                        elect_commit(kv_empty_addr + (kv_stage) * 8);
                        kv_stage += 1;
                        if (kv_stage == 6) { kv_stage = 0; _phase_kv_full ^= 1; }
                    }
                }
                mbarrier_wait(tile_done_addr, _phase_tile_done_0);
                _phase_tile_done_0 ^= 1;
            }
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
    // ---- Role: load_warp ----
    if (warp == 13) {
        { // load_warp_main
            unsigned int load_stage = 0;
            unsigned int q_load_stage = 0;
            int count_3 = (total_tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
            int my_tiles_3 = count_3;
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (int tile_iter_3 = 0; tile_iter_3 < my_tiles_3; tile_iter_3++) {
                int tile_idx_2 = blockIdx.x + tile_iter_3 * gridDim.x;
                int q_block_2 = tile_idx_2 % q_blocks;
                int bh_3 = tile_idx_2 / q_blocks;
                int head_3 = bh_3 % num_heads;
                int batch_4 = bh_3 / num_heads;
                int kv_head_4 = head_3 / (num_heads / num_kv_heads);
                int query_base_3 = q_block_2 * 64;
                int q_valid_4 = seqlen_q - query_base_3;
                if (q_valid_4 > 64) {
                    q_valid_4 = 64;
                }
                if (q_valid_4 < 0) {
                    q_valid_4 = 0;
                }
                int row_slot_2 = bh_3 * q_blocks + q_block_2;
                int selected_count_3 = block_sparse_num;
                if (has_block_nums != 0) {
                    selected_count_3 = block_nums[row_slot_2];
                }
                if (selected_count_3 < 0) {
                    selected_count_3 = 0;
                }
                if (selected_count_3 > capacity) {
                    selected_count_3 = capacity;
                }
                int pair_count_4 = (selected_count_3 + 1) / 2;
                if (pair_count_4 < 1) {
                    pair_count_4 = 1;
                }
                int index_base_3 = row_slot_2 * capacity;
                mbarrier_wait(q_empty_addr + (q_load_stage) * 8, _phase_q_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + (q_load_stage) * 8, 8192);
                    tma_4d_gmem2smem(q_smem_addr + q_load_stage * 8192, (&q), 0, head_3, query_base_3, batch_4, q_full_addr + (q_load_stage) * 8);
                }
                q_load_stage += 1;
                if (q_load_stage == 2) { q_load_stage = 0; _phase_q_empty ^= 1; }
                int entry0 = 0;
                int block0 = 0;
                if (entry0 < selected_count_3) {
                    block0 = block_index[index_base_3 + entry0];
                }
                int block1 = block0;
                if (selected_count_3 > entry0 + 1) {
                    block1 = block_index[index_base_3 + entry0 + 1];
                }
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384, (&k), 0, block0 * 64, kv_head_4, batch_4, kv_full_addr + (load_stage) * 8);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384 + 8192, (&k), 0, block1 * 64, kv_head_4, batch_4, kv_full_addr + (load_stage) * 8);
                }
                load_stage += 1;
                if (load_stage == 6) { load_stage = 0; _phase_kv_empty ^= 1; }
                if (pair_count_4 > 1) {
                    int entry0_0 = 2;
                    int block0_0 = 0;
                    if (entry0_0 < selected_count_3) {
                        block0_0 = block_index[index_base_3 + entry0_0];
                    }
                    int block1_0 = block0_0;
                    if (selected_count_3 > entry0_0 + 1) {
                        block1_0 = block_index[index_base_3 + entry0_0 + 1];
                    }
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384, (&k), 0, block0_0 * 64, kv_head_4, batch_4, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384 + 8192, (&k), 0, block1_0 * 64, kv_head_4, batch_4, kv_full_addr + (load_stage) * 8);
                    }
                    load_stage += 1;
                    if (load_stage == 6) { load_stage = 0; _phase_kv_empty ^= 1; }
                }
                int lane_0 = lane;
                int k_scale_base_1 = (batch_4 * num_kv_heads + kv_head_4) * k_scale_groups;
                float acc = 0.0f;
                if (lane_0 < q_valid_4) {
                    acc = q_scale[bh_3 * seqlen_q + query_base_3 + lane_0];
                }
                if (q_valid_4 > lane_0 + 32) {
                    acc = acc + q_scale[bh_3 * seqlen_q + query_base_3 + lane_0 + 32];
                }
                int valid = 0;
                if (lane_0 < 16) {
                    int entry = lane_0 / 4;
                    int block = 0;
                    if (entry < selected_count_3) {
                        block = block_index[index_base_3 + entry];
                    } else if (entry % 2 == 1 && selected_count_3 > entry - 1) {
                        block = block_index[index_base_3 + entry - 1];
                    }
                    int group_idx = block * 4 + lane_0 % 4;
                    if (group_idx > k_scale_groups - 1) {
                        group_idx = k_scale_groups - 1;
                    }
                    if (group_idx < 0) {
                        group_idx = 0;
                    }
                    acc = acc + k_scale[k_scale_base_1 + group_idx];
                    if (lane_0 % 4 == 0) {
                        int valid_1 = seqlen_k - block * 64;
                        if (valid_1 > 64) {
                            valid_1 = 64;
                        }
                        if (valid_1 < 0) {
                            valid_1 = 0;
                        }
                        if (block_sizes_mode != 0) {
                            int declared = 0;
                            if (block_sizes_mode == 1) {
                                declared = block_sizes[block];
                            } else if (block_sizes_mode == 2) {
                                declared = block_sizes[batch_4 * k_blocks + block];
                            } else {
                                declared = block_sizes[(batch_4 * num_kv_heads + kv_head_4) * k_blocks + block];
                            }
                            if (declared < 0) {
                                declared = 0;
                            }
                            if (declared < valid_1) {
                                valid_1 = declared;
                            }
                        }
                        valid = valid_1;
                    }
                }
                #pragma unroll 1
                for (int step_2 = 0; step_2 < pair_count_4; step_2++) {
                    int entry0_1 = step_2 * 2;
                    int block0_1 = 0;
                    if (entry0_1 < selected_count_3) {
                        block0_1 = block_index[index_base_3 + entry0_1];
                    }
                    int block1_1 = block0_1;
                    if (selected_count_3 > entry0_1 + 1) {
                        block1_1 = block_index[index_base_3 + entry0_1 + 1];
                    }
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 16384, (&v), 0, block0_1 * 64, kv_head_4, batch_4, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 16384 + 8192, (&v), 0, block1_1 * 64, kv_head_4, batch_4, kv_full_addr + (load_stage) * 8);
                    }
                    load_stage += 1;
                    if (load_stage == 6) { load_stage = 0; _phase_kv_empty ^= 1; }
                    if (pair_count_4 > step_2 + 2) {
                        int entry0_2 = (step_2 + 2) * 2;
                        int block0_2 = 0;
                        if (entry0_2 < selected_count_3) {
                            block0_2 = block_index[index_base_3 + entry0_2];
                        }
                        int block1_2 = block0_2;
                        if (selected_count_3 > entry0_2 + 1) {
                            block1_2 = block_index[index_base_3 + entry0_2 + 1];
                        }
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384, (&k), 0, block0_2 * 64, kv_head_4, batch_4, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384 + 8192, (&k), 0, block1_2 * 64, kv_head_4, batch_4, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 6) { load_stage = 0; _phase_kv_empty ^= 1; }
                    }
                }
                pf_sink_f[lane_0] = acc;
                pf_sink_i[lane_0] = valid;
            }
        }
    }
    // ---- Role: empty ----
    if (warp >= 14 && warp <= 15) {
        { // empty_main
            int p_group = make_warp_uniform(warp - 14);
            int p_lane = lane;
            int p_blk = ((p_lane < 8) ? p_lane / 4 : p_lane % 2);
            int p_kg = p_lane % 4;
            int p_n = p_group * 4;
            int count_4 = (total_tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
            int my_tiles_4 = count_4;
            unsigned int _phase_step_empty = 1;
            #pragma unroll 1
            for (int tile_iter_4 = 0; tile_iter_4 < my_tiles_4; tile_iter_4++) {
                int tile_idx_3 = blockIdx.x + tile_iter_4 * gridDim.x;
                int q_block_3 = tile_idx_3 % q_blocks;
                int bh_4 = tile_idx_3 / q_blocks;
                int head_4 = bh_4 % num_heads;
                int batch_5 = bh_4 / num_heads;
                int kv_head_5 = head_4 / (num_heads / num_kv_heads);
                int query_base_4 = q_block_3 * 64;
                int q_valid_5 = seqlen_q - query_base_4;
                if (q_valid_5 > 64) {
                    q_valid_5 = 64;
                }
                if (q_valid_5 < 0) {
                    q_valid_5 = 0;
                }
                int row_slot_3 = bh_4 * q_blocks + q_block_3;
                int selected_count_4 = block_sparse_num;
                if (has_block_nums != 0) {
                    selected_count_4 = block_nums[row_slot_3];
                }
                if (selected_count_4 < 0) {
                    selected_count_4 = 0;
                }
                if (selected_count_4 > capacity) {
                    selected_count_4 = capacity;
                }
                int pair_count_5 = (selected_count_4 + 1) / 2;
                if (pair_count_5 < 1) {
                    pair_count_5 = 1;
                }
                int index_base_4 = row_slot_3 * capacity;
                int k_scale_base_2 = (batch_5 * num_kv_heads + kv_head_5) * k_scale_groups;
                int e_entry0 = p_group * 2;
                int e_block = 0;
                if (p_lane < 12) {
                    if (selected_count_4 > e_entry0 + p_blk) {
                        e_block = block_index[index_base_4 + e_entry0 + p_blk];
                    } else if (p_blk == 1 && e_entry0 < selected_count_4) {
                        e_block = block_index[index_base_4 + e_entry0];
                    }
                }
                int cur_block = e_block;
                if (p_group == 0) {
                    float s_q_lo = 0.0f;
                    if (p_lane < q_valid_5) {
                        s_q_lo = q_scale[bh_4 * seqlen_q + query_base_4 + p_lane];
                    }
                    float s_q_hi = 0.0f;
                    if (q_valid_5 > p_lane + 32) {
                        s_q_hi = q_scale[bh_4 * seqlen_q + query_base_4 + p_lane + 32];
                    }
                    unsigned int s_stage = (unsigned int)(tile_iter_4 % 2);
                    int s_off = tile_iter_4 % 2 * 100;
                    int _stage_meta_empty = (int)((unsigned int)tile_iter_4 % 2);
                    int _parity_meta_empty = (int)((unsigned int)tile_iter_4 / 2 % 2) ^ 1;
                    mbarrier_wait(meta_empty_addr + (_stage_meta_empty) * 8, _parity_meta_empty);
                    meta_f[s_off + p_lane] = s_q_lo;
                    meta_f[s_off + 32 + p_lane] = s_q_hi;
                    meta_i[s_off + 88] = batch_5;
                    meta_i[s_off + 88 + 1] = kv_head_5;
                    meta_i[s_off + 88 + 2] = k_scale_base_2;
                    meta_i[s_off + 88 + 3] = q_valid_5;
                    meta_i[s_off + 88 + 4] = selected_count_4;
                    meta_i[s_off + 88 + 5] = pair_count_5;
                    meta_i[s_off + 88 + 6] = index_base_4;
                    meta_i[s_off + 88 + 7] = head_4;
                    meta_i[s_off + 88 + 8] = bh_4;
                    meta_i[s_off + 88 + 9] = query_base_4;
                    mbarrier_arrive(meta_full_addr + ((int)((unsigned int)tile_iter_4 % 2)) * 8);
                }
                #pragma unroll 1
                for (int pi = p_group; pi < pair_count_5; pi += 2) {
                    int e_entry0_0 = (pi + 2) * 2;
                    int e_block_0 = 0;
                    if (p_lane < 12) {
                        if (selected_count_4 > e_entry0_0 + p_blk) {
                            e_block_0 = block_index[index_base_4 + e_entry0_0 + p_blk];
                        } else if (p_blk == 1 && e_entry0_0 < selected_count_4) {
                            e_block_0 = block_index[index_base_4 + e_entry0_0];
                        }
                    }
                    int nxt_block = e_block_0;
                    float w_f = 0.0f;
                    int w_i = 0;
                    if (p_lane < 8) {
                        int w_group = cur_block * 4 + p_kg;
                        if (w_group > k_scale_groups - 1) {
                            w_group = k_scale_groups - 1;
                        }
                        if (w_group < 0) {
                            w_group = 0;
                        }
                        w_f = k_scale[k_scale_base_2 + w_group];
                    } else if (p_lane < 10) {
                        w_i = cur_block;
                    } else {
                        if (p_lane < 12) {
                            int valid_2 = seqlen_k - cur_block * 64;
                            if (valid_2 > 64) {
                                valid_2 = 64;
                            }
                            if (valid_2 < 0) {
                                valid_2 = 0;
                            }
                            if (block_sizes_mode != 0) {
                                int declared_1 = 0;
                                if (block_sizes_mode == 1) {
                                    declared_1 = block_sizes[cur_block];
                                } else if (block_sizes_mode == 2) {
                                    declared_1 = block_sizes[batch_5 * k_blocks + cur_block];
                                } else {
                                    declared_1 = block_sizes[(batch_5 * num_kv_heads + kv_head_5) * k_blocks + cur_block];
                                }
                                if (declared_1 < 0) {
                                    declared_1 = 0;
                                }
                                if (declared_1 < valid_2) {
                                    valid_2 = declared_1;
                                }
                            }
                            w_i = valid_2;
                        }
                    }
                    int r_e = p_n % 4;
                    int r_base = (p_group * 4 + r_e) * 12;
                    mbarrier_wait(step_empty_addr + (p_n) * 8, _phase_step_empty);
                    if (p_lane < 8) {
                        ring_f[r_base + 4 + p_lane] = w_f;
                    } else if (p_lane < 10) {
                        ring_i[r_base + p_blk] = w_i;
                    } else {
                        if (p_lane < 12) {
                            ring_i[r_base + 2 + p_blk] = w_i;
                        }
                    }
                    mbarrier_arrive(step_full_addr + (p_n) * 8);
                    cur_block = nxt_block;
                    p_n += 1;
                    if (p_n == (p_group * 4 + 4)) { p_n = p_group * 4; _phase_step_empty ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
