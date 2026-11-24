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
#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
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

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
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




__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
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
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xFFFFFFFF;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdp_25d81fdd5cf38d846a0d(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, const __grid_constant__ CUtensorMap Omap, uint8_t* __restrict__ tmap_ws, float* __restrict__ gate, float* __restrict__ beta, int* __restrict__ cu_seqlens, int* __restrict__ items, __nv_bfloat16* __restrict__ initial_state, __nv_bfloat16* __restrict__ output_state, long long state_stride_slot, float scale, int num_items, int total_tokens, int num_heads, int num_householder, int flags, uint8_t* __restrict__ split_ws, int num_seqs, int* __restrict__ state_indices, int* __restrict__ output_state_indices, long long initial_state_stride_slot, long long output_state_stride_slot, long long* __restrict__ rand_seed, int philox_rounds, long long initial_state_size_p0, long long output_state_size_p0)
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
    #define kq_full_addr (mbar_base + 0)
    #define kq_empty_addr (mbar_base + 32)
    #define v_full_addr (mbar_base + 64)
    #define v_empty_addr (mbar_base + 80)
    #define gate_full_addr (mbar_base + 96)
    #define gate_empty_cg0_addr (mbar_base + 128)
    #define gate_empty_cg1_addr (mbar_base + 160)
    #define acc_ready_addr (mbar_base + 192)
    #define acc_done_addr (mbar_base + 208)
    #define tinv_ready_addr (mbar_base + 224)
    #define tinv_empty_addr (mbar_base + 248)
    #define a_ready_addr (mbar_base + 272)
    #define a_empty_addr (mbar_base + 296)
    #define merge_full_addr (mbar_base + 320)
    #define img_ready_addr (mbar_base + 336)
    #define ks_full_addr (mbar_base + 344)
    #define y_ready_addr (mbar_base + 352)
    #define u_full_addr (mbar_base + 360)
    #define uimg_ready_addr (mbar_base + 368)
    #define du_ready_addr (mbar_base + 376)
    #define state_full_addr (mbar_base + 384)
    #define qs_full_addr (mbar_base + 392)
    #define qs_scaled_addr (mbar_base + 400)
    #define o_full_addr (mbar_base + 408)
    #define o_empty_addr (mbar_base + 416)
    #define o_stage_ready_addr (mbar_base + 424)
    #define o_stage_empty_addr (mbar_base + 432)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* smem_kq = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_kq_addr = smem + 1024;
    __nv_bfloat16* smem_kq_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_kq_mn_addr = smem + 1024;
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_k_addr = smem + 1024;
    __nv_bfloat16* smem_k_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_k_mn_addr = smem + 1024;
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_q_addr = smem + 17408;
    __nv_bfloat16* smem_kq_h = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_kq_h_addr = smem + 1024;
    __nv_bfloat16* smem_kq_mn_h = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_kq_mn_h_addr = smem + 1024;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int smem_v_addr = smem + 132096;
    __nv_bfloat16* smem_tinv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int smem_tinv_addr = smem + 148480;
    __nv_bfloat16* smem_tinv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int smem_tinv_mn_addr = smem + 148480;
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 173056);
    const int smem_a_addr = smem + 173056;
    __nv_bfloat16* smem_o = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int smem_o_addr = smem + 197632;
    __nv_bfloat16* smem_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 211984);
    const int smem_x_addr = smem + 211984;
    __nv_bfloat16* smem_x_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 211984);
    const int smem_x_mn_addr = smem + 211984;
    float* smem_cs = reinterpret_cast<float*>(smem_raw + 205824);
    const int smem_cs_addr = smem + 205824;
    float* smem_blk = reinterpret_cast<float*>(smem_raw + 206848);
    const int smem_blk_addr = smem + 206848;
    float* smem_beta = reinterpret_cast<float*>(smem_raw + 207872);
    const int smem_beta_addr = smem + 207872;
    float* smem_ct = reinterpret_cast<float*>(smem_raw + 208896);
    const int smem_ct_addr = smem + 208896;
    unsigned int* smem_dump = reinterpret_cast<unsigned int*>(smem_raw + 209920);
    const int smem_dump_addr = smem + 209920;
    int* smem_scan = reinterpret_cast<int*>(smem_raw + 211968);
    const int smem_scan_addr = smem + 211968;
    float* smem_ecs = reinterpret_cast<float*>(smem_raw + 220176);
    const int smem_ecs_addr = smem + 220176;
    float* smem_edec = reinterpret_cast<float*>(smem_raw + 221200);
    const int smem_edec_addr = smem + 221200;
    float* smem_ect = reinterpret_cast<float*>(smem_raw + 222224);
    const int smem_ect_addr = smem + 222224;

    // Kernel setup ops
    int num_hh = 3;
    if (tid == 0) {
        smem_scan[0] = 0;
    }

    // Mbarrier init (55 barriers at smem_raw[0..440))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'kq_pipe' ---
            // kq_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // kq_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // v_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 80, 128);
            mbarrier_init(smem + 88, 128);
            // --- pipeline 'gate_pipe' ---
            // gate_full: 4 barriers, init_count=32
            mbarrier_init(smem + 96, 32);
            mbarrier_init(smem + 104, 32);
            mbarrier_init(smem + 112, 32);
            mbarrier_init(smem + 120, 32);
            // gate_empty_cg0: 4 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            mbarrier_init(smem + 136, 128);
            mbarrier_init(smem + 144, 128);
            mbarrier_init(smem + 152, 128);
            // gate_empty_cg1: 4 barriers, init_count=128
            mbarrier_init(smem + 160, 128);
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
            mbarrier_init(smem + 184, 128);
            // --- pipeline 'acc_pipe' ---
            // acc_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            // acc_done: 2 barriers, init_count=64
            mbarrier_init(smem + 208, 64);
            mbarrier_init(smem + 216, 64);
            // --- pipeline 'tinv_pipe' ---
            // tinv_ready: 3 barriers, init_count=128
            mbarrier_init(smem + 224, 128);
            mbarrier_init(smem + 232, 128);
            mbarrier_init(smem + 240, 128);
            // tinv_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            // --- pipeline 'a_pipe' ---
            // a_ready: 3 barriers, init_count=64
            mbarrier_init(smem + 272, 64);
            mbarrier_init(smem + 280, 64);
            mbarrier_init(smem + 288, 64);
            // a_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // merge_full: 2 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            // img_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 336, 128);
            // ks_full: 1 barriers, init_count=1
            mbarrier_init(smem + 344, 1);
            // y_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 352, 128);
            // u_full: 1 barriers, init_count=1
            mbarrier_init(smem + 360, 1);
            // uimg_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 368, 128);
            // du_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 376, 128);
            // state_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            // qs_full: 1 barriers, init_count=1
            mbarrier_init(smem + 392, 1);
            // qs_scaled: 1 barriers, init_count=128
            mbarrier_init(smem + 400, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 408, 1);
            // o_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 416, 128);
            // o_stage_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 424, 128);
            // o_stage_empty: 1 barriers, init_count=32
            mbarrier_init(smem + 432, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 440);
    if (warp == 10) {
        int _tmem_hold = smem + 440;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_state = taddr;
    const unsigned int tmem_o = taddr + 128;
    const unsigned int tmem_img = taddr + 192;
    const unsigned int tmem_acc = taddr + 256;
    const unsigned int tmem_ks = taddr + 384;
    const unsigned int tmem_y = taddr + 448;
    const unsigned int tmem_du = taddr + 480;

    // Kernel post-init ops
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    }

    // ---- Role: cg0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // cg0_main
            int warp_id_in_role = warp - 0;
            int warp0 = warp_id_in_role;
            int half0 = warp0 / 2;
            int sub0 = warp0 & 1;
            int tid64_0 = sub0 * 32 + lane;
            unsigned int g_cur0 = 0;
            unsigned int tinv_cur0 = 0;
            unsigned int a_cur0 = 0;
            unsigned int acc_cur0 = 0;
            asm volatile("barrier.sync 3, 384;" ::: "memory");
            unsigned int _phase_acc_ready = 0;
            unsigned int _phase_gate_full = 0;
            unsigned int _phase_tinv_empty = 1;
            unsigned int _phase_a_empty = 1;
            #pragma unroll 1
            for (unsigned int it0 = ((smem_scan[0] == 0) ? (unsigned int)smem_scan[2] : (unsigned int)smem_scan[1]); it0 < (unsigned int)smem_scan[1]; it0++) {
                int w0 = 0;
                int xh = 0;
                int seq = items[(int)it0 * 4];
                int head = items[(int)it0 * 4 + 1];
                int cut = items[(int)it0 * 4 + 2];
                int span = items[(int)it0 * 4 + 3];
                int rs = cu_seqlens[seq];
                int re = cu_seqlens[seq + 1];
                int start = ((rs < 0) ? 0 : ((rs > total_tokens) ? total_tokens : rs));
                int end = ((re < start) ? start : ((re > total_tokens) ? total_tokens : re));
                int tok0 = start + cut * 64;
                int tokend = start + (cut + span) * 64;
                tokend = ((end < tokend) ? end : tokend);
                int sp_at = num_seqs * num_heads + (int)it0 * 2;
                int sp_lo = (int)reinterpret_cast<unsigned int*>(split_ws)[sp_at];
                int sp_num = (int)reinterpret_cast<unsigned int*>(split_ws)[sp_at + 1];
                int sp_wend = sp_lo + sp_num;
                int start_tok = start;
                tokend = start + (sp_wend + num_hh - 1) / num_hh * 64;
                tokend = ((end < tokend) ? end : tokend);
                int cstart = sp_lo;
                int abs0 = cut;
                int nch = ((sp_num == 0) ? 0 : (tokend - start + 63) / 64 * num_hh - sp_lo);
                int wstart = cut;
                int owns_end = (((sp_num > 0 || cut == 0) && sp_wend == 2 * (((end - start) * num_hh + 127) / 128)) ? 1 : 0);
                int npairs0 = (nch + 1) / 2;
                int blk0c = cstart / num_hh;
                int cb0c = cstart - blk0c * num_hh;
                #pragma unroll 1
                for (int p0 = 0; p0 < npairs0; p0++) {
                    int i00 = p0 * 2;
                    int has_m1_0 = ((i00 + 1 < nch) ? 1 : 0);
                    int blk00 = blk0c;
                    int cb00 = cb0c;
                    int blk01 = ((cb0c == num_hh - 1) ? blk0c + 1 : blk0c);
                    int cb01 = ((cb0c == num_hh - 1) ? 0 : cb0c + 1);
                    int vt00 = tokend - (start_tok + blk00 * 64);
                    vt00 = ((vt00 > 64) ? 64 : ((vt00 < 0) ? 0 : vt00));
                    int vt01 = tokend - (start_tok + blk01 * 64);
                    vt01 = ((vt01 > 64) ? 64 : ((vt01 < 0) ? 0 : vt01));
                    unsigned int acc_cur01 = ((acc_cur0 + 1 < 2) ? acc_cur0 + 1 : (unsigned int)0);
                    unsigned int g_cur01 = ((g_cur0 + 1 < 4) ? g_cur0 + 1 : (unsigned int)0);
                    unsigned int tinv_cur01 = ((tinv_cur0 + 1 < 3) ? tinv_cur0 + 1 : (unsigned int)0);
                    unsigned int a_cur01 = ((a_cur0 + 1 < 3) ? a_cur0 + 1 : (unsigned int)0);
                    mbarrier_wait(acc_ready_addr + (acc_cur0) * 8, _phase_acc_ready);
                    if (has_m1_0 != 0) {
                        mbarrier_wait(acc_ready_addr + (((acc_cur0 + 1 < 2) ? acc_cur0 + 1 : (unsigned int)0)) * 8, ((acc_cur0 + 1 < 2) ? _phase_acc_ready : _phase_acc_ready ^ 1));
                    }
                    mbarrier_wait(gate_full_addr + (g_cur0) * 8, _phase_gate_full);
                    if (has_m1_0 != 0) {
                        mbarrier_wait(gate_full_addr + (((g_cur0 + 1 < 4) ? g_cur0 + 1 : (unsigned int)0)) * 8, ((g_cur0 + 1 < 4) ? _phase_gate_full : _phase_gate_full ^ 1));
                    }
                    int gbase00 = g_cur0 * 64;
                    int gbase01 = g_cur01 * 64;
                    mbarrier_wait(tinv_empty_addr + (tinv_cur0) * 8, _phase_tinv_empty);
                    if (has_m1_0 != 0) {
                        mbarrier_wait(tinv_empty_addr + (((tinv_cur0 + 1 < 3) ? tinv_cur0 + 1 : (unsigned int)0)) * 8, ((tinv_cur0 + 1 < 3) ? _phase_tinv_empty : _phase_tinv_empty ^ 1));
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int tinv_st00 = smem_tinv_addr + tinv_cur0 * 8192;
                    unsigned int tinv_st01 = smem_tinv_addr + tinv_cur01 * 8192;
                    int own_present0 = ((half0 == 0 || has_m1_0 != 0) ? 1 : 0);
                    unsigned int inv_base0 = ((own_present0 != 0) ? ((half0 == 0) ? tinv_st00 : tinv_st01) : smem_x_addr);
                    int kk_gb0 = ((half0 == 0) ? gbase00 : ((has_m1_0 != 0) ? gbase01 : gbase00));
                    unsigned int kk_cols0 = 256 + 64 * ((half0 == 0) ? acc_cur0 : acc_cur01);
                    #pragma unroll
                    for (int kh0 = 0; kh0 < 2; kh0++) {
                        int kkb_lane_quad = lane & 3;
                        int kkb_lane_row = lane / 4;
                        int kkb_stsm_row = sub0 * 32 + kh0 * 16 + (lane & 7) + (lane & 8);
                        int kkb_stsm_col = (lane & 16) / 2;
                        float kk[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&kk[0])), "=r"(*reinterpret_cast<uint32_t*>(&kk[1])), "=r"(*reinterpret_cast<uint32_t*>(&kk[2])), "=r"(*reinterpret_cast<uint32_t*>(&kk[3])), "=r"(*reinterpret_cast<uint32_t*>(&kk[4])), "=r"(*reinterpret_cast<uint32_t*>(&kk[5])), "=r"(*reinterpret_cast<uint32_t*>(&kk[6])), "=r"(*reinterpret_cast<uint32_t*>(&kk[7])), "=r"(*reinterpret_cast<uint32_t*>(&kk[8])), "=r"(*reinterpret_cast<uint32_t*>(&kk[9])), "=r"(*reinterpret_cast<uint32_t*>(&kk[10])), "=r"(*reinterpret_cast<uint32_t*>(&kk[11])), "=r"(*reinterpret_cast<uint32_t*>(&kk[12])), "=r"(*reinterpret_cast<uint32_t*>(&kk[13])), "=r"(*reinterpret_cast<uint32_t*>(&kk[14])), "=r"(*reinterpret_cast<uint32_t*>(&kk[15])), "=r"(*reinterpret_cast<uint32_t*>(&kk[16])), "=r"(*reinterpret_cast<uint32_t*>(&kk[17])), "=r"(*reinterpret_cast<uint32_t*>(&kk[18])), "=r"(*reinterpret_cast<uint32_t*>(&kk[19])), "=r"(*reinterpret_cast<uint32_t*>(&kk[20])), "=r"(*reinterpret_cast<uint32_t*>(&kk[21])), "=r"(*reinterpret_cast<uint32_t*>(&kk[22])), "=r"(*reinterpret_cast<uint32_t*>(&kk[23])), "=r"(*reinterpret_cast<uint32_t*>(&kk[24])), "=r"(*reinterpret_cast<uint32_t*>(&kk[25])), "=r"(*reinterpret_cast<uint32_t*>(&kk[26])), "=r"(*reinterpret_cast<uint32_t*>(&kk[27])), "=r"(*reinterpret_cast<uint32_t*>(&kk[28])), "=r"(*reinterpret_cast<uint32_t*>(&kk[29])), "=r"(*reinterpret_cast<uint32_t*>(&kk[30])), "=r"(*reinterpret_cast<uint32_t*>(&kk[31]))
                            : "r"(taddr + kk_cols0 + (unsigned int)(warp0 * 32 + kh0 * 16 << 16)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        int kkb_i0 = sub0 * 32 + kh0 * 16 + kkb_lane_row;
                        int kkb_i1 = kkb_i0 + 8;
                        float kkb_cs_i0 = smem_cs[kk_gb0 + kkb_i0];
                        float kkb_cs_i1 = smem_cs[kk_gb0 + kkb_i1];
                        float kkb_b_i0 = smem_beta[kk_gb0 + kkb_i0];
                        float kkb_b_i1 = smem_beta[kk_gb0 + kkb_i1];
                        float lt[32];
                        #pragma unroll
                        for (int jk = 0; jk < 32; jk++) {
                            int kkb_jc = jk / 4 * 8 + kkb_lane_quad * 2 + (jk & 1);
                            int kkb_hi = (jk & 3) / 2;
                            int kkb_ir = ((kkb_hi != 0) ? kkb_i1 : kkb_i0);
                            float kkb_cs = ((kkb_hi != 0) ? kkb_cs_i1 : kkb_cs_i0);
                            float kkb_b = ((kkb_hi != 0) ? kkb_b_i1 : kkb_b_i0);
                            float kkb_cs_j = smem_cs[kk_gb0 + kkb_jc];
                            float _exp2_0 = approx_exp2(kkb_cs - kkb_cs_j);
                            lt[jk] = ((kkb_jc < kkb_ir) ? kk[jk] * _exp2_0 * kkb_b : 0.0f);
                        }
                        #pragma unroll
                        for (int rk = 0; rk < 4; rk++) {
                            unsigned int ltb[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((lt + rk * 8)[_lp*2 + 0], (lt + rk * 8)[_lp*2+1 + 0]));
                                ltb[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int kkb_addr = inv_base0 + (unsigned int)(kkb_stsm_row * 128 + (((rk * 16 + kkb_stsm_col) * 2 >> 4 ^ kkb_stsm_row & 7) << 4) + ((rk * 16 + kkb_stsm_col) * 2 & 15));
                            uint32_t _stmatrix_addr_1 = static_cast<uint32_t>((unsigned long long)kkb_addr);
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&ltb[0])), "r"(*reinterpret_cast<const uint32_t*>(&ltb[1])), "r"(*reinterpret_cast<const uint32_t*>(&ltb[2])), "r"(*reinterpret_cast<const uint32_t*>(&ltb[3]))
                                : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    int gj_blk = tid64_0 / 8;
                    int gj_lane_in = lane & 7;
                    int gj_col0 = gj_blk * 8;
                    int gj_row = tid64_0;
                    unsigned int gj_words[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gj_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&gj_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gj_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gj_words[(0) + 3]))
                        : "r"(inv_base0 + (unsigned int)(gj_row * 128 + ((gj_col0 * 2 >> 4 ^ gj_row & 7) << 4) + (gj_col0 * 2 & 15))));
                    float gj_v[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&gj_v[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(gj_words[_pair]) << 16);
                        (&gj_v[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(gj_words[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int gj_c = 0; gj_c < 8; gj_c++) {
                        gj_v[gj_c] = ((gj_lane_in == gj_c) ? 1.0f : ((gj_lane_in < gj_c) ? 0.0f : gj_v[gj_c]));
                    }
                    int gj_sub = lane - gj_lane_in;
                    float gj_scale_0 = -gj_v[0];
                    if (gj_lane_in > 0) {
                        gj_v[0] = gj_scale_0;
                    }
                    float gj_scale_1 = -gj_v[1];
                    float gj_piv_0 = __shfl_sync(0xFFFFFFFF, gj_v[0], gj_sub + 1);
                    if (gj_lane_in > 1) {
                        gj_v[0] = __fmaf_rn(gj_scale_1, gj_piv_0, gj_v[0]);
                    }
                    if (gj_lane_in > 1) {
                        gj_v[1] = gj_scale_1;
                    }
                    float gj_scale_2 = -gj_v[2];
                    float gj_piv_1 = __shfl_sync(0xFFFFFFFF, gj_v[0], gj_sub + 2);
                    if (gj_lane_in > 2) {
                        gj_v[0] = __fmaf_rn(gj_scale_2, gj_piv_1, gj_v[0]);
                    }
                    float gj_piv_2 = __shfl_sync(0xFFFFFFFF, gj_v[1], gj_sub + 2);
                    if (gj_lane_in > 2) {
                        gj_v[1] = __fmaf_rn(gj_scale_2, gj_piv_2, gj_v[1]);
                    }
                    if (gj_lane_in > 2) {
                        gj_v[2] = gj_scale_2;
                    }
                    float gj_scale_3 = -gj_v[3];
                    float gj_piv_3 = __shfl_sync(0xFFFFFFFF, gj_v[0], gj_sub + 3);
                    if (gj_lane_in > 3) {
                        gj_v[0] = __fmaf_rn(gj_scale_3, gj_piv_3, gj_v[0]);
                    }
                    float gj_piv_4 = __shfl_sync(0xFFFFFFFF, gj_v[1], gj_sub + 3);
                    if (gj_lane_in > 3) {
                        gj_v[1] = __fmaf_rn(gj_scale_3, gj_piv_4, gj_v[1]);
                    }
                    float gj_piv_5 = __shfl_sync(0xFFFFFFFF, gj_v[2], gj_sub + 3);
                    if (gj_lane_in > 3) {
                        gj_v[2] = __fmaf_rn(gj_scale_3, gj_piv_5, gj_v[2]);
                    }
                    if (gj_lane_in > 3) {
                        gj_v[3] = gj_scale_3;
                    }
                    float gj_scale_4 = -gj_v[4];
                    float gj_piv_6 = __shfl_sync(0xFFFFFFFF, gj_v[0], gj_sub + 4);
                    if (gj_lane_in > 4) {
                        gj_v[0] = __fmaf_rn(gj_scale_4, gj_piv_6, gj_v[0]);
                    }
                    float gj_piv_7 = __shfl_sync(0xFFFFFFFF, gj_v[1], gj_sub + 4);
                    if (gj_lane_in > 4) {
                        gj_v[1] = __fmaf_rn(gj_scale_4, gj_piv_7, gj_v[1]);
                    }
                    float gj_piv_8 = __shfl_sync(0xFFFFFFFF, gj_v[2], gj_sub + 4);
                    if (gj_lane_in > 4) {
                        gj_v[2] = __fmaf_rn(gj_scale_4, gj_piv_8, gj_v[2]);
                    }
                    float gj_piv_9 = __shfl_sync(0xFFFFFFFF, gj_v[3], gj_sub + 4);
                    if (gj_lane_in > 4) {
                        gj_v[3] = __fmaf_rn(gj_scale_4, gj_piv_9, gj_v[3]);
                    }
                    if (gj_lane_in > 4) {
                        gj_v[4] = gj_scale_4;
                    }
                    float gj_scale_5 = -gj_v[5];
                    float gj_piv_10 = __shfl_sync(0xFFFFFFFF, gj_v[0], gj_sub + 5);
                    if (gj_lane_in > 5) {
                        gj_v[0] = __fmaf_rn(gj_scale_5, gj_piv_10, gj_v[0]);
                    }
                    float gj_piv_11 = __shfl_sync(0xFFFFFFFF, gj_v[1], gj_sub + 5);
                    if (gj_lane_in > 5) {
                        gj_v[1] = __fmaf_rn(gj_scale_5, gj_piv_11, gj_v[1]);
                    }
                    float gj_piv_12 = __shfl_sync(0xFFFFFFFF, gj_v[2], gj_sub + 5);
                    if (gj_lane_in > 5) {
                        gj_v[2] = __fmaf_rn(gj_scale_5, gj_piv_12, gj_v[2]);
                    }
                    float gj_piv_13 = __shfl_sync(0xFFFFFFFF, gj_v[3], gj_sub + 5);
                    if (gj_lane_in > 5) {
                        gj_v[3] = __fmaf_rn(gj_scale_5, gj_piv_13, gj_v[3]);
                    }
                    float gj_piv_14 = __shfl_sync(0xFFFFFFFF, gj_v[4], gj_sub + 5);
                    if (gj_lane_in > 5) {
                        gj_v[4] = __fmaf_rn(gj_scale_5, gj_piv_14, gj_v[4]);
                    }
                    if (gj_lane_in > 5) {
                        gj_v[5] = gj_scale_5;
                    }
                    float gj_scale_6 = -gj_v[6];
                    float gj_piv_15 = __shfl_sync(0xFFFFFFFF, gj_v[0], gj_sub + 6);
                    if (gj_lane_in > 6) {
                        gj_v[0] = __fmaf_rn(gj_scale_6, gj_piv_15, gj_v[0]);
                    }
                    float gj_piv_16 = __shfl_sync(0xFFFFFFFF, gj_v[1], gj_sub + 6);
                    if (gj_lane_in > 6) {
                        gj_v[1] = __fmaf_rn(gj_scale_6, gj_piv_16, gj_v[1]);
                    }
                    float gj_piv_17 = __shfl_sync(0xFFFFFFFF, gj_v[2], gj_sub + 6);
                    if (gj_lane_in > 6) {
                        gj_v[2] = __fmaf_rn(gj_scale_6, gj_piv_17, gj_v[2]);
                    }
                    float gj_piv_18 = __shfl_sync(0xFFFFFFFF, gj_v[3], gj_sub + 6);
                    if (gj_lane_in > 6) {
                        gj_v[3] = __fmaf_rn(gj_scale_6, gj_piv_18, gj_v[3]);
                    }
                    float gj_piv_19 = __shfl_sync(0xFFFFFFFF, gj_v[4], gj_sub + 6);
                    if (gj_lane_in > 6) {
                        gj_v[4] = __fmaf_rn(gj_scale_6, gj_piv_19, gj_v[4]);
                    }
                    float gj_piv_20 = __shfl_sync(0xFFFFFFFF, gj_v[5], gj_sub + 6);
                    if (gj_lane_in > 6) {
                        gj_v[5] = __fmaf_rn(gj_scale_6, gj_piv_20, gj_v[5]);
                    }
                    if (gj_lane_in > 6) {
                        gj_v[6] = gj_scale_6;
                    }
                    unsigned int gj_out[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(gj_v[_lp*2 + 0], gj_v[_lp*2+1 + 0]));
                        gj_out[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(inv_base0 + (unsigned int)(gj_row * 128 + ((gj_col0 * 2 >> 4 ^ gj_row & 7) << 4) + (gj_col0 * 2 & 15))), "r"(gj_out[0]), "r"(gj_out[1]), "r"(gj_out[2]), "r"(gj_out[3]) : "memory");
                    __syncwarp();
                    #pragma unroll
                    for (int l1b0 = 0; l1b0 < 2; l1b0++) {
                        int l1_r = lane & 7;
                        unsigned int l1_d[2];
                        unsigned int l1_c[1];
                        float l1_t[4];
                        unsigned int l1_tp[2];
                        unsigned int l1_a[1];
                        float l1_o[4];
                        unsigned int l1_op[2];
                        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                            : "=r"(l1_d[0])
                            : "r"(inv_base0 + (unsigned int)((sub0 * 32 + l1b0 * 16 + 8 + l1_r) * 128 + (((sub0 * 32 + l1b0 * 16 + 8) * 2 >> 4 ^ sub0 * 32 + l1b0 * 16 + 8 + l1_r & 7) << 4) + ((sub0 * 32 + l1b0 * 16 + 8) * 2 & 15)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                            : "=r"(l1_d[1])
                            : "r"(inv_base0 + (unsigned int)((sub0 * 32 + l1b0 * 16 + 8 + l1_r) * 128 + (((sub0 * 32 + l1b0 * 16 + 8) * 2 >> 4 ^ sub0 * 32 + l1b0 * 16 + 8 + l1_r & 7) << 4) + ((sub0 * 32 + l1b0 * 16 + 8) * 2 & 15)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                            : "=r"(l1_c[0])
                            : "r"(inv_base0 + (unsigned int)((sub0 * 32 + l1b0 * 16 + 8 + l1_r) * 128 + (((sub0 * 32 + l1b0 * 16) * 2 >> 4 ^ sub0 * 32 + l1b0 * 16 + 8 + l1_r & 7) << 4) + ((sub0 * 32 + l1b0 * 16) * 2 & 15)))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                            : "=f"(l1_t[0]), "=f"(l1_t[1]), "=f"(l1_t[2]), "=f"(l1_t[3])
                            : "r"(l1_d[0]), "r"(l1_d[1]), "r"(l1_c[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        const float2 _scale2_2 = {-1.0f, -1.0f};
                        #pragma unroll
                        for (int _ls = 0; _ls < 2; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(l1_t)[_ls], _scale2_2);
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(l1_t[_lp*2 + 0], l1_t[_lp*2+1 + 0]));
                            l1_tp[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                            : "=r"(l1_a[0])
                            : "r"(inv_base0 + (unsigned int)((sub0 * 32 + l1b0 * 16 + l1_r) * 128 + (((sub0 * 32 + l1b0 * 16) * 2 >> 4 ^ sub0 * 32 + l1b0 * 16 + l1_r & 7) << 4) + ((sub0 * 32 + l1b0 * 16) * 2 & 15)))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                            : "=f"(l1_o[0]), "=f"(l1_o[1]), "=f"(l1_o[2]), "=f"(l1_o[3])
                            : "r"(l1_tp[0]), "r"(l1_tp[1]), "r"(l1_a[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(l1_o[_lp*2 + 0], l1_o[_lp*2+1 + 0]));
                            l1_op[_lp] = *(uint32_t*)&_bf2;
                        }
                        __syncwarp();
                        uint32_t _stmatrix_addr_3 = static_cast<uint32_t>((unsigned long long)(inv_base0 + (unsigned int)((sub0 * 32 + l1b0 * 16 + 8 + l1_r) * 128 + (((sub0 * 32 + l1b0 * 16) * 2 >> 4 ^ sub0 * 32 + l1b0 * 16 + 8 + l1_r & 7) << 4) + ((sub0 * 32 + l1b0 * 16) * 2 & 15))));
                        asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
                            :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&l1_op[0]))
                            : "memory");
                    }
                    __syncwarp();
                    int l2_row = lane & 15;
                    int l2_col = lane / 16 * 8;
                    unsigned int l2_d[4];
                    unsigned int l2_c[4];
                    float l2_t[8];
                    unsigned int l2_tp[4];
                    unsigned int l2_a[4];
                    float l2_o[8];
                    unsigned int l2_op[4];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(l2_d[0]), "=r"(l2_d[1]), "=r"(l2_d[2]), "=r"(l2_d[3])
                        : "r"(inv_base0 + (unsigned int)((sub0 * 32 + 16 + l2_row) * 128 + (((sub0 * 32 + 16 + l2_col) * 2 >> 4 ^ sub0 * 32 + 16 + l2_row & 7) << 4) + ((sub0 * 32 + 16 + l2_col) * 2 & 15)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(l2_c[0]), "=r"(l2_c[1]), "=r"(l2_c[2]), "=r"(l2_c[3])
                        : "r"(inv_base0 + (unsigned int)((sub0 * 32 + 16 + l2_row) * 128 + (((sub0 * 32 + l2_col) * 2 >> 4 ^ sub0 * 32 + 16 + l2_row & 7) << 4) + ((sub0 * 32 + l2_col) * 2 & 15)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(l2_t[0]), "=f"(l2_t[1]), "=f"(l2_t[2]), "=f"(l2_t[3])
                        : "r"(l2_d[0]), "r"(l2_d[1]), "r"(l2_d[2]), "r"(l2_d[3]), "r"(l2_c[0]), "r"(l2_c[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(l2_t[4]), "=f"(l2_t[(4) + 1]), "=f"(l2_t[(4) + 2]), "=f"(l2_t[(4) + 3])
                        : "r"(l2_d[0]), "r"(l2_d[1]), "r"(l2_d[2]), "r"(l2_d[3]), "r"(l2_c[2]), "r"(l2_c[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    const float2 _scale2_4 = {-1.0f, -1.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(l2_t)[_ls], _scale2_4);
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(l2_t[_lp*2 + 0], l2_t[_lp*2+1 + 0]));
                        l2_tp[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(l2_a[0]), "=r"(l2_a[1]), "=r"(l2_a[2]), "=r"(l2_a[3])
                        : "r"(inv_base0 + (unsigned int)((sub0 * 32 + l2_row) * 128 + (((sub0 * 32 + l2_col) * 2 >> 4 ^ sub0 * 32 + l2_row & 7) << 4) + ((sub0 * 32 + l2_col) * 2 & 15)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(l2_o[0]), "=f"(l2_o[1]), "=f"(l2_o[2]), "=f"(l2_o[3])
                        : "r"(l2_tp[0]), "r"(l2_tp[1]), "r"(l2_tp[2]), "r"(l2_tp[3]), "r"(l2_a[0]), "r"(l2_a[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(l2_o[4]), "=f"(l2_o[(4) + 1]), "=f"(l2_o[(4) + 2]), "=f"(l2_o[(4) + 3])
                        : "r"(l2_tp[0]), "r"(l2_tp[1]), "r"(l2_tp[2]), "r"(l2_tp[3]), "r"(l2_a[2]), "r"(l2_a[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(l2_o[_lp*2 + 0], l2_o[_lp*2+1 + 0]));
                        l2_op[_lp] = *(uint32_t*)&_bf2;
                    }
                    __syncwarp();
                    uint32_t _stmatrix_addr_5 = static_cast<uint32_t>((unsigned long long)(inv_base0 + (unsigned int)((sub0 * 32 + 16 + l2_row) * 128 + (((sub0 * 32 + l2_col) * 2 >> 4 ^ sub0 * 32 + 16 + l2_row & 7) << 4) + ((sub0 * 32 + l2_col) * 2 & 15))));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&l2_op[0])), "r"(*reinterpret_cast<const uint32_t*>(&l2_op[1])), "r"(*reinterpret_cast<const uint32_t*>(&l2_op[2])), "r"(*reinterpret_cast<const uint32_t*>(&l2_op[3]))
                        : "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    int f3_row = lane & 15;
                    int f3_col = lane / 16 * 8;
                    int f3_rd = 32 + sub0 * 16 + f3_row;
                    unsigned int f3_d[8];
                    unsigned int f3_c[16];
                    float f3_t[16];
                    unsigned int f3_tp[8];
                    unsigned int f3_a[16];
                    float f3_o[16];
                    unsigned int f3_op[8];
                    #pragma unroll
                    for (int f3_vs = 0; f3_vs < 2; f3_vs++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(f3_d[f3_vs * 4]), "=r"(f3_d[f3_vs * 4 + 1]), "=r"(f3_d[f3_vs * 4 + 2]), "=r"(f3_d[f3_vs * 4 + 3])
                            : "r"(inv_base0 + (unsigned int)(f3_rd * 128 + (((32 + f3_vs * 16 + f3_col) * 2 >> 4 ^ f3_rd & 7) << 4) + ((32 + f3_vs * 16 + f3_col) * 2 & 15)))
                            : "memory");
                    }
                    #pragma unroll
                    for (int f3_cs = 0; f3_cs < 4; f3_cs++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(f3_c[f3_cs * 4]), "=r"(f3_c[f3_cs * 4 + 1]), "=r"(f3_c[f3_cs * 4 + 2]), "=r"(f3_c[f3_cs * 4 + 3])
                            : "r"(inv_base0 + (unsigned int)((32 + f3_cs / 2 * 16 + f3_row) * 128 + ((((f3_cs & 1) * 16 + f3_col) * 2 >> 4 ^ 32 + f3_cs / 2 * 16 + f3_row & 7) << 4) + (((f3_cs & 1) * 16 + f3_col) * 2 & 15)))
                            : "memory");
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(f3_t[0]), "=f"(f3_t[1]), "=f"(f3_t[2]), "=f"(f3_t[3])
                        : "r"(f3_d[0]), "r"(f3_d[1]), "r"(f3_d[2]), "r"(f3_d[3]), "r"(f3_c[0]), "r"(f3_c[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(f3_t[4]), "=f"(f3_t[(4) + 1]), "=f"(f3_t[(4) + 2]), "=f"(f3_t[(4) + 3])
                        : "r"(f3_d[0]), "r"(f3_d[1]), "r"(f3_d[2]), "r"(f3_d[3]), "r"(f3_c[2]), "r"(f3_c[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(f3_t[8]), "=f"(f3_t[(8) + 1]), "=f"(f3_t[(8) + 2]), "=f"(f3_t[(8) + 3])
                        : "r"(f3_d[0]), "r"(f3_d[1]), "r"(f3_d[2]), "r"(f3_d[3]), "r"(f3_c[4]), "r"(f3_c[(4) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(f3_t[12]), "=f"(f3_t[(12) + 1]), "=f"(f3_t[(12) + 2]), "=f"(f3_t[(12) + 3])
                        : "r"(f3_d[0]), "r"(f3_d[1]), "r"(f3_d[2]), "r"(f3_d[3]), "r"(f3_c[6]), "r"(f3_c[(6) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(f3_t[0]), "+f"(f3_t[1]), "+f"(f3_t[2]), "+f"(f3_t[3])
                        : "r"(f3_d[4]), "r"(f3_d[(4) + 1]), "r"(f3_d[(4) + 2]), "r"(f3_d[(4) + 3]), "r"(f3_c[8]), "r"(f3_c[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(f3_t[4]), "+f"(f3_t[(4) + 1]), "+f"(f3_t[(4) + 2]), "+f"(f3_t[(4) + 3])
                        : "r"(f3_d[4]), "r"(f3_d[(4) + 1]), "r"(f3_d[(4) + 2]), "r"(f3_d[(4) + 3]), "r"(f3_c[10]), "r"(f3_c[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(f3_t[8]), "+f"(f3_t[(8) + 1]), "+f"(f3_t[(8) + 2]), "+f"(f3_t[(8) + 3])
                        : "r"(f3_d[4]), "r"(f3_d[(4) + 1]), "r"(f3_d[(4) + 2]), "r"(f3_d[(4) + 3]), "r"(f3_c[12]), "r"(f3_c[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(f3_t[12]), "+f"(f3_t[(12) + 1]), "+f"(f3_t[(12) + 2]), "+f"(f3_t[(12) + 3])
                        : "r"(f3_d[4]), "r"(f3_d[(4) + 1]), "r"(f3_d[(4) + 2]), "r"(f3_d[(4) + 3]), "r"(f3_c[14]), "r"(f3_c[(14) + 1]));
                    const float2 _scale2_6 = {-1.0f, -1.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(f3_t)[_ls], _scale2_6);
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(f3_t[_lp*2 + 0], f3_t[_lp*2+1 + 0]));
                        f3_tp[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int f3_as = 0; f3_as < 4; f3_as++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(f3_a[f3_as * 4]), "=r"(f3_a[f3_as * 4 + 1]), "=r"(f3_a[f3_as * 4 + 2]), "=r"(f3_a[f3_as * 4 + 3])
                            : "r"(inv_base0 + (unsigned int)((f3_as / 2 * 16 + f3_row) * 128 + ((((f3_as & 1) * 16 + f3_col) * 2 >> 4 ^ f3_as / 2 * 16 + f3_row & 7) << 4) + (((f3_as & 1) * 16 + f3_col) * 2 & 15)))
                            : "memory");
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(f3_o[0]), "=f"(f3_o[1]), "=f"(f3_o[2]), "=f"(f3_o[3])
                        : "r"(f3_tp[0]), "r"(f3_tp[1]), "r"(f3_tp[2]), "r"(f3_tp[3]), "r"(f3_a[0]), "r"(f3_a[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(f3_o[4]), "=f"(f3_o[(4) + 1]), "=f"(f3_o[(4) + 2]), "=f"(f3_o[(4) + 3])
                        : "r"(f3_tp[0]), "r"(f3_tp[1]), "r"(f3_tp[2]), "r"(f3_tp[3]), "r"(f3_a[2]), "r"(f3_a[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(f3_o[8]), "=f"(f3_o[(8) + 1]), "=f"(f3_o[(8) + 2]), "=f"(f3_o[(8) + 3])
                        : "r"(f3_tp[0]), "r"(f3_tp[1]), "r"(f3_tp[2]), "r"(f3_tp[3]), "r"(f3_a[4]), "r"(f3_a[(4) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(f3_o[12]), "=f"(f3_o[(12) + 1]), "=f"(f3_o[(12) + 2]), "=f"(f3_o[(12) + 3])
                        : "r"(f3_tp[0]), "r"(f3_tp[1]), "r"(f3_tp[2]), "r"(f3_tp[3]), "r"(f3_a[6]), "r"(f3_a[(6) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(f3_o[0]), "+f"(f3_o[1]), "+f"(f3_o[2]), "+f"(f3_o[3])
                        : "r"(f3_tp[4]), "r"(f3_tp[(4) + 1]), "r"(f3_tp[(4) + 2]), "r"(f3_tp[(4) + 3]), "r"(f3_a[8]), "r"(f3_a[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(f3_o[4]), "+f"(f3_o[(4) + 1]), "+f"(f3_o[(4) + 2]), "+f"(f3_o[(4) + 3])
                        : "r"(f3_tp[4]), "r"(f3_tp[(4) + 1]), "r"(f3_tp[(4) + 2]), "r"(f3_tp[(4) + 3]), "r"(f3_a[10]), "r"(f3_a[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(f3_o[8]), "+f"(f3_o[(8) + 1]), "+f"(f3_o[(8) + 2]), "+f"(f3_o[(8) + 3])
                        : "r"(f3_tp[4]), "r"(f3_tp[(4) + 1]), "r"(f3_tp[(4) + 2]), "r"(f3_tp[(4) + 3]), "r"(f3_a[12]), "r"(f3_a[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(f3_o[12]), "+f"(f3_o[(12) + 1]), "+f"(f3_o[(12) + 2]), "+f"(f3_o[(12) + 3])
                        : "r"(f3_tp[4]), "r"(f3_tp[(4) + 1]), "r"(f3_tp[(4) + 2]), "r"(f3_tp[(4) + 3]), "r"(f3_a[14]), "r"(f3_a[(14) + 1]));
                    int f3_q = (lane & 3) * 2;
                    unsigned int f3_bw[8];
                    #pragma unroll
                    for (int f3_bj = 0; f3_bj < 4; f3_bj++) {
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&f3_bw[f3_bj * 2])), "=r"(*reinterpret_cast<uint32_t*>(&f3_bw[(f3_bj * 2) + 1]))
                            : "r"(smem_beta_addr + (unsigned int)((kk_gb0 + f3_bj * 8 + f3_q) * 4)));
                    }
                    #pragma unroll
                    for (int f3_p = 0; f3_p < 8; f3_p++) {
                        float2 _f32x2_mul_a_7 = make_float2(f3_o[f3_p * 2], f3_o[f3_p * 2 + 1]);
                        float2 _f32x2_mul_b_8 = make_float2(__uint_as_float(f3_bw[f3_p / 2 * 2]), __uint_as_float(f3_bw[f3_p / 2 * 2 + 1]));
                        float2 f3_m;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&f3_m) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
                        f3_o[f3_p * 2] = f3_m.x;
                        f3_o[f3_p * 2 + 1] = f3_m.y;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(f3_o[_lp*2 + 0], f3_o[_lp*2+1 + 0]));
                        f3_op[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    uint32_t _stmatrix_addr_9 = static_cast<uint32_t>((unsigned long long)(inv_base0 + (unsigned int)(f3_rd * 128 + ((f3_col * 2 >> 4 ^ f3_rd & 7) << 4) + (f3_col * 2 & 15))));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&f3_op[0])), "r"(*reinterpret_cast<const uint32_t*>(&f3_op[1])), "r"(*reinterpret_cast<const uint32_t*>(&f3_op[2])), "r"(*reinterpret_cast<const uint32_t*>(&f3_op[3]))
                        : "memory");
                    uint32_t _stmatrix_addr_10 = static_cast<uint32_t>((unsigned long long)(inv_base0 + (unsigned int)(f3_rd * 128 + (((16 + f3_col) * 2 >> 4 ^ f3_rd & 7) << 4) + ((16 + f3_col) * 2 & 15))));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&f3_op[4])), "r"(*reinterpret_cast<const uint32_t*>(&f3_op[5])), "r"(*reinterpret_cast<const uint32_t*>(&f3_op[6])), "r"(*reinterpret_cast<const uint32_t*>(&f3_op[7]))
                        : "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    #pragma unroll
                    for (int bh7e = 0; bh7e < 2; bh7e++) {
                        int b7d_row = sub0 * 32 + bh7e * 16 + (lane & 15);
                        int b7d_col = lane / 16 * 8;
                        int b7d_q = (lane & 3) * 2;
                        int b7d_cb = (sub0 * 32 + bh7e * 16) / 32 * 2;
                        unsigned int b7d_f[8];
                        #pragma unroll
                        for (int b7d_c = 0; b7d_c < 2; b7d_c++) {
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(b7d_f[b7d_c * 4]), "=r"(b7d_f[b7d_c * 4 + 1]), "=r"(b7d_f[b7d_c * 4 + 2]), "=r"(b7d_f[b7d_c * 4 + 3])
                                : "r"(inv_base0 + (unsigned int)(b7d_row * 128 + ((((b7d_cb + b7d_c) * 16 + b7d_col) * 2 >> 4 ^ b7d_row & 7) << 4) + (((b7d_cb + b7d_c) * 16 + b7d_col) * 2 & 15)))
                                : "memory");
                        }
                        unsigned int b7d_bw[8];
                        #pragma unroll
                        for (int b7d_k = 0; b7d_k < 4; b7d_k++) {
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&b7d_bw[b7d_k * 2])), "=r"(*reinterpret_cast<uint32_t*>(&b7d_bw[(b7d_k * 2) + 1]))
                                : "r"(smem_beta_addr + (unsigned int)((kk_gb0 + b7d_cb * 16 + b7d_k * 8 + b7d_q) * 4)));
                        }
                        float b7d_s[16];
                        #pragma unroll
                        for (int b7d_j = 0; b7d_j < 8; b7d_j++) {
                            float b7d_vals[2];
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&b7d_vals[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((b7d_f + b7d_j)[_pair]) << 16);
                                (&b7d_vals[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((b7d_f + b7d_j)[_pair]) & 0xffff0000u);
                            }
                            float2 _f32x2_mul_a_11 = make_float2(b7d_vals[0], b7d_vals[1]);
                            float2 _f32x2_mul_b_12 = make_float2(__uint_as_float(b7d_bw[b7d_j / 2 * 2]), __uint_as_float(b7d_bw[b7d_j / 2 * 2 + 1]));
                            float2 b7d_m;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&b7d_m) : "l"(*(const unsigned long long*)&_f32x2_mul_a_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
                            b7d_s[b7d_j * 2] = b7d_m.x;
                            b7d_s[b7d_j * 2 + 1] = b7d_m.y;
                        }
                        unsigned int b7d_o[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(b7d_s[_lp*2 + 0], b7d_s[_lp*2+1 + 0]));
                            b7d_o[_lp] = *(uint32_t*)&_bf2;
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int b7d_c2 = 0; b7d_c2 < 2; b7d_c2++) {
                            uint32_t _stmatrix_addr_13 = static_cast<uint32_t>((unsigned long long)(inv_base0 + (unsigned int)(b7d_row * 128 + ((((b7d_cb + b7d_c2) * 16 + b7d_col) * 2 >> 4 ^ b7d_row & 7) << 4) + (((b7d_cb + b7d_c2) * 16 + b7d_col) * 2 & 15))));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&b7d_o[b7d_c2 * 4])), "r"(*reinterpret_cast<const uint32_t*>(&b7d_o[b7d_c2 * 4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&b7d_o[b7d_c2 * 4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&b7d_o[b7d_c2 * 4 + 3]))
                                : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(tinv_ready_addr + (tinv_cur0) * 8);
                    if (has_m1_0 != 0) {
                        mbarrier_arrive(tinv_ready_addr + (((tinv_cur0 + 1 < 3) ? tinv_cur0 + 1 : (unsigned int)0)) * 8);
                    }
                    int a_m0 = 1 - half0;
                    int do_a0 = ((a_m0 == 0 || has_m1_0 != 0) ? 1 : 0);
                    unsigned int a_slot0 = ((a_m0 != 0) ? a_cur01 : a_cur0);
                    unsigned int a_acc_slot0 = ((a_m0 != 0) ? acc_cur01 : acc_cur0);
                    if (do_a0 != 0) {
                        const bool _selected_wait_cond_14 = a_m0 != 0;
                        mbarrier_wait((_selected_wait_cond_14 ? (a_empty_addr + (((a_cur0 + 1 < 3) ? a_cur0 + 1 : (unsigned int)0)) * 8) : (a_empty_addr + (a_cur0) * 8)), (_selected_wait_cond_14 ? (((a_cur0 + 1 < 3) ? _phase_a_empty : _phase_a_empty ^ 1)) : (_phase_a_empty)));
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float qa3[32];
                    float qb3[32];
                    if (do_a0 != 0) {
                        unsigned int a_cols3 = 256 + 64 * a_acc_slot0;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qa3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[1])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[2])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[3])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[4])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[5])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[6])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[7])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[8])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[9])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[10])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[11])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[12])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[13])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[14])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[15])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[16])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[17])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[18])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[19])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[20])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[21])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[22])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[23])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[24])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[25])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[26])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[27])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[28])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[29])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[30])), "=r"(*reinterpret_cast<uint32_t*>(&qa3[31]))
                            : "r"(taddr + a_cols3 + (unsigned int)(warp0 * 32 << 16)));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qb3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[1])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[2])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[3])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[4])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[5])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[6])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[7])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[8])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[9])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[10])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[11])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[12])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[13])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[14])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[15])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[16])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[17])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[18])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[19])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[20])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[21])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[22])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[23])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[24])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[25])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[26])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[27])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[28])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[29])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[30])), "=r"(*reinterpret_cast<uint32_t*>(&qb3[31]))
                            : "r"(taddr + a_cols3 + (unsigned int)(warp0 * 32 + 16 << 16)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (do_a0 != 0) {
                        if (a_m0 != 0) {
                            mbarrier_arrive(acc_done_addr + (((acc_cur0 + 1 < 2) ? acc_cur0 + 1 : (unsigned int)0)) * 8);
                        } else {
                            mbarrier_arrive(acc_done_addr + (acc_cur0) * 8);
                        }
                    }
                    if (do_a0 != 0) {
                        unsigned int a_dst3 = smem_a_addr + a_slot0 * 8192;
                        #pragma unroll 1
                        for (int ah3 = 0; ah3 < 2; ah3++) {
                            int a7_lane_quad = lane & 3;
                            int a7_stsm_row = sub0 * 32 + ah3 * 16 + (lane & 7) + (lane & 8);
                            int a7_stsm_col = (lane & 16) / 2;
                            int a7_t0 = sub0 * 32 + ah3 * 16 + lane / 4;
                            int a7_t1 = a7_t0 + 8;
                            float a7_ct0 = smem_ct[((a_m0 != 0) ? gbase01 : gbase00) + a7_t0];
                            float a7_ct1 = smem_ct[((a_m0 != 0) ? gbase01 : gbase00) + a7_t1];
                            int a7_e0 = a7_t0 * num_hh + num_hh - 1 - ((a_m0 != 0) ? cb01 : cb00) * 64 - a7_lane_quad * 2;
                            int a7_e1 = a7_t1 * num_hh + num_hh - 1 - ((a_m0 != 0) ? cb01 : cb00) * 64 - a7_lane_quad * 2;
                            a7_e0 = ((a7_t0 < ((a_m0 != 0) ? vt01 : vt00)) ? a7_e0 : -1);
                            a7_e1 = ((a7_t1 < ((a_m0 != 0) ? vt01 : vt00)) ? a7_e1 : -1);
                            unsigned int a7_bw[16];
                            #pragma unroll
                            for (int a7_k = 0; a7_k < 8; a7_k++) {
                                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&a7_bw[a7_k * 2])), "=r"(*reinterpret_cast<uint32_t*>(&a7_bw[(a7_k * 2) + 1]))
                                    : "r"(smem_blk_addr + (unsigned int)((((a_m0 != 0) ? gbase01 : gbase00) + a7_k * 8 + a7_lane_quad * 2) * 4)));
                            }
                            float a7_x[32];
                            #pragma unroll
                            for (int a7_p = 0; a7_p < 16; a7_p++) {
                                float a7_ct = (((a7_p & 1) != 0) ? a7_ct1 : a7_ct0);
                                float2 a7_d = sub_f32x2(make_float2(a7_ct, a7_ct), make_float2(__uint_as_float(a7_bw[a7_p / 2 * 2]), __uint_as_float(a7_bw[a7_p / 2 * 2 + 1])));
                                float _exp2_15 = approx_exp2(a7_d.x);
                                a7_x[a7_p * 2] = _exp2_15;
                                float _exp2_16 = approx_exp2(a7_d.y);
                                a7_x[a7_p * 2 + 1] = _exp2_16;
                            }
                            float2 a7_sc = make_float2(scale, scale);
                            float at7[32];
                            #pragma unroll
                            for (int a7_q = 0; a7_q < 16; a7_q++) {
                                int a7_e = (((a7_q & 1) != 0) ? a7_e1 : a7_e0);
                                float2 _f32x2_mul_a_17 = make_float2(qa3[a7_q * 2], qa3[a7_q * 2 + 1]);
                                float2 _f32x2_mul_b_18 = make_float2(a7_x[a7_q * 2], a7_x[a7_q * 2 + 1]);
                                float2 a7_m;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a7_m) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
                                float2 a7_v;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a7_v) : "l"(*(const unsigned long long*)&a7_m), "l"(*(const unsigned long long*)&a7_sc));
                                at7[a7_q * 2] = ((a7_q / 2 * 8 <= a7_e) ? a7_v.x : 0.0f);
                                at7[a7_q * 2 + 1] = ((a7_q / 2 * 8 + 1 <= a7_e) ? a7_v.y : 0.0f);
                            }
                            #pragma unroll
                            for (int a7_r = 0; a7_r < 4; a7_r++) {
                                unsigned int at7b[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((at7 + a7_r * 8)[_lp*2 + 0], (at7 + a7_r * 8)[_lp*2+1 + 0]));
                                    at7b[_lp] = *(uint32_t*)&_bf2;
                                }
                                unsigned int a7_addr = a_dst3 + (unsigned int)(a7_stsm_row * 128 + (((a7_r * 16 + a7_stsm_col) * 2 >> 4 ^ a7_stsm_row & 7) << 4) + ((a7_r * 16 + a7_stsm_col) * 2 & 15));
                                uint32_t _stmatrix_addr_19 = static_cast<uint32_t>((unsigned long long)a7_addr);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&at7b[0])), "r"(*reinterpret_cast<const uint32_t*>(&at7b[1])), "r"(*reinterpret_cast<const uint32_t*>(&at7b[2])), "r"(*reinterpret_cast<const uint32_t*>(&at7b[3]))
                                    : "memory");
                            }
                            #pragma unroll
                            for (int j3 = 0; j3 < 32; j3++) {
                                qa3[j3] = qb3[j3];
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (do_a0 != 0) {
                        if (a_m0 != 0) {
                            mbarrier_arrive(a_ready_addr + (((a_cur0 + 1 < 3) ? a_cur0 + 1 : (unsigned int)0)) * 8);
                        } else {
                            mbarrier_arrive(a_ready_addr + (a_cur0) * 8);
                        }
                    }
                    mbarrier_arrive(gate_empty_cg0_addr + (g_cur0) * 8);
                    g_cur0 += 1;
                    if (g_cur0 == 4) { g_cur0 = 0; _phase_gate_full ^= 1; }
                    tinv_cur0 += 1;
                    if (tinv_cur0 == 3) { tinv_cur0 = 0; _phase_tinv_empty ^= 1; }
                    a_cur0 += 1;
                    if (a_cur0 == 3) { a_cur0 = 0; _phase_a_empty ^= 1; }
                    acc_cur0 += 1;
                    if (acc_cur0 == 2) { acc_cur0 = 0; _phase_acc_ready ^= 1; }
                    if (has_m1_0 != 0) {
                        mbarrier_arrive(gate_empty_cg0_addr + (g_cur0) * 8);
                        g_cur0 += 1;
                        if (g_cur0 == 4) { g_cur0 = 0; _phase_gate_full ^= 1; }
                        tinv_cur0 += 1;
                        if (tinv_cur0 == 3) { tinv_cur0 = 0; _phase_tinv_empty ^= 1; }
                        a_cur0 += 1;
                        if (a_cur0 == 3) { a_cur0 = 0; _phase_a_empty ^= 1; }
                        acc_cur0 += 1;
                        if (acc_cur0 == 2) { acc_cur0 = 0; _phase_acc_ready ^= 1; }
                    }
                    blk0c = ((cb01 == num_hh - 1) ? blk01 + 1 : blk01);
                    cb0c = ((cb01 == num_hh - 1) ? 0 : cb01 + 1);
                }
                if ((nch & 1) != 0) {
                    mbarrier_wait(acc_ready_addr + (acc_cur0) * 8, _phase_acc_ready);
                    if (half0 == 0) {
                        mbarrier_arrive(acc_done_addr + (acc_cur0) * 8);
                    }
                    acc_cur0 += 1;
                    if (acc_cur0 == 2) { acc_cur0 = 0; _phase_acc_ready ^= 1; }
                }
            }
        }
    }
    // ---- Role: cg1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 240;");
        { // cg1_main
            int warp_id_in_role_1 = warp - 4;
            int warp1 = warp_id_in_role_1;
            int row_base1 = warp1 * 32 << 16;
            int lane_quad1 = lane & 3;
            int lane_row1 = lane / 4;
            int row_top1 = warp1 * 16 + lane_row1;
            int mtx1 = lane / 8;
            int row_in_mtx1 = lane & 7;
            unsigned int g_cur1 = 0;
            unsigned int v_cur1 = 0;
            int pl_lo = items[num_items * 4 + (int)bid];
            int pl_hi = items[num_items * 4 + (int)bid + 1];
            if (warp1 == 0 && lane == 0) {
                smem_scan[1] = pl_hi;
                smem_scan[2] = pl_lo;
            }
            int pl_nb = 1;
            int pl_echo0 = num_seqs * num_heads;
            #pragma unroll 1
            for (int pl_it0 = pl_lo + warp1 * 32 * pl_nb; pl_it0 < pl_hi; pl_it0 += 128 * pl_nb) {
                int pl_it = pl_it0 + lane * pl_nb;
                int pl_need = 0;
                if (pl_it < pl_hi) {
                    int pl_seq = items[pl_it * 4];
                    int pl_cut = items[pl_it * 4 + 2];
                    int pl_span = items[pl_it * 4 + 3];
                    int pl_rs = cu_seqlens[pl_seq];
                    int pl_re = cu_seqlens[pl_seq + 1];
                    int pl_s0 = ((pl_rs < 0) ? 0 : ((pl_rs > total_tokens) ? total_tokens : pl_rs));
                    int pl_e0 = ((pl_re < pl_s0) ? pl_s0 : ((pl_re > total_tokens) ? total_tokens : pl_re));
                    int pl_chunks = 2 * (((pl_e0 - pl_s0) * num_hh + 127) / 128);
                    int pl_one = ((pl_cut == 0 && (pl_span == 0 || pl_span >= pl_chunks)) ? 1 : 0);
                    if (pl_one != 0) {
                        unsigned int pl_lo0 = 0;
                        unsigned int pl_n0 = (unsigned int)pl_chunks;
                        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_ws) + (pl_echo0 + pl_it * 2)) + (0)) = pl_lo0;
                        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_ws) + (pl_echo0 + pl_it * 2 + 1)) + (0)) = pl_n0;
                    }
                    pl_need = 1 - pl_one;
                }
                unsigned int pl_mask = __ballot_sync(0xFFFFFFFF, pl_need != 0);
                #pragma unroll 1
                for (int pl_j = 0; pl_j < 32; pl_j++) {
                    if ((pl_mask >> (unsigned int)pl_j & 1) != 0) {
                        int pj_it = pl_it0 + pl_j * pl_nb;
                        int pj_seq = items[pj_it * 4];
                        int pj_head = items[pj_it * 4 + 1];
                        int pj_cut = items[pj_it * 4 + 2];
                        int pj_span = items[pj_it * 4 + 3];
                        int pj_rs = cu_seqlens[pj_seq];
                        int pj_re = cu_seqlens[pj_seq + 1];
                        int pj_s0 = ((pj_rs < 0) ? 0 : ((pj_rs > total_tokens) ? total_tokens : pj_rs));
                        int pj_e0 = ((pj_re < pj_s0) ? pj_s0 : ((pj_re > total_tokens) ? total_tokens : pj_re));
                        int pj_chunks = 2 * (((pj_e0 - pj_s0) * num_hh + 127) / 128);
                        int pj_sp = ((pj_span < 0) ? -pj_span : pj_span);
                        int pj_q = (((num_hh & 1) == 0) ? num_hh : 2 * num_hh);
                        int pj_cp = (pj_chunks + pj_q - 1) / pj_q * pj_q;
                        int pj_ph = 0;
                        if (pj_span < 0) {
                            pj_ph = pj_head * pj_cp - pj_head * pj_cp / pj_sp * pj_sp;
                        }
                        int pj_x = ((pj_cut > 0) ? pj_cut : pj_sp - pj_ph);
                        int pj_fw = ((pj_cut > 0) ? 0 : 1);
                        int pj_warm = 0;
                        int pj_sat = ((pj_cut == 0) ? 1 : 0);
                        int pj_wend = pj_chunks;
                        int pj_go = ((pj_span < 0 && pj_cut >= pj_chunks) ? 0 : 1);
                        while (pj_go != 0) {
                            pj_go = ((pj_fw == 0 || pj_fw <= 48 && pj_x < pj_chunks && pj_span != 0) ? 1 : 0);
                            if (pj_go != 0) {
                                int pk_lane = lane;
                                float pk_acc = 0.0f;
                                int w = 0;
                                int pk_b = 0;
                                int pk_go = 1;
                                while (pk_go != 0) {
                                    int pk_x0 = pj_x - pk_b * 8;
                                    float pk_g[16];
                                    int pk_cnt[16];
                                    int pk_row_0 = (pk_x0 - 1) * 64 + pk_lane;
                                    int pk_tok_0 = pk_row_0 / num_hh;
                                    pk_cnt[0] = ((pk_x0 - 1 >= 0 && pk_row_0 - pk_tok_0 * num_hh == 0 && pk_tok_0 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[0] = 1.0f;
                                    if (pk_cnt[0] != 0) {
                                        pk_g[0] = gate[(pj_s0 + pk_tok_0) * num_heads + pj_head];
                                    }
                                    int pk_row_1 = (pk_x0 - 1) * 64 + pk_lane + 32;
                                    int pk_tok_1 = pk_row_1 / num_hh;
                                    pk_cnt[1] = ((pk_x0 - 1 >= 0 && pk_row_1 - pk_tok_1 * num_hh == 0 && pk_tok_1 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[1] = 1.0f;
                                    if (pk_cnt[1] != 0) {
                                        pk_g[1] = gate[(pj_s0 + pk_tok_1) * num_heads + pj_head];
                                    }
                                    int pk_row_2 = (pk_x0 - 1 - 1) * 64 + pk_lane;
                                    int pk_tok_2 = pk_row_2 / num_hh;
                                    pk_cnt[2] = ((pk_x0 - 1 - 1 >= 0 && pk_row_2 - pk_tok_2 * num_hh == 0 && pk_tok_2 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[2] = 1.0f;
                                    if (pk_cnt[2] != 0) {
                                        pk_g[2] = gate[(pj_s0 + pk_tok_2) * num_heads + pj_head];
                                    }
                                    int pk_row_3 = (pk_x0 - 1 - 1) * 64 + pk_lane + 32;
                                    int pk_tok_3 = pk_row_3 / num_hh;
                                    pk_cnt[3] = ((pk_x0 - 1 - 1 >= 0 && pk_row_3 - pk_tok_3 * num_hh == 0 && pk_tok_3 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[3] = 1.0f;
                                    if (pk_cnt[3] != 0) {
                                        pk_g[3] = gate[(pj_s0 + pk_tok_3) * num_heads + pj_head];
                                    }
                                    int pk_row_4 = (pk_x0 - 1 - 2) * 64 + pk_lane;
                                    int pk_tok_4 = pk_row_4 / num_hh;
                                    pk_cnt[4] = ((pk_x0 - 1 - 2 >= 0 && pk_row_4 - pk_tok_4 * num_hh == 0 && pk_tok_4 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[4] = 1.0f;
                                    if (pk_cnt[4] != 0) {
                                        pk_g[4] = gate[(pj_s0 + pk_tok_4) * num_heads + pj_head];
                                    }
                                    int pk_row_5 = (pk_x0 - 1 - 2) * 64 + pk_lane + 32;
                                    int pk_tok_5 = pk_row_5 / num_hh;
                                    pk_cnt[5] = ((pk_x0 - 1 - 2 >= 0 && pk_row_5 - pk_tok_5 * num_hh == 0 && pk_tok_5 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[5] = 1.0f;
                                    if (pk_cnt[5] != 0) {
                                        pk_g[5] = gate[(pj_s0 + pk_tok_5) * num_heads + pj_head];
                                    }
                                    int pk_row_6 = (pk_x0 - 1 - 3) * 64 + pk_lane;
                                    int pk_tok_6 = pk_row_6 / num_hh;
                                    pk_cnt[6] = ((pk_x0 - 1 - 3 >= 0 && pk_row_6 - pk_tok_6 * num_hh == 0 && pk_tok_6 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[6] = 1.0f;
                                    if (pk_cnt[6] != 0) {
                                        pk_g[6] = gate[(pj_s0 + pk_tok_6) * num_heads + pj_head];
                                    }
                                    int pk_row_7 = (pk_x0 - 1 - 3) * 64 + pk_lane + 32;
                                    int pk_tok_7 = pk_row_7 / num_hh;
                                    pk_cnt[7] = ((pk_x0 - 1 - 3 >= 0 && pk_row_7 - pk_tok_7 * num_hh == 0 && pk_tok_7 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[7] = 1.0f;
                                    if (pk_cnt[7] != 0) {
                                        pk_g[7] = gate[(pj_s0 + pk_tok_7) * num_heads + pj_head];
                                    }
                                    int pk_row_8 = (pk_x0 - 1 - 4) * 64 + pk_lane;
                                    int pk_tok_8 = pk_row_8 / num_hh;
                                    pk_cnt[8] = ((pk_x0 - 1 - 4 >= 0 && pk_row_8 - pk_tok_8 * num_hh == 0 && pk_tok_8 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[8] = 1.0f;
                                    if (pk_cnt[8] != 0) {
                                        pk_g[8] = gate[(pj_s0 + pk_tok_8) * num_heads + pj_head];
                                    }
                                    int pk_row_9 = (pk_x0 - 1 - 4) * 64 + pk_lane + 32;
                                    int pk_tok_9 = pk_row_9 / num_hh;
                                    pk_cnt[9] = ((pk_x0 - 1 - 4 >= 0 && pk_row_9 - pk_tok_9 * num_hh == 0 && pk_tok_9 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[9] = 1.0f;
                                    if (pk_cnt[9] != 0) {
                                        pk_g[9] = gate[(pj_s0 + pk_tok_9) * num_heads + pj_head];
                                    }
                                    int pk_row_10 = (pk_x0 - 1 - 5) * 64 + pk_lane;
                                    int pk_tok_10 = pk_row_10 / num_hh;
                                    pk_cnt[10] = ((pk_x0 - 1 - 5 >= 0 && pk_row_10 - pk_tok_10 * num_hh == 0 && pk_tok_10 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[10] = 1.0f;
                                    if (pk_cnt[10] != 0) {
                                        pk_g[10] = gate[(pj_s0 + pk_tok_10) * num_heads + pj_head];
                                    }
                                    int pk_row_11 = (pk_x0 - 1 - 5) * 64 + pk_lane + 32;
                                    int pk_tok_11 = pk_row_11 / num_hh;
                                    pk_cnt[11] = ((pk_x0 - 1 - 5 >= 0 && pk_row_11 - pk_tok_11 * num_hh == 0 && pk_tok_11 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[11] = 1.0f;
                                    if (pk_cnt[11] != 0) {
                                        pk_g[11] = gate[(pj_s0 + pk_tok_11) * num_heads + pj_head];
                                    }
                                    int pk_row_12 = (pk_x0 - 1 - 6) * 64 + pk_lane;
                                    int pk_tok_12 = pk_row_12 / num_hh;
                                    pk_cnt[12] = ((pk_x0 - 1 - 6 >= 0 && pk_row_12 - pk_tok_12 * num_hh == 0 && pk_tok_12 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[12] = 1.0f;
                                    if (pk_cnt[12] != 0) {
                                        pk_g[12] = gate[(pj_s0 + pk_tok_12) * num_heads + pj_head];
                                    }
                                    int pk_row_13 = (pk_x0 - 1 - 6) * 64 + pk_lane + 32;
                                    int pk_tok_13 = pk_row_13 / num_hh;
                                    pk_cnt[13] = ((pk_x0 - 1 - 6 >= 0 && pk_row_13 - pk_tok_13 * num_hh == 0 && pk_tok_13 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[13] = 1.0f;
                                    if (pk_cnt[13] != 0) {
                                        pk_g[13] = gate[(pj_s0 + pk_tok_13) * num_heads + pj_head];
                                    }
                                    int pk_row_14 = (pk_x0 - 1 - 7) * 64 + pk_lane;
                                    int pk_tok_14 = pk_row_14 / num_hh;
                                    pk_cnt[14] = ((pk_x0 - 1 - 7 >= 0 && pk_row_14 - pk_tok_14 * num_hh == 0 && pk_tok_14 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[14] = 1.0f;
                                    if (pk_cnt[14] != 0) {
                                        pk_g[14] = gate[(pj_s0 + pk_tok_14) * num_heads + pj_head];
                                    }
                                    int pk_row_15 = (pk_x0 - 1 - 7) * 64 + pk_lane + 32;
                                    int pk_tok_15 = pk_row_15 / num_hh;
                                    pk_cnt[15] = ((pk_x0 - 1 - 7 >= 0 && pk_row_15 - pk_tok_15 * num_hh == 0 && pk_tok_15 < pj_e0 - pj_s0) ? 1 : 0);
                                    pk_g[15] = 1.0f;
                                    if (pk_cnt[15] != 0) {
                                        pk_g[15] = gate[(pj_s0 + pk_tok_15) * num_heads + pj_head];
                                    }
                                    float pk_part_0 = 0.0f;
                                    float pk_lg_0;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_0) : "f"(pk_g[0] + 1e-10f));
                                    pk_part_0 = pk_part_0 + ((pk_cnt[0] != 0) ? pk_lg_0 : 0.0f);
                                    float pk_lg_1;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_1) : "f"(pk_g[1] + 1e-10f));
                                    pk_part_0 = pk_part_0 + ((pk_cnt[1] != 0) ? pk_lg_1 : 0.0f);
                                    float pk_o_0 = __shfl_sync(0xFFFFFFFF, pk_part_0, pk_lane ^ 1);
                                    pk_part_0 = pk_part_0 + pk_o_0;
                                    float pk_o_1 = __shfl_sync(0xFFFFFFFF, pk_part_0, pk_lane ^ 2);
                                    pk_part_0 = pk_part_0 + pk_o_1;
                                    float pk_o_2 = __shfl_sync(0xFFFFFFFF, pk_part_0, pk_lane ^ 4);
                                    pk_part_0 = pk_part_0 + pk_o_2;
                                    float pk_o_3 = __shfl_sync(0xFFFFFFFF, pk_part_0, pk_lane ^ 8);
                                    pk_part_0 = pk_part_0 + pk_o_3;
                                    float pk_o_4 = __shfl_sync(0xFFFFFFFF, pk_part_0, pk_lane ^ 16);
                                    pk_part_0 = pk_part_0 + pk_o_4;
                                    if (w == 0 && pk_x0 - 1 >= 0) {
                                        pk_acc = pk_acc + pk_part_0;
                                        int pk_hit_0 = __any_sync(0xFFFFFFFF, pk_acc <= -14.43f);
                                        if (pk_hit_0 != 0) {
                                            w = pk_b * 8 + 1;
                                        }
                                    }
                                    float pk_part_1 = 0.0f;
                                    float pk_lg_2;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_2) : "f"(pk_g[2] + 1e-10f));
                                    pk_part_1 = pk_part_1 + ((pk_cnt[2] != 0) ? pk_lg_2 : 0.0f);
                                    float pk_lg_3;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_3) : "f"(pk_g[3] + 1e-10f));
                                    pk_part_1 = pk_part_1 + ((pk_cnt[3] != 0) ? pk_lg_3 : 0.0f);
                                    float pk_o_5 = __shfl_sync(0xFFFFFFFF, pk_part_1, pk_lane ^ 1);
                                    pk_part_1 = pk_part_1 + pk_o_5;
                                    float pk_o_6 = __shfl_sync(0xFFFFFFFF, pk_part_1, pk_lane ^ 2);
                                    pk_part_1 = pk_part_1 + pk_o_6;
                                    float pk_o_7 = __shfl_sync(0xFFFFFFFF, pk_part_1, pk_lane ^ 4);
                                    pk_part_1 = pk_part_1 + pk_o_7;
                                    float pk_o_8 = __shfl_sync(0xFFFFFFFF, pk_part_1, pk_lane ^ 8);
                                    pk_part_1 = pk_part_1 + pk_o_8;
                                    float pk_o_9 = __shfl_sync(0xFFFFFFFF, pk_part_1, pk_lane ^ 16);
                                    pk_part_1 = pk_part_1 + pk_o_9;
                                    if (w == 0 && pk_x0 - 1 - 1 >= 0) {
                                        pk_acc = pk_acc + pk_part_1;
                                        int pk_hit_1 = __any_sync(0xFFFFFFFF, pk_acc <= -14.43f);
                                        if (pk_hit_1 != 0) {
                                            w = pk_b * 8 + 1 + 1;
                                        }
                                    }
                                    float pk_part_2 = 0.0f;
                                    float pk_lg_4;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_4) : "f"(pk_g[4] + 1e-10f));
                                    pk_part_2 = pk_part_2 + ((pk_cnt[4] != 0) ? pk_lg_4 : 0.0f);
                                    float pk_lg_5;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_5) : "f"(pk_g[5] + 1e-10f));
                                    pk_part_2 = pk_part_2 + ((pk_cnt[5] != 0) ? pk_lg_5 : 0.0f);
                                    float pk_o_10 = __shfl_sync(0xFFFFFFFF, pk_part_2, pk_lane ^ 1);
                                    pk_part_2 = pk_part_2 + pk_o_10;
                                    float pk_o_11 = __shfl_sync(0xFFFFFFFF, pk_part_2, pk_lane ^ 2);
                                    pk_part_2 = pk_part_2 + pk_o_11;
                                    float pk_o_12 = __shfl_sync(0xFFFFFFFF, pk_part_2, pk_lane ^ 4);
                                    pk_part_2 = pk_part_2 + pk_o_12;
                                    float pk_o_13 = __shfl_sync(0xFFFFFFFF, pk_part_2, pk_lane ^ 8);
                                    pk_part_2 = pk_part_2 + pk_o_13;
                                    float pk_o_14 = __shfl_sync(0xFFFFFFFF, pk_part_2, pk_lane ^ 16);
                                    pk_part_2 = pk_part_2 + pk_o_14;
                                    if (w == 0 && pk_x0 - 1 - 2 >= 0) {
                                        pk_acc = pk_acc + pk_part_2;
                                        int pk_hit_2 = __any_sync(0xFFFFFFFF, pk_acc <= -14.43f);
                                        if (pk_hit_2 != 0) {
                                            w = pk_b * 8 + 2 + 1;
                                        }
                                    }
                                    float pk_part_3 = 0.0f;
                                    float pk_lg_6;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_6) : "f"(pk_g[6] + 1e-10f));
                                    pk_part_3 = pk_part_3 + ((pk_cnt[6] != 0) ? pk_lg_6 : 0.0f);
                                    float pk_lg_7;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_7) : "f"(pk_g[7] + 1e-10f));
                                    pk_part_3 = pk_part_3 + ((pk_cnt[7] != 0) ? pk_lg_7 : 0.0f);
                                    float pk_o_15 = __shfl_sync(0xFFFFFFFF, pk_part_3, pk_lane ^ 1);
                                    pk_part_3 = pk_part_3 + pk_o_15;
                                    float pk_o_16 = __shfl_sync(0xFFFFFFFF, pk_part_3, pk_lane ^ 2);
                                    pk_part_3 = pk_part_3 + pk_o_16;
                                    float pk_o_17 = __shfl_sync(0xFFFFFFFF, pk_part_3, pk_lane ^ 4);
                                    pk_part_3 = pk_part_3 + pk_o_17;
                                    float pk_o_18 = __shfl_sync(0xFFFFFFFF, pk_part_3, pk_lane ^ 8);
                                    pk_part_3 = pk_part_3 + pk_o_18;
                                    float pk_o_19 = __shfl_sync(0xFFFFFFFF, pk_part_3, pk_lane ^ 16);
                                    pk_part_3 = pk_part_3 + pk_o_19;
                                    if (w == 0 && pk_x0 - 1 - 3 >= 0) {
                                        pk_acc = pk_acc + pk_part_3;
                                        int pk_hit_3 = __any_sync(0xFFFFFFFF, pk_acc <= -14.43f);
                                        if (pk_hit_3 != 0) {
                                            w = pk_b * 8 + 3 + 1;
                                        }
                                    }
                                    float pk_part_4 = 0.0f;
                                    float pk_lg_8;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_8) : "f"(pk_g[8] + 1e-10f));
                                    pk_part_4 = pk_part_4 + ((pk_cnt[8] != 0) ? pk_lg_8 : 0.0f);
                                    float pk_lg_9;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_9) : "f"(pk_g[9] + 1e-10f));
                                    pk_part_4 = pk_part_4 + ((pk_cnt[9] != 0) ? pk_lg_9 : 0.0f);
                                    float pk_o_20 = __shfl_sync(0xFFFFFFFF, pk_part_4, pk_lane ^ 1);
                                    pk_part_4 = pk_part_4 + pk_o_20;
                                    float pk_o_21 = __shfl_sync(0xFFFFFFFF, pk_part_4, pk_lane ^ 2);
                                    pk_part_4 = pk_part_4 + pk_o_21;
                                    float pk_o_22 = __shfl_sync(0xFFFFFFFF, pk_part_4, pk_lane ^ 4);
                                    pk_part_4 = pk_part_4 + pk_o_22;
                                    float pk_o_23 = __shfl_sync(0xFFFFFFFF, pk_part_4, pk_lane ^ 8);
                                    pk_part_4 = pk_part_4 + pk_o_23;
                                    float pk_o_24 = __shfl_sync(0xFFFFFFFF, pk_part_4, pk_lane ^ 16);
                                    pk_part_4 = pk_part_4 + pk_o_24;
                                    if (w == 0 && pk_x0 - 1 - 4 >= 0) {
                                        pk_acc = pk_acc + pk_part_4;
                                        int pk_hit_4 = __any_sync(0xFFFFFFFF, pk_acc <= -14.43f);
                                        if (pk_hit_4 != 0) {
                                            w = pk_b * 8 + 4 + 1;
                                        }
                                    }
                                    float pk_part_5 = 0.0f;
                                    float pk_lg_10;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_10) : "f"(pk_g[10] + 1e-10f));
                                    pk_part_5 = pk_part_5 + ((pk_cnt[10] != 0) ? pk_lg_10 : 0.0f);
                                    float pk_lg_11;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_11) : "f"(pk_g[11] + 1e-10f));
                                    pk_part_5 = pk_part_5 + ((pk_cnt[11] != 0) ? pk_lg_11 : 0.0f);
                                    float pk_o_25 = __shfl_sync(0xFFFFFFFF, pk_part_5, pk_lane ^ 1);
                                    pk_part_5 = pk_part_5 + pk_o_25;
                                    float pk_o_26 = __shfl_sync(0xFFFFFFFF, pk_part_5, pk_lane ^ 2);
                                    pk_part_5 = pk_part_5 + pk_o_26;
                                    float pk_o_27 = __shfl_sync(0xFFFFFFFF, pk_part_5, pk_lane ^ 4);
                                    pk_part_5 = pk_part_5 + pk_o_27;
                                    float pk_o_28 = __shfl_sync(0xFFFFFFFF, pk_part_5, pk_lane ^ 8);
                                    pk_part_5 = pk_part_5 + pk_o_28;
                                    float pk_o_29 = __shfl_sync(0xFFFFFFFF, pk_part_5, pk_lane ^ 16);
                                    pk_part_5 = pk_part_5 + pk_o_29;
                                    if (w == 0 && pk_x0 - 1 - 5 >= 0) {
                                        pk_acc = pk_acc + pk_part_5;
                                        int pk_hit_5 = __any_sync(0xFFFFFFFF, pk_acc <= -14.43f);
                                        if (pk_hit_5 != 0) {
                                            w = pk_b * 8 + 5 + 1;
                                        }
                                    }
                                    float pk_part_6 = 0.0f;
                                    float pk_lg_12;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_12) : "f"(pk_g[12] + 1e-10f));
                                    pk_part_6 = pk_part_6 + ((pk_cnt[12] != 0) ? pk_lg_12 : 0.0f);
                                    float pk_lg_13;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_13) : "f"(pk_g[13] + 1e-10f));
                                    pk_part_6 = pk_part_6 + ((pk_cnt[13] != 0) ? pk_lg_13 : 0.0f);
                                    float pk_o_30 = __shfl_sync(0xFFFFFFFF, pk_part_6, pk_lane ^ 1);
                                    pk_part_6 = pk_part_6 + pk_o_30;
                                    float pk_o_31 = __shfl_sync(0xFFFFFFFF, pk_part_6, pk_lane ^ 2);
                                    pk_part_6 = pk_part_6 + pk_o_31;
                                    float pk_o_32 = __shfl_sync(0xFFFFFFFF, pk_part_6, pk_lane ^ 4);
                                    pk_part_6 = pk_part_6 + pk_o_32;
                                    float pk_o_33 = __shfl_sync(0xFFFFFFFF, pk_part_6, pk_lane ^ 8);
                                    pk_part_6 = pk_part_6 + pk_o_33;
                                    float pk_o_34 = __shfl_sync(0xFFFFFFFF, pk_part_6, pk_lane ^ 16);
                                    pk_part_6 = pk_part_6 + pk_o_34;
                                    if (w == 0 && pk_x0 - 1 - 6 >= 0) {
                                        pk_acc = pk_acc + pk_part_6;
                                        int pk_hit_6 = __any_sync(0xFFFFFFFF, pk_acc <= -14.43f);
                                        if (pk_hit_6 != 0) {
                                            w = pk_b * 8 + 6 + 1;
                                        }
                                    }
                                    float pk_part_7 = 0.0f;
                                    float pk_lg_14;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_14) : "f"(pk_g[14] + 1e-10f));
                                    pk_part_7 = pk_part_7 + ((pk_cnt[14] != 0) ? pk_lg_14 : 0.0f);
                                    float pk_lg_15;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(pk_lg_15) : "f"(pk_g[15] + 1e-10f));
                                    pk_part_7 = pk_part_7 + ((pk_cnt[15] != 0) ? pk_lg_15 : 0.0f);
                                    float pk_o_35 = __shfl_sync(0xFFFFFFFF, pk_part_7, pk_lane ^ 1);
                                    pk_part_7 = pk_part_7 + pk_o_35;
                                    float pk_o_36 = __shfl_sync(0xFFFFFFFF, pk_part_7, pk_lane ^ 2);
                                    pk_part_7 = pk_part_7 + pk_o_36;
                                    float pk_o_37 = __shfl_sync(0xFFFFFFFF, pk_part_7, pk_lane ^ 4);
                                    pk_part_7 = pk_part_7 + pk_o_37;
                                    float pk_o_38 = __shfl_sync(0xFFFFFFFF, pk_part_7, pk_lane ^ 8);
                                    pk_part_7 = pk_part_7 + pk_o_38;
                                    float pk_o_39 = __shfl_sync(0xFFFFFFFF, pk_part_7, pk_lane ^ 16);
                                    pk_part_7 = pk_part_7 + pk_o_39;
                                    if (w == 0 && pk_x0 - 1 - 7 >= 0) {
                                        pk_acc = pk_acc + pk_part_7;
                                        int pk_hit_7 = __any_sync(0xFFFFFFFF, pk_acc <= -14.43f);
                                        if (pk_hit_7 != 0) {
                                            w = pk_b * 8 + 7 + 1;
                                        }
                                    }
                                    pk_b = pk_b + 1;
                                    pk_go = ((w == 0 && pk_b < 16 && pk_x0 - 8 > 0) ? 1 : 0);
                                }
                                if (pj_fw == 0) {
                                    pj_warm = w;
                                    pj_sat = ((w != 0) ? 1 : 0);
                                    pj_go = pj_sat;
                                }
                                if (pj_fw > 0 && w != 0) {
                                    pj_wend = pj_x;
                                    pj_go = 0;
                                }
                                pj_x = pj_x + pj_sp;
                                pj_fw = pj_fw + 1;
                            }
                        }
                        int pj_lo = pj_cut - pj_warm;
                        if ((pj_wend - pj_lo & 1) != 0) {
                            pj_lo = pj_lo - 1;
                        }
                        unsigned int pj_elo = (unsigned int)((pj_sat != 0) ? pj_lo : pj_cut);
                        unsigned int pj_en = (unsigned int)((pj_sat != 0) ? pj_wend - pj_lo : 0);
                        if (lane == 0) {
                            *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_ws) + (pl_echo0 + pj_it * 2)) + (0)) = pj_elo;
                            *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_ws) + (pl_echo0 + pj_it * 2 + 1)) + (0)) = pj_en;
                        }
                    }
                }
            }
            __threadfence_block();
            asm volatile("barrier.sync 3, 384;" ::: "memory");
            unsigned int _phase_gate_full_1 = 0;
            unsigned int _phase_state_full_0 = 0;
            unsigned int _phase_ks_full_0 = 0;
            unsigned int _phase_v_full = 0;
            unsigned int _phase_u_full_0 = 0;
            unsigned int _phase_qs_full_0 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_stage_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int it1 = ((smem_scan[0] == 0) ? (unsigned int)smem_scan[2] : (unsigned int)smem_scan[1]); it1 < (unsigned int)smem_scan[1]; it1++) {
                int w1 = 0;
                int xh_1 = 0;
                int seq_1 = items[(int)it1 * 4];
                int head_1 = items[(int)it1 * 4 + 1];
                int cut_1 = items[(int)it1 * 4 + 2];
                int span_1 = items[(int)it1 * 4 + 3];
                int rs_1 = cu_seqlens[seq_1];
                int re_1 = cu_seqlens[seq_1 + 1];
                int start_1 = ((rs_1 < 0) ? 0 : ((rs_1 > total_tokens) ? total_tokens : rs_1));
                int end_1 = ((re_1 < start_1) ? start_1 : ((re_1 > total_tokens) ? total_tokens : re_1));
                int tok0_1 = start_1 + cut_1 * 64;
                int tokend_1 = start_1 + (cut_1 + span_1) * 64;
                tokend_1 = ((end_1 < tokend_1) ? end_1 : tokend_1);
                int sp_at_1 = num_seqs * num_heads + (int)it1 * 2;
                int sp_lo_1 = (int)reinterpret_cast<unsigned int*>(split_ws)[sp_at_1];
                int sp_num_1 = (int)reinterpret_cast<unsigned int*>(split_ws)[sp_at_1 + 1];
                int sp_wend_1 = sp_lo_1 + sp_num_1;
                int start_tok_1 = start_1;
                tokend_1 = start_1 + (sp_wend_1 + num_hh - 1) / num_hh * 64;
                tokend_1 = ((end_1 < tokend_1) ? end_1 : tokend_1);
                int cstart_1 = sp_lo_1;
                int abs0_1 = cut_1;
                int nch_1 = ((sp_num_1 == 0) ? 0 : (tokend_1 - start_1 + 63) / 64 * num_hh - sp_lo_1);
                int wstart_1 = cut_1;
                int owns_end_1 = (((sp_num_1 > 0 || cut_1 == 0) && sp_wend_1 == 2 * (((end_1 - start_1) * num_hh + 127) / 128)) ? 1 : 0);
                int rd_s = state_indices[seq_1];
                int rd_ok = ((rd_s >= 0 && (long long)rd_s < initial_state_size_p0) ? 1 : 0);
                long long rd_head = (long long)rd_s * initial_state_stride_slot + (long long)head_1 * 8192ll;
                int seed_init1 = ((abs0_1 == 0 && (flags & 1) != 0 && rd_ok != 0) ? 1 : 0);
                if (nch_1 != 0) {
                    #pragma unroll
                    for (int half = 0; half < 2; half++) {
                        float frag[32];
                        frag[0] = 0.0f;
                        frag[1] = 0.0f;
                        frag[2] = 0.0f;
                        frag[3] = 0.0f;
                        frag[4] = 0.0f;
                        frag[5] = 0.0f;
                        frag[6] = 0.0f;
                        frag[7] = 0.0f;
                        frag[8] = 0.0f;
                        frag[9] = 0.0f;
                        frag[10] = 0.0f;
                        frag[11] = 0.0f;
                        frag[12] = 0.0f;
                        frag[13] = 0.0f;
                        frag[14] = 0.0f;
                        frag[15] = 0.0f;
                        frag[16] = 0.0f;
                        frag[17] = 0.0f;
                        frag[18] = 0.0f;
                        frag[19] = 0.0f;
                        frag[20] = 0.0f;
                        frag[21] = 0.0f;
                        frag[22] = 0.0f;
                        frag[23] = 0.0f;
                        frag[24] = 0.0f;
                        frag[25] = 0.0f;
                        frag[26] = 0.0f;
                        frag[27] = 0.0f;
                        frag[28] = 0.0f;
                        frag[29] = 0.0f;
                        frag[30] = 0.0f;
                        frag[31] = 0.0f;
                        if (seed_init1 != 0) {
                            long long base_top = rd_head + (long long)row_top1 * 128ll;
                            long long base_bot = base_top + 1024ll;
                            #pragma unroll
                            for (int group = 0; group < 8; group++) {
                                int col_pair = half * 64 + group * 8 + lane_quad1 * 2;
                                unsigned int sp_w[2];
                                {
                                    uint32_t _bf16x2_bits_0;
                                    _bf16x2_bits_0 = *reinterpret_cast<const uint32_t*>(initial_state + base_top + (long long)col_pair);
                                    sp_w[0] = _bf16x2_bits_0;
                                }
                                {
                                    uint32_t _bf16x2_bits_1;
                                    _bf16x2_bits_1 = *reinterpret_cast<const uint32_t*>(initial_state + base_bot + (long long)col_pair);
                                    sp_w[1] = _bf16x2_bits_1;
                                }
                                frag[group * 4] = __uint_as_float(sp_w[0] << 16);
                                frag[group * 4 + 1] = __uint_as_float(sp_w[0] & 4294901760u);
                                frag[group * 4 + 2] = __uint_as_float(sp_w[1] << 16);
                                frag[group * 4 + 2 + 1] = __uint_as_float(sp_w[1] & 4294901760u);
                            }
                        }
                        int seed_addr = taddr + (unsigned int)row_base1 + (unsigned int)(half * 64);
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                            :: "r"(seed_addr), "r"(*reinterpret_cast<const uint32_t*>(&frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&frag[3])), "r"(*reinterpret_cast<const uint32_t*>(&frag[4])), "r"(*reinterpret_cast<const uint32_t*>(&frag[5])), "r"(*reinterpret_cast<const uint32_t*>(&frag[6])), "r"(*reinterpret_cast<const uint32_t*>(&frag[7])), "r"(*reinterpret_cast<const uint32_t*>(&frag[8])), "r"(*reinterpret_cast<const uint32_t*>(&frag[9])), "r"(*reinterpret_cast<const uint32_t*>(&frag[10])), "r"(*reinterpret_cast<const uint32_t*>(&frag[11])), "r"(*reinterpret_cast<const uint32_t*>(&frag[12])), "r"(*reinterpret_cast<const uint32_t*>(&frag[13])), "r"(*reinterpret_cast<const uint32_t*>(&frag[14])), "r"(*reinterpret_cast<const uint32_t*>(&frag[15])), "r"(*reinterpret_cast<const uint32_t*>(&frag[16])), "r"(*reinterpret_cast<const uint32_t*>(&frag[17])), "r"(*reinterpret_cast<const uint32_t*>(&frag[18])), "r"(*reinterpret_cast<const uint32_t*>(&frag[19])), "r"(*reinterpret_cast<const uint32_t*>(&frag[20])), "r"(*reinterpret_cast<const uint32_t*>(&frag[21])), "r"(*reinterpret_cast<const uint32_t*>(&frag[22])), "r"(*reinterpret_cast<const uint32_t*>(&frag[23])), "r"(*reinterpret_cast<const uint32_t*>(&frag[24])), "r"(*reinterpret_cast<const uint32_t*>(&frag[25])), "r"(*reinterpret_cast<const uint32_t*>(&frag[26])), "r"(*reinterpret_cast<const uint32_t*>(&frag[27])), "r"(*reinterpret_cast<const uint32_t*>(&frag[28])), "r"(*reinterpret_cast<const uint32_t*>(&frag[29])), "r"(*reinterpret_cast<const uint32_t*>(&frag[30])), "r"(*reinterpret_cast<const uint32_t*>(&frag[31])));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (abs0_1 == 0 && owns_end_1 == 0 && (flags & 3) == 3) {
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (warp1 == 0 && lane == 0) {
                            __threadfence();
                            asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_ws)) + (seq_1 * num_heads + head_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                    }
                }
                int blk1c = cstart_1 / num_hh;
                int cb1c = cstart_1 - blk1c * num_hh;
                #pragma unroll 1
                for (int c1 = 0; c1 < nch_1; c1++) {
                    int blk1 = blk1c;
                    int cb1 = cb1c;
                    int is_head1 = ((c1 == 0 || cb1 == 0) ? 1 : 0);
                    int last_in_blk1 = ((cb1 == num_hh - 1 || c1 == nch_1 - 1) ? 1 : 0);
                    int write1 = ((blk1 * num_hh >= wstart_1) ? 1 : 0);
                    int tok0_blk1 = start_tok_1 + blk1 * 64;
                    int vt1 = tokend_1 - tok0_blk1;
                    vt1 = ((vt1 > 64) ? 64 : ((vt1 < 0) ? 0 : vt1));
                    int vr1 = vt1 * num_hh - cb1 * 64;
                    vr1 = ((vr1 > 64) ? 64 : ((vr1 < 0) ? 0 : vr1));
                    mbarrier_wait(gate_full_addr + (g_cur1) * 8, _phase_gate_full_1);
                    int gbase1 = g_cur1 * 64;
                    float cs63 = smem_cs[gbase1 + 63];
                    if (c1 > 0) {
                        mbarrier_wait(state_full_addr, _phase_state_full_0);
                        _phase_state_full_0 ^= 1;
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int st_addr0 = taddr + (unsigned int)row_base1;
                    int st_addr1 = st_addr0 + 64;
                    float s0[32];
                    float s1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&s0[0])), "=r"(*reinterpret_cast<uint32_t*>(&s0[1])), "=r"(*reinterpret_cast<uint32_t*>(&s0[2])), "=r"(*reinterpret_cast<uint32_t*>(&s0[3])), "=r"(*reinterpret_cast<uint32_t*>(&s0[4])), "=r"(*reinterpret_cast<uint32_t*>(&s0[5])), "=r"(*reinterpret_cast<uint32_t*>(&s0[6])), "=r"(*reinterpret_cast<uint32_t*>(&s0[7])), "=r"(*reinterpret_cast<uint32_t*>(&s0[8])), "=r"(*reinterpret_cast<uint32_t*>(&s0[9])), "=r"(*reinterpret_cast<uint32_t*>(&s0[10])), "=r"(*reinterpret_cast<uint32_t*>(&s0[11])), "=r"(*reinterpret_cast<uint32_t*>(&s0[12])), "=r"(*reinterpret_cast<uint32_t*>(&s0[13])), "=r"(*reinterpret_cast<uint32_t*>(&s0[14])), "=r"(*reinterpret_cast<uint32_t*>(&s0[15])), "=r"(*reinterpret_cast<uint32_t*>(&s0[16])), "=r"(*reinterpret_cast<uint32_t*>(&s0[17])), "=r"(*reinterpret_cast<uint32_t*>(&s0[18])), "=r"(*reinterpret_cast<uint32_t*>(&s0[19])), "=r"(*reinterpret_cast<uint32_t*>(&s0[20])), "=r"(*reinterpret_cast<uint32_t*>(&s0[21])), "=r"(*reinterpret_cast<uint32_t*>(&s0[22])), "=r"(*reinterpret_cast<uint32_t*>(&s0[23])), "=r"(*reinterpret_cast<uint32_t*>(&s0[24])), "=r"(*reinterpret_cast<uint32_t*>(&s0[25])), "=r"(*reinterpret_cast<uint32_t*>(&s0[26])), "=r"(*reinterpret_cast<uint32_t*>(&s0[27])), "=r"(*reinterpret_cast<uint32_t*>(&s0[28])), "=r"(*reinterpret_cast<uint32_t*>(&s0[29])), "=r"(*reinterpret_cast<uint32_t*>(&s0[30])), "=r"(*reinterpret_cast<uint32_t*>(&s0[31]))
                        : "r"(st_addr0));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&s1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s1[1])), "=r"(*reinterpret_cast<uint32_t*>(&s1[2])), "=r"(*reinterpret_cast<uint32_t*>(&s1[3])), "=r"(*reinterpret_cast<uint32_t*>(&s1[4])), "=r"(*reinterpret_cast<uint32_t*>(&s1[5])), "=r"(*reinterpret_cast<uint32_t*>(&s1[6])), "=r"(*reinterpret_cast<uint32_t*>(&s1[7])), "=r"(*reinterpret_cast<uint32_t*>(&s1[8])), "=r"(*reinterpret_cast<uint32_t*>(&s1[9])), "=r"(*reinterpret_cast<uint32_t*>(&s1[10])), "=r"(*reinterpret_cast<uint32_t*>(&s1[11])), "=r"(*reinterpret_cast<uint32_t*>(&s1[12])), "=r"(*reinterpret_cast<uint32_t*>(&s1[13])), "=r"(*reinterpret_cast<uint32_t*>(&s1[14])), "=r"(*reinterpret_cast<uint32_t*>(&s1[15])), "=r"(*reinterpret_cast<uint32_t*>(&s1[16])), "=r"(*reinterpret_cast<uint32_t*>(&s1[17])), "=r"(*reinterpret_cast<uint32_t*>(&s1[18])), "=r"(*reinterpret_cast<uint32_t*>(&s1[19])), "=r"(*reinterpret_cast<uint32_t*>(&s1[20])), "=r"(*reinterpret_cast<uint32_t*>(&s1[21])), "=r"(*reinterpret_cast<uint32_t*>(&s1[22])), "=r"(*reinterpret_cast<uint32_t*>(&s1[23])), "=r"(*reinterpret_cast<uint32_t*>(&s1[24])), "=r"(*reinterpret_cast<uint32_t*>(&s1[25])), "=r"(*reinterpret_cast<uint32_t*>(&s1[26])), "=r"(*reinterpret_cast<uint32_t*>(&s1[27])), "=r"(*reinterpret_cast<uint32_t*>(&s1[28])), "=r"(*reinterpret_cast<uint32_t*>(&s1[29])), "=r"(*reinterpret_cast<uint32_t*>(&s1[30])), "=r"(*reinterpret_cast<uint32_t*>(&s1[31]))
                        : "r"(st_addr1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int s0b[16];
                    unsigned int s1b[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s0[_lp*2 + 0], s0[_lp*2+1 + 0]));
                        s0b[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s1[_lp*2 + 0], s1[_lp*2+1 + 0]));
                        s1b[_lp] = *(uint32_t*)&_bf2;
                    }
                    int img_addr0 = taddr + 192 + (unsigned int)row_base1;
                    int img_addr1 = img_addr0 + 32;
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(img_addr0), "r"(*reinterpret_cast<const uint32_t*>(&s0b[0])), "r"(*reinterpret_cast<const uint32_t*>(&s0b[1])), "r"(*reinterpret_cast<const uint32_t*>(&s0b[2])), "r"(*reinterpret_cast<const uint32_t*>(&s0b[3])), "r"(*reinterpret_cast<const uint32_t*>(&s0b[4])), "r"(*reinterpret_cast<const uint32_t*>(&s0b[5])), "r"(*reinterpret_cast<const uint32_t*>(&s0b[6])), "r"(*reinterpret_cast<const uint32_t*>(&s0b[7])), "r"(*reinterpret_cast<const uint32_t*>(&s0b[8])), "r"(*reinterpret_cast<const uint32_t*>(&s0b[9])), "r"(*reinterpret_cast<const uint32_t*>(&s0b[10])), "r"(*reinterpret_cast<const uint32_t*>(&s0b[11])), "r"(*reinterpret_cast<const uint32_t*>(&s0b[12])), "r"(*reinterpret_cast<const uint32_t*>(&s0b[13])), "r"(*reinterpret_cast<const uint32_t*>(&s0b[14])), "r"(*reinterpret_cast<const uint32_t*>(&s0b[15])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(img_addr1), "r"(*reinterpret_cast<const uint32_t*>(&s1b[0])), "r"(*reinterpret_cast<const uint32_t*>(&s1b[1])), "r"(*reinterpret_cast<const uint32_t*>(&s1b[2])), "r"(*reinterpret_cast<const uint32_t*>(&s1b[3])), "r"(*reinterpret_cast<const uint32_t*>(&s1b[4])), "r"(*reinterpret_cast<const uint32_t*>(&s1b[5])), "r"(*reinterpret_cast<const uint32_t*>(&s1b[6])), "r"(*reinterpret_cast<const uint32_t*>(&s1b[7])), "r"(*reinterpret_cast<const uint32_t*>(&s1b[8])), "r"(*reinterpret_cast<const uint32_t*>(&s1b[9])), "r"(*reinterpret_cast<const uint32_t*>(&s1b[10])), "r"(*reinterpret_cast<const uint32_t*>(&s1b[11])), "r"(*reinterpret_cast<const uint32_t*>(&s1b[12])), "r"(*reinterpret_cast<const uint32_t*>(&s1b[13])), "r"(*reinterpret_cast<const uint32_t*>(&s1b[14])), "r"(*reinterpret_cast<const uint32_t*>(&s1b[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(img_ready_addr);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&s0[0])), "=r"(*reinterpret_cast<uint32_t*>(&s0[1])), "=r"(*reinterpret_cast<uint32_t*>(&s0[2])), "=r"(*reinterpret_cast<uint32_t*>(&s0[3])), "=r"(*reinterpret_cast<uint32_t*>(&s0[4])), "=r"(*reinterpret_cast<uint32_t*>(&s0[5])), "=r"(*reinterpret_cast<uint32_t*>(&s0[6])), "=r"(*reinterpret_cast<uint32_t*>(&s0[7])), "=r"(*reinterpret_cast<uint32_t*>(&s0[8])), "=r"(*reinterpret_cast<uint32_t*>(&s0[9])), "=r"(*reinterpret_cast<uint32_t*>(&s0[10])), "=r"(*reinterpret_cast<uint32_t*>(&s0[11])), "=r"(*reinterpret_cast<uint32_t*>(&s0[12])), "=r"(*reinterpret_cast<uint32_t*>(&s0[13])), "=r"(*reinterpret_cast<uint32_t*>(&s0[14])), "=r"(*reinterpret_cast<uint32_t*>(&s0[15])), "=r"(*reinterpret_cast<uint32_t*>(&s0[16])), "=r"(*reinterpret_cast<uint32_t*>(&s0[17])), "=r"(*reinterpret_cast<uint32_t*>(&s0[18])), "=r"(*reinterpret_cast<uint32_t*>(&s0[19])), "=r"(*reinterpret_cast<uint32_t*>(&s0[20])), "=r"(*reinterpret_cast<uint32_t*>(&s0[21])), "=r"(*reinterpret_cast<uint32_t*>(&s0[22])), "=r"(*reinterpret_cast<uint32_t*>(&s0[23])), "=r"(*reinterpret_cast<uint32_t*>(&s0[24])), "=r"(*reinterpret_cast<uint32_t*>(&s0[25])), "=r"(*reinterpret_cast<uint32_t*>(&s0[26])), "=r"(*reinterpret_cast<uint32_t*>(&s0[27])), "=r"(*reinterpret_cast<uint32_t*>(&s0[28])), "=r"(*reinterpret_cast<uint32_t*>(&s0[29])), "=r"(*reinterpret_cast<uint32_t*>(&s0[30])), "=r"(*reinterpret_cast<uint32_t*>(&s0[31]))
                        : "r"(st_addr0));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&s1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s1[1])), "=r"(*reinterpret_cast<uint32_t*>(&s1[2])), "=r"(*reinterpret_cast<uint32_t*>(&s1[3])), "=r"(*reinterpret_cast<uint32_t*>(&s1[4])), "=r"(*reinterpret_cast<uint32_t*>(&s1[5])), "=r"(*reinterpret_cast<uint32_t*>(&s1[6])), "=r"(*reinterpret_cast<uint32_t*>(&s1[7])), "=r"(*reinterpret_cast<uint32_t*>(&s1[8])), "=r"(*reinterpret_cast<uint32_t*>(&s1[9])), "=r"(*reinterpret_cast<uint32_t*>(&s1[10])), "=r"(*reinterpret_cast<uint32_t*>(&s1[11])), "=r"(*reinterpret_cast<uint32_t*>(&s1[12])), "=r"(*reinterpret_cast<uint32_t*>(&s1[13])), "=r"(*reinterpret_cast<uint32_t*>(&s1[14])), "=r"(*reinterpret_cast<uint32_t*>(&s1[15])), "=r"(*reinterpret_cast<uint32_t*>(&s1[16])), "=r"(*reinterpret_cast<uint32_t*>(&s1[17])), "=r"(*reinterpret_cast<uint32_t*>(&s1[18])), "=r"(*reinterpret_cast<uint32_t*>(&s1[19])), "=r"(*reinterpret_cast<uint32_t*>(&s1[20])), "=r"(*reinterpret_cast<uint32_t*>(&s1[21])), "=r"(*reinterpret_cast<uint32_t*>(&s1[22])), "=r"(*reinterpret_cast<uint32_t*>(&s1[23])), "=r"(*reinterpret_cast<uint32_t*>(&s1[24])), "=r"(*reinterpret_cast<uint32_t*>(&s1[25])), "=r"(*reinterpret_cast<uint32_t*>(&s1[26])), "=r"(*reinterpret_cast<uint32_t*>(&s1[27])), "=r"(*reinterpret_cast<uint32_t*>(&s1[28])), "=r"(*reinterpret_cast<uint32_t*>(&s1[29])), "=r"(*reinterpret_cast<uint32_t*>(&s1[30])), "=r"(*reinterpret_cast<uint32_t*>(&s1[31]))
                        : "r"(st_addr1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float gamma1 = approx_exp2(cs63);
                    const float2 _scale2_2 = {gamma1, gamma1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(s0)[_ls], _scale2_2);
                    const float2 _scale2_3 = {gamma1, gamma1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(s1)[_ls], _scale2_3);
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x256b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(st_addr0), "r"(*reinterpret_cast<const uint32_t*>(&s0[0])), "r"(*reinterpret_cast<const uint32_t*>(&s0[1])), "r"(*reinterpret_cast<const uint32_t*>(&s0[2])), "r"(*reinterpret_cast<const uint32_t*>(&s0[3])), "r"(*reinterpret_cast<const uint32_t*>(&s0[4])), "r"(*reinterpret_cast<const uint32_t*>(&s0[5])), "r"(*reinterpret_cast<const uint32_t*>(&s0[6])), "r"(*reinterpret_cast<const uint32_t*>(&s0[7])), "r"(*reinterpret_cast<const uint32_t*>(&s0[8])), "r"(*reinterpret_cast<const uint32_t*>(&s0[9])), "r"(*reinterpret_cast<const uint32_t*>(&s0[10])), "r"(*reinterpret_cast<const uint32_t*>(&s0[11])), "r"(*reinterpret_cast<const uint32_t*>(&s0[12])), "r"(*reinterpret_cast<const uint32_t*>(&s0[13])), "r"(*reinterpret_cast<const uint32_t*>(&s0[14])), "r"(*reinterpret_cast<const uint32_t*>(&s0[15])), "r"(*reinterpret_cast<const uint32_t*>(&s0[16])), "r"(*reinterpret_cast<const uint32_t*>(&s0[17])), "r"(*reinterpret_cast<const uint32_t*>(&s0[18])), "r"(*reinterpret_cast<const uint32_t*>(&s0[19])), "r"(*reinterpret_cast<const uint32_t*>(&s0[20])), "r"(*reinterpret_cast<const uint32_t*>(&s0[21])), "r"(*reinterpret_cast<const uint32_t*>(&s0[22])), "r"(*reinterpret_cast<const uint32_t*>(&s0[23])), "r"(*reinterpret_cast<const uint32_t*>(&s0[24])), "r"(*reinterpret_cast<const uint32_t*>(&s0[25])), "r"(*reinterpret_cast<const uint32_t*>(&s0[26])), "r"(*reinterpret_cast<const uint32_t*>(&s0[27])), "r"(*reinterpret_cast<const uint32_t*>(&s0[28])), "r"(*reinterpret_cast<const uint32_t*>(&s0[29])), "r"(*reinterpret_cast<const uint32_t*>(&s0[30])), "r"(*reinterpret_cast<const uint32_t*>(&s0[31])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x256b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(st_addr1), "r"(*reinterpret_cast<const uint32_t*>(&s1[0])), "r"(*reinterpret_cast<const uint32_t*>(&s1[1])), "r"(*reinterpret_cast<const uint32_t*>(&s1[2])), "r"(*reinterpret_cast<const uint32_t*>(&s1[3])), "r"(*reinterpret_cast<const uint32_t*>(&s1[4])), "r"(*reinterpret_cast<const uint32_t*>(&s1[5])), "r"(*reinterpret_cast<const uint32_t*>(&s1[6])), "r"(*reinterpret_cast<const uint32_t*>(&s1[7])), "r"(*reinterpret_cast<const uint32_t*>(&s1[8])), "r"(*reinterpret_cast<const uint32_t*>(&s1[9])), "r"(*reinterpret_cast<const uint32_t*>(&s1[10])), "r"(*reinterpret_cast<const uint32_t*>(&s1[11])), "r"(*reinterpret_cast<const uint32_t*>(&s1[12])), "r"(*reinterpret_cast<const uint32_t*>(&s1[13])), "r"(*reinterpret_cast<const uint32_t*>(&s1[14])), "r"(*reinterpret_cast<const uint32_t*>(&s1[15])), "r"(*reinterpret_cast<const uint32_t*>(&s1[16])), "r"(*reinterpret_cast<const uint32_t*>(&s1[17])), "r"(*reinterpret_cast<const uint32_t*>(&s1[18])), "r"(*reinterpret_cast<const uint32_t*>(&s1[19])), "r"(*reinterpret_cast<const uint32_t*>(&s1[20])), "r"(*reinterpret_cast<const uint32_t*>(&s1[21])), "r"(*reinterpret_cast<const uint32_t*>(&s1[22])), "r"(*reinterpret_cast<const uint32_t*>(&s1[23])), "r"(*reinterpret_cast<const uint32_t*>(&s1[24])), "r"(*reinterpret_cast<const uint32_t*>(&s1[25])), "r"(*reinterpret_cast<const uint32_t*>(&s1[26])), "r"(*reinterpret_cast<const uint32_t*>(&s1[27])), "r"(*reinterpret_cast<const uint32_t*>(&s1[28])), "r"(*reinterpret_cast<const uint32_t*>(&s1[29])), "r"(*reinterpret_cast<const uint32_t*>(&s1[30])), "r"(*reinterpret_cast<const uint32_t*>(&s1[31])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_wait(ks_full_addr, _phase_ks_full_0);
                    _phase_ks_full_0 ^= 1;
                    mbarrier_wait(v_full_addr + (v_cur1) * 8, _phase_v_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int ks_addr = taddr + 384 + (unsigned int)row_base1;
                    int y_addr = taddr + 448 + (unsigned int)row_base1;
                    unsigned int yb[16];
                    float yv[32];
                    float ks[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ks[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks[3])), "=r"(*reinterpret_cast<uint32_t*>(&ks[4])), "=r"(*reinterpret_cast<uint32_t*>(&ks[5])), "=r"(*reinterpret_cast<uint32_t*>(&ks[6])), "=r"(*reinterpret_cast<uint32_t*>(&ks[7])), "=r"(*reinterpret_cast<uint32_t*>(&ks[8])), "=r"(*reinterpret_cast<uint32_t*>(&ks[9])), "=r"(*reinterpret_cast<uint32_t*>(&ks[10])), "=r"(*reinterpret_cast<uint32_t*>(&ks[11])), "=r"(*reinterpret_cast<uint32_t*>(&ks[12])), "=r"(*reinterpret_cast<uint32_t*>(&ks[13])), "=r"(*reinterpret_cast<uint32_t*>(&ks[14])), "=r"(*reinterpret_cast<uint32_t*>(&ks[15])), "=r"(*reinterpret_cast<uint32_t*>(&ks[16])), "=r"(*reinterpret_cast<uint32_t*>(&ks[17])), "=r"(*reinterpret_cast<uint32_t*>(&ks[18])), "=r"(*reinterpret_cast<uint32_t*>(&ks[19])), "=r"(*reinterpret_cast<uint32_t*>(&ks[20])), "=r"(*reinterpret_cast<uint32_t*>(&ks[21])), "=r"(*reinterpret_cast<uint32_t*>(&ks[22])), "=r"(*reinterpret_cast<uint32_t*>(&ks[23])), "=r"(*reinterpret_cast<uint32_t*>(&ks[24])), "=r"(*reinterpret_cast<uint32_t*>(&ks[25])), "=r"(*reinterpret_cast<uint32_t*>(&ks[26])), "=r"(*reinterpret_cast<uint32_t*>(&ks[27])), "=r"(*reinterpret_cast<uint32_t*>(&ks[28])), "=r"(*reinterpret_cast<uint32_t*>(&ks[29])), "=r"(*reinterpret_cast<uint32_t*>(&ks[30])), "=r"(*reinterpret_cast<uint32_t*>(&ks[31]))
                        : "r"(ks_addr));
                    unsigned int vw[16];
                    unsigned int v_stage_addr1 = smem_v_addr + (unsigned int)(v_cur1 * 8192);
                    #pragma unroll
                    for (int q4 = 0; q4 < 4; q4++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(vw[q4 * 4]), "=r"(vw[q4 * 4 + 1]), "=r"(vw[q4 * 4 + 2]), "=r"(vw[q4 * 4 + 3])
                            : "r"(v_stage_addr1 + (unsigned int)(((q4 * 2 + mtx1 / 2) * 8 + row_in_mtx1) * 128 + (((warp1 * 16 + (mtx1 & 1) * 8) * 2 >> 4 ^ (q4 * 2 + mtx1 / 2) * 8 + row_in_mtx1 & 7) << 4) + ((warp1 * 16 + (mtx1 & 1) * 8) * 2 & 15)))
                            : "memory");
                    }
                    unsigned int ew[16];
                    #pragma unroll
                    for (int k8 = 0; k8 < 8; k8++) {
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ew[k8 * 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(k8 * 2) + 1]))
                            : "r"(smem_ecs_addr + (unsigned int)((gbase1 + k8 * 8 + lane_quad1 * 2) * 4)));
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int p2 = 0; p2 < 16; p2++) {
                        float2 _f32x2_mul_a_4 = make_float2(ks[p2 * 2], ks[p2 * 2 + 1]);
                        float2 _f32x2_mul_b_5 = make_float2(__uint_as_float(ew[p2 / 2 * 2]), __uint_as_float(ew[p2 / 2 * 2 + 1]));
                        float2 kse;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kse) : "l"(*(const unsigned long long*)&_f32x2_mul_a_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_5));
                        ks[p2 * 2] = kse.x;
                        ks[p2 * 2 + 1] = kse.y;
                    }
                    unsigned int ksb[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks[_lp*2 + 0], ks[_lp*2+1 + 0]));
                        ksb[_lp] = *(uint32_t*)&_bf2;
                    }
                    float ksr[32];
                    #pragma unroll
                    for (int _pair = 0; _pair < 16; _pair++) {
                        (&ksr[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ksb[_pair]) << 16);
                        (&ksr[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ksb[_pair]) & 0xffff0000u);
                    }
                    float vf[32];
                    #pragma unroll
                    for (int _pair = 0; _pair < 16; _pair++) {
                        (&vf[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(vw[_pair]) << 16);
                        (&vf[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(vw[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int p3 = 0; p3 < 16; p3++) {
                        float2 yv2 = sub_f32x2(make_float2(vf[p3 * 2], vf[p3 * 2 + 1]), make_float2(ksr[p3 * 2], ksr[p3 * 2 + 1]));
                        yv[p3 * 2] = yv2.x;
                        yv[p3 * 2 + 1] = yv2.y;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv[_lp*2 + 0], yv[_lp*2+1 + 0]));
                        yb[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(y_addr), "r"(*reinterpret_cast<const uint32_t*>(&yb[0])), "r"(*reinterpret_cast<const uint32_t*>(&yb[1])), "r"(*reinterpret_cast<const uint32_t*>(&yb[2])), "r"(*reinterpret_cast<const uint32_t*>(&yb[3])), "r"(*reinterpret_cast<const uint32_t*>(&yb[4])), "r"(*reinterpret_cast<const uint32_t*>(&yb[5])), "r"(*reinterpret_cast<const uint32_t*>(&yb[6])), "r"(*reinterpret_cast<const uint32_t*>(&yb[7])), "r"(*reinterpret_cast<const uint32_t*>(&yb[8])), "r"(*reinterpret_cast<const uint32_t*>(&yb[9])), "r"(*reinterpret_cast<const uint32_t*>(&yb[10])), "r"(*reinterpret_cast<const uint32_t*>(&yb[11])), "r"(*reinterpret_cast<const uint32_t*>(&yb[12])), "r"(*reinterpret_cast<const uint32_t*>(&yb[13])), "r"(*reinterpret_cast<const uint32_t*>(&yb[14])), "r"(*reinterpret_cast<const uint32_t*>(&yb[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(y_ready_addr);
                    mbarrier_arrive(v_empty_addr + (v_cur1) * 8);
                    v_cur1 += 1;
                    if (v_cur1 == 2) { v_cur1 = 0; _phase_v_full ^= 1; }
                    unsigned int bw3[16];
                    unsigned int dw3[16];
                    mbarrier_wait(u_full_addr, _phase_u_full_0);
                    _phase_u_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int k83w = 0; k83w < 8; k83w++) {
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&dw3[k83w * 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw3[(k83w * 2) + 1]))
                            : "r"(smem_edec_addr + (unsigned int)((gbase1 + k83w * 8 + lane_quad1 * 2) * 4)));
                    }
                    float uu[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&uu[0])), "=r"(*reinterpret_cast<uint32_t*>(&uu[1])), "=r"(*reinterpret_cast<uint32_t*>(&uu[2])), "=r"(*reinterpret_cast<uint32_t*>(&uu[3])), "=r"(*reinterpret_cast<uint32_t*>(&uu[4])), "=r"(*reinterpret_cast<uint32_t*>(&uu[5])), "=r"(*reinterpret_cast<uint32_t*>(&uu[6])), "=r"(*reinterpret_cast<uint32_t*>(&uu[7])), "=r"(*reinterpret_cast<uint32_t*>(&uu[8])), "=r"(*reinterpret_cast<uint32_t*>(&uu[9])), "=r"(*reinterpret_cast<uint32_t*>(&uu[10])), "=r"(*reinterpret_cast<uint32_t*>(&uu[11])), "=r"(*reinterpret_cast<uint32_t*>(&uu[12])), "=r"(*reinterpret_cast<uint32_t*>(&uu[13])), "=r"(*reinterpret_cast<uint32_t*>(&uu[14])), "=r"(*reinterpret_cast<uint32_t*>(&uu[15])), "=r"(*reinterpret_cast<uint32_t*>(&uu[16])), "=r"(*reinterpret_cast<uint32_t*>(&uu[17])), "=r"(*reinterpret_cast<uint32_t*>(&uu[18])), "=r"(*reinterpret_cast<uint32_t*>(&uu[19])), "=r"(*reinterpret_cast<uint32_t*>(&uu[20])), "=r"(*reinterpret_cast<uint32_t*>(&uu[21])), "=r"(*reinterpret_cast<uint32_t*>(&uu[22])), "=r"(*reinterpret_cast<uint32_t*>(&uu[23])), "=r"(*reinterpret_cast<uint32_t*>(&uu[24])), "=r"(*reinterpret_cast<uint32_t*>(&uu[25])), "=r"(*reinterpret_cast<uint32_t*>(&uu[26])), "=r"(*reinterpret_cast<uint32_t*>(&uu[27])), "=r"(*reinterpret_cast<uint32_t*>(&uu[28])), "=r"(*reinterpret_cast<uint32_t*>(&uu[29])), "=r"(*reinterpret_cast<uint32_t*>(&uu[30])), "=r"(*reinterpret_cast<uint32_t*>(&uu[31]))
                        : "r"(ks_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int ub[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uu[_lp*2 + 0], uu[_lp*2+1 + 0]));
                        ub[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(y_addr), "r"(*reinterpret_cast<const uint32_t*>(&ub[0])), "r"(*reinterpret_cast<const uint32_t*>(&ub[1])), "r"(*reinterpret_cast<const uint32_t*>(&ub[2])), "r"(*reinterpret_cast<const uint32_t*>(&ub[3])), "r"(*reinterpret_cast<const uint32_t*>(&ub[4])), "r"(*reinterpret_cast<const uint32_t*>(&ub[5])), "r"(*reinterpret_cast<const uint32_t*>(&ub[6])), "r"(*reinterpret_cast<const uint32_t*>(&ub[7])), "r"(*reinterpret_cast<const uint32_t*>(&ub[8])), "r"(*reinterpret_cast<const uint32_t*>(&ub[9])), "r"(*reinterpret_cast<const uint32_t*>(&ub[10])), "r"(*reinterpret_cast<const uint32_t*>(&ub[11])), "r"(*reinterpret_cast<const uint32_t*>(&ub[12])), "r"(*reinterpret_cast<const uint32_t*>(&ub[13])), "r"(*reinterpret_cast<const uint32_t*>(&ub[14])), "r"(*reinterpret_cast<const uint32_t*>(&ub[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(uimg_ready_addr);
                    #pragma unroll
                    for (int p5 = 0; p5 < 16; p5++) {
                        float2 _f32x2_mul_a_6 = make_float2(uu[p5 * 2], uu[p5 * 2 + 1]);
                        float2 _f32x2_mul_b_7 = make_float2(__uint_as_float(dw3[p5 / 2 * 2]), __uint_as_float(dw3[p5 / 2 * 2 + 1]));
                        float2 du2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_7));
                        uu[p5 * 2] = du2.x;
                        uu[p5 * 2 + 1] = du2.y;
                    }
                    unsigned int dub[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uu[_lp*2 + 0], uu[_lp*2+1 + 0]));
                        dub[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 480 + (unsigned int)row_base1), "r"(*reinterpret_cast<const uint32_t*>(&dub[0])), "r"(*reinterpret_cast<const uint32_t*>(&dub[1])), "r"(*reinterpret_cast<const uint32_t*>(&dub[2])), "r"(*reinterpret_cast<const uint32_t*>(&dub[3])), "r"(*reinterpret_cast<const uint32_t*>(&dub[4])), "r"(*reinterpret_cast<const uint32_t*>(&dub[5])), "r"(*reinterpret_cast<const uint32_t*>(&dub[6])), "r"(*reinterpret_cast<const uint32_t*>(&dub[7])), "r"(*reinterpret_cast<const uint32_t*>(&dub[8])), "r"(*reinterpret_cast<const uint32_t*>(&dub[9])), "r"(*reinterpret_cast<const uint32_t*>(&dub[10])), "r"(*reinterpret_cast<const uint32_t*>(&dub[11])), "r"(*reinterpret_cast<const uint32_t*>(&dub[12])), "r"(*reinterpret_cast<const uint32_t*>(&dub[13])), "r"(*reinterpret_cast<const uint32_t*>(&dub[14])), "r"(*reinterpret_cast<const uint32_t*>(&dub[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(du_ready_addr);
                    if (is_head1 != 0) {
                        unsigned int ew5[16];
                        #pragma unroll
                        for (int k85 = 0; k85 < 8; k85++) {
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ew5[k85 * 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew5[(k85 * 2) + 1]))
                                : "r"(smem_ect_addr + (unsigned int)((gbase1 + k85 * 8 + lane_quad1 * 2) * 4)));
                        }
                        mbarrier_wait(qs_full_addr, _phase_qs_full_0);
                        _phase_qs_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int o_addr = taddr + 128 + (unsigned int)row_base1;
                        float oq[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&oq[0])), "=r"(*reinterpret_cast<uint32_t*>(&oq[1])), "=r"(*reinterpret_cast<uint32_t*>(&oq[2])), "=r"(*reinterpret_cast<uint32_t*>(&oq[3])), "=r"(*reinterpret_cast<uint32_t*>(&oq[4])), "=r"(*reinterpret_cast<uint32_t*>(&oq[5])), "=r"(*reinterpret_cast<uint32_t*>(&oq[6])), "=r"(*reinterpret_cast<uint32_t*>(&oq[7])), "=r"(*reinterpret_cast<uint32_t*>(&oq[8])), "=r"(*reinterpret_cast<uint32_t*>(&oq[9])), "=r"(*reinterpret_cast<uint32_t*>(&oq[10])), "=r"(*reinterpret_cast<uint32_t*>(&oq[11])), "=r"(*reinterpret_cast<uint32_t*>(&oq[12])), "=r"(*reinterpret_cast<uint32_t*>(&oq[13])), "=r"(*reinterpret_cast<uint32_t*>(&oq[14])), "=r"(*reinterpret_cast<uint32_t*>(&oq[15])), "=r"(*reinterpret_cast<uint32_t*>(&oq[16])), "=r"(*reinterpret_cast<uint32_t*>(&oq[17])), "=r"(*reinterpret_cast<uint32_t*>(&oq[18])), "=r"(*reinterpret_cast<uint32_t*>(&oq[19])), "=r"(*reinterpret_cast<uint32_t*>(&oq[20])), "=r"(*reinterpret_cast<uint32_t*>(&oq[21])), "=r"(*reinterpret_cast<uint32_t*>(&oq[22])), "=r"(*reinterpret_cast<uint32_t*>(&oq[23])), "=r"(*reinterpret_cast<uint32_t*>(&oq[24])), "=r"(*reinterpret_cast<uint32_t*>(&oq[25])), "=r"(*reinterpret_cast<uint32_t*>(&oq[26])), "=r"(*reinterpret_cast<uint32_t*>(&oq[27])), "=r"(*reinterpret_cast<uint32_t*>(&oq[28])), "=r"(*reinterpret_cast<uint32_t*>(&oq[29])), "=r"(*reinterpret_cast<uint32_t*>(&oq[30])), "=r"(*reinterpret_cast<uint32_t*>(&oq[31]))
                            : "r"(o_addr));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int p7 = 0; p7 < 16; p7++) {
                            float2 _f32x2_mul_a_8 = make_float2(oq[p7 * 2], oq[p7 * 2 + 1]);
                            float2 _f32x2_mul_b_9 = make_float2(__uint_as_float(ew5[p7 / 2 * 2]), __uint_as_float(ew5[p7 / 2 * 2 + 1]));
                            float2 oq2;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&oq2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
                            oq[p7 * 2] = oq2.x;
                            oq[p7 * 2 + 1] = oq2.y;
                        }
                        if (vt1 < 64) {
                            #pragma unroll
                            for (int j1w = 0; j1w < 32; j1w++) {
                                int tcolw = j1w / 4 * 8 + lane_quad1 * 2 + (j1w & 1);
                                oq[j1w] = ((tcolw < vt1) ? oq[j1w] : 0.0f);
                            }
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                            :: "r"(o_addr), "r"(*reinterpret_cast<const uint32_t*>(&oq[0])), "r"(*reinterpret_cast<const uint32_t*>(&oq[1])), "r"(*reinterpret_cast<const uint32_t*>(&oq[2])), "r"(*reinterpret_cast<const uint32_t*>(&oq[3])), "r"(*reinterpret_cast<const uint32_t*>(&oq[4])), "r"(*reinterpret_cast<const uint32_t*>(&oq[5])), "r"(*reinterpret_cast<const uint32_t*>(&oq[6])), "r"(*reinterpret_cast<const uint32_t*>(&oq[7])), "r"(*reinterpret_cast<const uint32_t*>(&oq[8])), "r"(*reinterpret_cast<const uint32_t*>(&oq[9])), "r"(*reinterpret_cast<const uint32_t*>(&oq[10])), "r"(*reinterpret_cast<const uint32_t*>(&oq[11])), "r"(*reinterpret_cast<const uint32_t*>(&oq[12])), "r"(*reinterpret_cast<const uint32_t*>(&oq[13])), "r"(*reinterpret_cast<const uint32_t*>(&oq[14])), "r"(*reinterpret_cast<const uint32_t*>(&oq[15])), "r"(*reinterpret_cast<const uint32_t*>(&oq[16])), "r"(*reinterpret_cast<const uint32_t*>(&oq[17])), "r"(*reinterpret_cast<const uint32_t*>(&oq[18])), "r"(*reinterpret_cast<const uint32_t*>(&oq[19])), "r"(*reinterpret_cast<const uint32_t*>(&oq[20])), "r"(*reinterpret_cast<const uint32_t*>(&oq[21])), "r"(*reinterpret_cast<const uint32_t*>(&oq[22])), "r"(*reinterpret_cast<const uint32_t*>(&oq[23])), "r"(*reinterpret_cast<const uint32_t*>(&oq[24])), "r"(*reinterpret_cast<const uint32_t*>(&oq[25])), "r"(*reinterpret_cast<const uint32_t*>(&oq[26])), "r"(*reinterpret_cast<const uint32_t*>(&oq[27])), "r"(*reinterpret_cast<const uint32_t*>(&oq[28])), "r"(*reinterpret_cast<const uint32_t*>(&oq[29])), "r"(*reinterpret_cast<const uint32_t*>(&oq[30])), "r"(*reinterpret_cast<const uint32_t*>(&oq[31])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(qs_scaled_addr);
                    }
                    if (last_in_blk1 != 0) {
                        mbarrier_wait(o_full_addr, _phase_o_full_0);
                        _phase_o_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int o_addr2 = taddr + 128 + (unsigned int)row_base1;
                        float oo[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&oo[0])), "=r"(*reinterpret_cast<uint32_t*>(&oo[1])), "=r"(*reinterpret_cast<uint32_t*>(&oo[2])), "=r"(*reinterpret_cast<uint32_t*>(&oo[3])), "=r"(*reinterpret_cast<uint32_t*>(&oo[4])), "=r"(*reinterpret_cast<uint32_t*>(&oo[5])), "=r"(*reinterpret_cast<uint32_t*>(&oo[6])), "=r"(*reinterpret_cast<uint32_t*>(&oo[7])), "=r"(*reinterpret_cast<uint32_t*>(&oo[8])), "=r"(*reinterpret_cast<uint32_t*>(&oo[9])), "=r"(*reinterpret_cast<uint32_t*>(&oo[10])), "=r"(*reinterpret_cast<uint32_t*>(&oo[11])), "=r"(*reinterpret_cast<uint32_t*>(&oo[12])), "=r"(*reinterpret_cast<uint32_t*>(&oo[13])), "=r"(*reinterpret_cast<uint32_t*>(&oo[14])), "=r"(*reinterpret_cast<uint32_t*>(&oo[15])), "=r"(*reinterpret_cast<uint32_t*>(&oo[16])), "=r"(*reinterpret_cast<uint32_t*>(&oo[17])), "=r"(*reinterpret_cast<uint32_t*>(&oo[18])), "=r"(*reinterpret_cast<uint32_t*>(&oo[19])), "=r"(*reinterpret_cast<uint32_t*>(&oo[20])), "=r"(*reinterpret_cast<uint32_t*>(&oo[21])), "=r"(*reinterpret_cast<uint32_t*>(&oo[22])), "=r"(*reinterpret_cast<uint32_t*>(&oo[23])), "=r"(*reinterpret_cast<uint32_t*>(&oo[24])), "=r"(*reinterpret_cast<uint32_t*>(&oo[25])), "=r"(*reinterpret_cast<uint32_t*>(&oo[26])), "=r"(*reinterpret_cast<uint32_t*>(&oo[27])), "=r"(*reinterpret_cast<uint32_t*>(&oo[28])), "=r"(*reinterpret_cast<uint32_t*>(&oo[29])), "=r"(*reinterpret_cast<uint32_t*>(&oo[30])), "=r"(*reinterpret_cast<uint32_t*>(&oo[31]))
                            : "r"(o_addr2));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(o_empty_addr);
                        if (write1 != 0) {
                            mbarrier_wait(o_stage_empty_addr, _phase_o_stage_empty_0);
                            _phase_o_stage_empty_0 ^= 1;
                            unsigned int oob[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oo[_lp*2 + 0], oo[_lp*2+1 + 0]));
                                oob[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int tg = 0; tg < 4; tg++) {
                                int v_base = warp1 * 16 + (mtx1 & 1) * 8;
                                int tok = tg * 16 + mtx1 / 2 * 8 + row_in_mtx1;
                                unsigned int o_st = smem_o_addr + (unsigned int)(tok * 128 + v_base * 2 ^ (tok * 128 + v_base * 2 >> 7 & 7) << 4);
                                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>((unsigned long long)o_st);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&oob[tg * 4])), "r"(*reinterpret_cast<const uint32_t*>(&oob[tg * 4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&oob[tg * 4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&oob[tg * 4 + 3]))
                                    : "memory");
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_arrive(o_stage_ready_addr);
                        }
                    }
                    mbarrier_arrive(gate_empty_cg1_addr + (g_cur1) * 8);
                    g_cur1 += 1;
                    if (g_cur1 == 4) { g_cur1 = 0; _phase_gate_full_1 ^= 1; }
                    cb1c = cb1c + 1;
                    if (cb1c == num_hh) {
                        cb1c = 0;
                        blk1c = blk1c + 1;
                    }
                }
                if (nch_1 != 0) {
                    mbarrier_wait(state_full_addr, _phase_state_full_0);
                    _phase_state_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (owns_end_1 != 0 && (flags & 2) != 0) {
                        if (abs0_1 != 0 && (flags & 1) != 0) {
                            if (warp1 == 0 && lane == 0) {
                                {
                                    unsigned int _poll_v_11;
                                    const unsigned long long _poll_t0_11 = clock64();
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_poll_v_11) : "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_ws)) + (seq_1 * num_heads + head_1))) : "memory");
                                    while (_poll_v_11 != static_cast<unsigned int>(1)) {
                                        const unsigned long long _poll_now_11 = clock64();
                                        if (_poll_now_11 - _poll_t0_11 > 17179869184ull) {
                                            asm volatile("trap;" ::: "memory");
                                        }
                                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_poll_v_11) : "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_ws)) + (seq_1 * num_heads + head_1))) : "memory");
                                    }
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                        }
                        int wr_s = output_state_indices[seq_1];
                        int wr_ok = ((wr_s >= 0 && (long long)wr_s < output_state_size_p0) ? 1 : 0);
                        long long wr_head = (long long)wr_s * output_state_stride_slot + (long long)head_1 * 8192ll;
                        int rd_s_0 = state_indices[seq_1];
                        int rd_ok_0 = ((rd_s_0 >= 0 && (long long)rd_s_0 < initial_state_size_p0) ? 1 : 0);
                        long long rd_head_0 = (long long)rd_s_0 * initial_state_stride_slot + (long long)head_1 * 8192ll;
                        long long ctr1 = (((flags & 1) != 0 && rd_ok_0 != 0) ? rd_head_0 : wr_head);
                        long long s16_top = wr_head + (long long)row_top1 * 128ll;
                        float s16_lo[32];
                        float s16_hi[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[3])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[5])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[6])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[7])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[8])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[9])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[10])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[11])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[12])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[13])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[14])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[15])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[16])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[17])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[18])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[19])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[20])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[21])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[22])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[23])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[24])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[25])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[26])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[27])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[28])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[29])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[30])), "=r"(*reinterpret_cast<uint32_t*>(&s16_lo[31]))
                            : "r"(taddr + (unsigned int)row_base1));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[3])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[5])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[6])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[7])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[8])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[9])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[10])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[11])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[12])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[13])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[14])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[15])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[16])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[17])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[18])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[19])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[20])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[21])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[22])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[23])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[24])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[25])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[26])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[27])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[28])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[29])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[30])), "=r"(*reinterpret_cast<uint32_t*>(&s16_hi[31]))
                            : "r"(taddr + (unsigned int)row_base1 + 64));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        unsigned int s16_pk[1];
                        unsigned int s16_rb[4];
                        int s16_par = lane_quad1 & 1;
                        if (philox_rounds == 0) {
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 0)[_lp*2 + 0], (s16_lo + 0)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 2)[_lp*2 + 0], (s16_lo + 2)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1024))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 0)[_lp*2 + 0], (s16_hi + 0)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 64))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 2)[_lp*2 + 0], (s16_hi + 2)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1088))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 4)[_lp*2 + 0], (s16_lo + 4)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 8))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 6)[_lp*2 + 0], (s16_lo + 6)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1032))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 4)[_lp*2 + 0], (s16_hi + 4)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 72))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 6)[_lp*2 + 0], (s16_hi + 6)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1096))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 8)[_lp*2 + 0], (s16_lo + 8)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 16))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 10)[_lp*2 + 0], (s16_lo + 10)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1040))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 8)[_lp*2 + 0], (s16_hi + 8)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 80))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 10)[_lp*2 + 0], (s16_hi + 10)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1104))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 12)[_lp*2 + 0], (s16_lo + 12)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 24))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 14)[_lp*2 + 0], (s16_lo + 14)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1048))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 12)[_lp*2 + 0], (s16_hi + 12)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 88))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 14)[_lp*2 + 0], (s16_hi + 14)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1112))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 16)[_lp*2 + 0], (s16_lo + 16)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 32))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 18)[_lp*2 + 0], (s16_lo + 18)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1056))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 16)[_lp*2 + 0], (s16_hi + 16)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 96))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 18)[_lp*2 + 0], (s16_hi + 18)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1120))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 20)[_lp*2 + 0], (s16_lo + 20)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 40))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 22)[_lp*2 + 0], (s16_lo + 22)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1064))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 20)[_lp*2 + 0], (s16_hi + 20)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 104))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 22)[_lp*2 + 0], (s16_hi + 22)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1128))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 24)[_lp*2 + 0], (s16_lo + 24)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 48))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 26)[_lp*2 + 0], (s16_lo + 26)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1072))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 24)[_lp*2 + 0], (s16_hi + 24)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 112))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 26)[_lp*2 + 0], (s16_hi + 26)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1136))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 28)[_lp*2 + 0], (s16_lo + 28)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 56))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_lo + 30)[_lp*2 + 0], (s16_lo + 30)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1080))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 28)[_lp*2 + 0], (s16_hi + 28)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 120))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((s16_hi + 30)[_lp*2 + 0], (s16_hi + 30)[_lp*2+1 + 0]));
                                s16_pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1144))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                        }
                        if (philox_rounds != 0) {
                            long long s16_seed = *rand_seed;
                            unsigned int s16_key_lo = (unsigned int)s16_seed;
                            unsigned int s16_key_hi = (unsigned int)(s16_seed >> 32ll);
                            long long s16_base = ctr1 + (long long)(row_top1 + 8 * s16_par) * 128ll + (long long)((lane_quad1 >> 1) * 4);
                            unsigned int s16_w[16];
                            unsigned int key0 = s16_key_lo;
                            unsigned int key1 = s16_key_hi;
                            long long o_0 = s16_base;
                            unsigned long long p0_0 = (unsigned long long)(unsigned int)o_0 * 3528531795ull;
                            s16_w[0] = (unsigned int)(o_0 >> 32ll) ^ key0;
                            s16_w[2] = (unsigned int)(p0_0 >> 32ull) ^ key1;
                            s16_w[3] = (unsigned int)p0_0;
                            long long o_1 = s16_base + 8ll;
                            unsigned long long p0_1 = (unsigned long long)(unsigned int)o_1 * 3528531795ull;
                            s16_w[4] = (unsigned int)(o_1 >> 32ll) ^ key0;
                            s16_w[6] = (unsigned int)(p0_1 >> 32ull) ^ key1;
                            s16_w[7] = (unsigned int)p0_1;
                            long long o_2 = s16_base + 16ll;
                            unsigned long long p0_2 = (unsigned long long)(unsigned int)o_2 * 3528531795ull;
                            s16_w[8] = (unsigned int)(o_2 >> 32ll) ^ key0;
                            s16_w[10] = (unsigned int)(p0_2 >> 32ull) ^ key1;
                            s16_w[11] = (unsigned int)p0_2;
                            long long o_3 = s16_base + 24ll;
                            unsigned long long p0_3 = (unsigned long long)(unsigned int)o_3 * 3528531795ull;
                            s16_w[12] = (unsigned int)(o_3 >> 32ll) ^ key0;
                            s16_w[14] = (unsigned int)(p0_3 >> 32ull) ^ key1;
                            s16_w[15] = (unsigned int)p0_3;
                            key0 = key0 + 2654435769u;
                            key1 = key1 + 3144134277u;
                            unsigned long long q0_0 = (unsigned long long)s16_w[0] * 3528531795ull;
                            unsigned long long q1_0 = (unsigned long long)s16_w[2] * 3449720151ull;
                            s16_w[0] = (unsigned int)(q1_0 >> 32ull) ^ key0;
                            s16_w[1] = (unsigned int)q1_0;
                            s16_w[2] = (unsigned int)(q0_0 >> 32ull) ^ s16_w[3] ^ key1;
                            s16_w[3] = (unsigned int)q0_0;
                            unsigned long long q0_1 = (unsigned long long)s16_w[4] * 3528531795ull;
                            unsigned long long q1_1 = (unsigned long long)s16_w[6] * 3449720151ull;
                            s16_w[4] = (unsigned int)(q1_1 >> 32ull) ^ key0;
                            s16_w[5] = (unsigned int)q1_1;
                            s16_w[6] = (unsigned int)(q0_1 >> 32ull) ^ s16_w[7] ^ key1;
                            s16_w[7] = (unsigned int)q0_1;
                            unsigned long long q0_2 = (unsigned long long)s16_w[8] * 3528531795ull;
                            unsigned long long q1_2 = (unsigned long long)s16_w[10] * 3449720151ull;
                            s16_w[8] = (unsigned int)(q1_2 >> 32ull) ^ key0;
                            s16_w[9] = (unsigned int)q1_2;
                            s16_w[10] = (unsigned int)(q0_2 >> 32ull) ^ s16_w[11] ^ key1;
                            s16_w[11] = (unsigned int)q0_2;
                            unsigned long long q0_3 = (unsigned long long)s16_w[12] * 3528531795ull;
                            unsigned long long q1_3 = (unsigned long long)s16_w[14] * 3449720151ull;
                            s16_w[12] = (unsigned int)(q1_3 >> 32ull) ^ key0;
                            s16_w[13] = (unsigned int)q1_3;
                            s16_w[14] = (unsigned int)(q0_3 >> 32ull) ^ s16_w[15] ^ key1;
                            s16_w[15] = (unsigned int)q0_3;
                            key0 = key0 + 2654435769u;
                            key1 = key1 + 3144134277u;
                            unsigned long long s0_0 = (unsigned long long)s16_w[0] * 3528531795ull;
                            unsigned long long s1_0 = (unsigned long long)s16_w[2] * 3449720151ull;
                            s16_w[0] = (unsigned int)(s1_0 >> 32ull) ^ s16_w[1] ^ key0;
                            s16_w[1] = (unsigned int)s1_0;
                            s16_w[2] = (unsigned int)(s0_0 >> 32ull) ^ s16_w[3] ^ key1;
                            s16_w[3] = (unsigned int)s0_0;
                            unsigned long long s0_1 = (unsigned long long)s16_w[4] * 3528531795ull;
                            unsigned long long s1_1 = (unsigned long long)s16_w[6] * 3449720151ull;
                            s16_w[4] = (unsigned int)(s1_1 >> 32ull) ^ s16_w[5] ^ key0;
                            s16_w[5] = (unsigned int)s1_1;
                            s16_w[6] = (unsigned int)(s0_1 >> 32ull) ^ s16_w[7] ^ key1;
                            s16_w[7] = (unsigned int)s0_1;
                            unsigned long long s0_2 = (unsigned long long)s16_w[8] * 3528531795ull;
                            unsigned long long s1_2 = (unsigned long long)s16_w[10] * 3449720151ull;
                            s16_w[8] = (unsigned int)(s1_2 >> 32ull) ^ s16_w[9] ^ key0;
                            s16_w[9] = (unsigned int)s1_2;
                            s16_w[10] = (unsigned int)(s0_2 >> 32ull) ^ s16_w[11] ^ key1;
                            s16_w[11] = (unsigned int)s0_2;
                            unsigned long long s0_3 = (unsigned long long)s16_w[12] * 3528531795ull;
                            unsigned long long s1_3 = (unsigned long long)s16_w[14] * 3449720151ull;
                            s16_w[12] = (unsigned int)(s1_3 >> 32ull) ^ s16_w[13] ^ key0;
                            s16_w[13] = (unsigned int)s1_3;
                            s16_w[14] = (unsigned int)(s0_3 >> 32ull) ^ s16_w[15] ^ key1;
                            s16_w[15] = (unsigned int)s0_3;
                            key0 = key0 + 2654435769u;
                            key1 = key1 + 3144134277u;
                            unsigned long long s0_4 = (unsigned long long)s16_w[0] * 3528531795ull;
                            unsigned long long s1_4 = (unsigned long long)s16_w[2] * 3449720151ull;
                            s16_w[0] = (unsigned int)(s1_4 >> 32ull) ^ s16_w[1] ^ key0;
                            s16_w[1] = (unsigned int)s1_4;
                            s16_w[2] = (unsigned int)(s0_4 >> 32ull) ^ s16_w[3] ^ key1;
                            s16_w[3] = (unsigned int)s0_4;
                            unsigned long long s0_5 = (unsigned long long)s16_w[4] * 3528531795ull;
                            unsigned long long s1_5 = (unsigned long long)s16_w[6] * 3449720151ull;
                            s16_w[4] = (unsigned int)(s1_5 >> 32ull) ^ s16_w[5] ^ key0;
                            s16_w[5] = (unsigned int)s1_5;
                            s16_w[6] = (unsigned int)(s0_5 >> 32ull) ^ s16_w[7] ^ key1;
                            s16_w[7] = (unsigned int)s0_5;
                            unsigned long long s0_6 = (unsigned long long)s16_w[8] * 3528531795ull;
                            unsigned long long s1_6 = (unsigned long long)s16_w[10] * 3449720151ull;
                            s16_w[8] = (unsigned int)(s1_6 >> 32ull) ^ s16_w[9] ^ key0;
                            s16_w[9] = (unsigned int)s1_6;
                            s16_w[10] = (unsigned int)(s0_6 >> 32ull) ^ s16_w[11] ^ key1;
                            s16_w[11] = (unsigned int)s0_6;
                            unsigned long long s0_7 = (unsigned long long)s16_w[12] * 3528531795ull;
                            unsigned long long s1_7 = (unsigned long long)s16_w[14] * 3449720151ull;
                            s16_w[12] = (unsigned int)(s1_7 >> 32ull) ^ s16_w[13] ^ key0;
                            s16_w[13] = (unsigned int)s1_7;
                            s16_w[14] = (unsigned int)(s0_7 >> 32ull) ^ s16_w[15] ^ key1;
                            s16_w[15] = (unsigned int)s0_7;
                            key0 = key0 + 2654435769u;
                            key1 = key1 + 3144134277u;
                            unsigned long long s0_8 = (unsigned long long)s16_w[0] * 3528531795ull;
                            unsigned long long s1_8 = (unsigned long long)s16_w[2] * 3449720151ull;
                            s16_w[0] = (unsigned int)(s1_8 >> 32ull) ^ s16_w[1] ^ key0;
                            s16_w[1] = (unsigned int)s1_8;
                            s16_w[2] = (unsigned int)(s0_8 >> 32ull) ^ s16_w[3] ^ key1;
                            s16_w[3] = (unsigned int)s0_8;
                            unsigned long long s0_9 = (unsigned long long)s16_w[4] * 3528531795ull;
                            unsigned long long s1_9 = (unsigned long long)s16_w[6] * 3449720151ull;
                            s16_w[4] = (unsigned int)(s1_9 >> 32ull) ^ s16_w[5] ^ key0;
                            s16_w[5] = (unsigned int)s1_9;
                            s16_w[6] = (unsigned int)(s0_9 >> 32ull) ^ s16_w[7] ^ key1;
                            s16_w[7] = (unsigned int)s0_9;
                            unsigned long long s0_10 = (unsigned long long)s16_w[8] * 3528531795ull;
                            unsigned long long s1_10 = (unsigned long long)s16_w[10] * 3449720151ull;
                            s16_w[8] = (unsigned int)(s1_10 >> 32ull) ^ s16_w[9] ^ key0;
                            s16_w[9] = (unsigned int)s1_10;
                            s16_w[10] = (unsigned int)(s0_10 >> 32ull) ^ s16_w[11] ^ key1;
                            s16_w[11] = (unsigned int)s0_10;
                            unsigned long long s0_11 = (unsigned long long)s16_w[12] * 3528531795ull;
                            unsigned long long s1_11 = (unsigned long long)s16_w[14] * 3449720151ull;
                            s16_w[12] = (unsigned int)(s1_11 >> 32ull) ^ s16_w[13] ^ key0;
                            s16_w[13] = (unsigned int)s1_11;
                            s16_w[14] = (unsigned int)(s0_11 >> 32ull) ^ s16_w[15] ^ key1;
                            s16_w[15] = (unsigned int)s0_11;
                            key0 = key0 + 2654435769u;
                            key1 = key1 + 3144134277u;
                            #pragma unroll 1
                            for (int _round = 5; _round < philox_rounds; _round++) {
                                unsigned long long t0_0 = (unsigned long long)s16_w[0] * 3528531795ull;
                                unsigned long long t1_0 = (unsigned long long)s16_w[2] * 3449720151ull;
                                s16_w[0] = (unsigned int)(t1_0 >> 32ull) ^ s16_w[1] ^ key0;
                                s16_w[1] = (unsigned int)t1_0;
                                s16_w[2] = (unsigned int)(t0_0 >> 32ull) ^ s16_w[3] ^ key1;
                                s16_w[3] = (unsigned int)t0_0;
                                unsigned long long t0_1 = (unsigned long long)s16_w[4] * 3528531795ull;
                                unsigned long long t1_1 = (unsigned long long)s16_w[6] * 3449720151ull;
                                s16_w[4] = (unsigned int)(t1_1 >> 32ull) ^ s16_w[5] ^ key0;
                                s16_w[5] = (unsigned int)t1_1;
                                s16_w[6] = (unsigned int)(t0_1 >> 32ull) ^ s16_w[7] ^ key1;
                                s16_w[7] = (unsigned int)t0_1;
                                unsigned long long t0_2 = (unsigned long long)s16_w[8] * 3528531795ull;
                                unsigned long long t1_2 = (unsigned long long)s16_w[10] * 3449720151ull;
                                s16_w[8] = (unsigned int)(t1_2 >> 32ull) ^ s16_w[9] ^ key0;
                                s16_w[9] = (unsigned int)t1_2;
                                s16_w[10] = (unsigned int)(t0_2 >> 32ull) ^ s16_w[11] ^ key1;
                                s16_w[11] = (unsigned int)t0_2;
                                unsigned long long t0_3 = (unsigned long long)s16_w[12] * 3528531795ull;
                                unsigned long long t1_3 = (unsigned long long)s16_w[14] * 3449720151ull;
                                s16_w[12] = (unsigned int)(t1_3 >> 32ull) ^ s16_w[13] ^ key0;
                                s16_w[13] = (unsigned int)t1_3;
                                s16_w[14] = (unsigned int)(t0_3 >> 32ull) ^ s16_w[15] ^ key1;
                                s16_w[15] = (unsigned int)t0_3;
                                key0 = key0 + 2654435769u;
                                key1 = key1 + 3144134277u;
                            }
                            unsigned int s16_give0_0 = ((s16_par != 0) ? s16_w[0] : s16_w[1]);
                            unsigned int s16_give1_0 = ((s16_par != 0) ? s16_w[2] : s16_w[3]);
                            unsigned int s16_own0_0 = ((s16_par != 0) ? s16_w[1] : s16_w[0]);
                            unsigned int s16_own1_0 = ((s16_par != 0) ? s16_w[3] : s16_w[2]);
                            unsigned int s16_got0_0 = __shfl_xor_sync(0xFFFFFFFF, s16_give0_0, 1);
                            unsigned int s16_got1_0 = __shfl_xor_sync(0xFFFFFFFF, s16_give1_0, 1);
                            s16_rb[0] = ((s16_par != 0) ? s16_got0_0 : s16_own0_0);
                            s16_rb[1] = ((s16_par != 0) ? s16_own0_0 : s16_got0_0);
                            s16_rb[2] = ((s16_par != 0) ? s16_got1_0 : s16_own1_0);
                            s16_rb[3] = ((s16_par != 0) ? s16_own1_0 : s16_got1_0);
                            uint32_t _cvt_rs_x2_f32_12;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_12) : "f"(s16_lo[1]), "f"(s16_lo[0]), "r"(s16_rb[0]));
                            s16_pk[0] = _cvt_rs_x2_f32_12;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_13;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_13) : "f"(s16_lo[3]), "f"(s16_lo[2]), "r"(s16_rb[1]));
                            s16_pk[0] = _cvt_rs_x2_f32_13;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1024))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_14;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_14) : "f"(s16_hi[1]), "f"(s16_hi[0]), "r"(s16_rb[2]));
                            s16_pk[0] = _cvt_rs_x2_f32_14;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 64))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_15;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_15) : "f"(s16_hi[3]), "f"(s16_hi[2]), "r"(s16_rb[3]));
                            s16_pk[0] = _cvt_rs_x2_f32_15;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1088))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            unsigned int s16_give0_1 = ((s16_par != 0) ? s16_w[4] : s16_w[5]);
                            unsigned int s16_give1_1 = ((s16_par != 0) ? s16_w[6] : s16_w[7]);
                            unsigned int s16_own0_1 = ((s16_par != 0) ? s16_w[5] : s16_w[4]);
                            unsigned int s16_own1_1 = ((s16_par != 0) ? s16_w[7] : s16_w[6]);
                            unsigned int s16_got0_1 = __shfl_xor_sync(0xFFFFFFFF, s16_give0_1, 1);
                            unsigned int s16_got1_1 = __shfl_xor_sync(0xFFFFFFFF, s16_give1_1, 1);
                            s16_rb[0] = ((s16_par != 0) ? s16_got0_1 : s16_own0_1);
                            s16_rb[1] = ((s16_par != 0) ? s16_own0_1 : s16_got0_1);
                            s16_rb[2] = ((s16_par != 0) ? s16_got1_1 : s16_own1_1);
                            s16_rb[3] = ((s16_par != 0) ? s16_own1_1 : s16_got1_1);
                            uint32_t _cvt_rs_x2_f32_16;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_16) : "f"(s16_lo[5]), "f"(s16_lo[4]), "r"(s16_rb[0]));
                            s16_pk[0] = _cvt_rs_x2_f32_16;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 8))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_17;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_17) : "f"(s16_lo[7]), "f"(s16_lo[6]), "r"(s16_rb[1]));
                            s16_pk[0] = _cvt_rs_x2_f32_17;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1032))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_18;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_18) : "f"(s16_hi[5]), "f"(s16_hi[4]), "r"(s16_rb[2]));
                            s16_pk[0] = _cvt_rs_x2_f32_18;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 72))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_19;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_19) : "f"(s16_hi[7]), "f"(s16_hi[6]), "r"(s16_rb[3]));
                            s16_pk[0] = _cvt_rs_x2_f32_19;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1096))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            unsigned int s16_give0_2 = ((s16_par != 0) ? s16_w[8] : s16_w[9]);
                            unsigned int s16_give1_2 = ((s16_par != 0) ? s16_w[10] : s16_w[11]);
                            unsigned int s16_own0_2 = ((s16_par != 0) ? s16_w[9] : s16_w[8]);
                            unsigned int s16_own1_2 = ((s16_par != 0) ? s16_w[11] : s16_w[10]);
                            unsigned int s16_got0_2 = __shfl_xor_sync(0xFFFFFFFF, s16_give0_2, 1);
                            unsigned int s16_got1_2 = __shfl_xor_sync(0xFFFFFFFF, s16_give1_2, 1);
                            s16_rb[0] = ((s16_par != 0) ? s16_got0_2 : s16_own0_2);
                            s16_rb[1] = ((s16_par != 0) ? s16_own0_2 : s16_got0_2);
                            s16_rb[2] = ((s16_par != 0) ? s16_got1_2 : s16_own1_2);
                            s16_rb[3] = ((s16_par != 0) ? s16_own1_2 : s16_got1_2);
                            uint32_t _cvt_rs_x2_f32_20;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_20) : "f"(s16_lo[9]), "f"(s16_lo[8]), "r"(s16_rb[0]));
                            s16_pk[0] = _cvt_rs_x2_f32_20;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 16))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_21;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_21) : "f"(s16_lo[11]), "f"(s16_lo[10]), "r"(s16_rb[1]));
                            s16_pk[0] = _cvt_rs_x2_f32_21;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1040))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_22;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_22) : "f"(s16_hi[9]), "f"(s16_hi[8]), "r"(s16_rb[2]));
                            s16_pk[0] = _cvt_rs_x2_f32_22;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 80))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_23;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_23) : "f"(s16_hi[11]), "f"(s16_hi[10]), "r"(s16_rb[3]));
                            s16_pk[0] = _cvt_rs_x2_f32_23;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1104))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            unsigned int s16_give0_3 = ((s16_par != 0) ? s16_w[12] : s16_w[13]);
                            unsigned int s16_give1_3 = ((s16_par != 0) ? s16_w[14] : s16_w[15]);
                            unsigned int s16_own0_3 = ((s16_par != 0) ? s16_w[13] : s16_w[12]);
                            unsigned int s16_own1_3 = ((s16_par != 0) ? s16_w[15] : s16_w[14]);
                            unsigned int s16_got0_3 = __shfl_xor_sync(0xFFFFFFFF, s16_give0_3, 1);
                            unsigned int s16_got1_3 = __shfl_xor_sync(0xFFFFFFFF, s16_give1_3, 1);
                            s16_rb[0] = ((s16_par != 0) ? s16_got0_3 : s16_own0_3);
                            s16_rb[1] = ((s16_par != 0) ? s16_own0_3 : s16_got0_3);
                            s16_rb[2] = ((s16_par != 0) ? s16_got1_3 : s16_own1_3);
                            s16_rb[3] = ((s16_par != 0) ? s16_own1_3 : s16_got1_3);
                            uint32_t _cvt_rs_x2_f32_24;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_24) : "f"(s16_lo[13]), "f"(s16_lo[12]), "r"(s16_rb[0]));
                            s16_pk[0] = _cvt_rs_x2_f32_24;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 24))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_25;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_25) : "f"(s16_lo[15]), "f"(s16_lo[14]), "r"(s16_rb[1]));
                            s16_pk[0] = _cvt_rs_x2_f32_25;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1048))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_26;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_26) : "f"(s16_hi[13]), "f"(s16_hi[12]), "r"(s16_rb[2]));
                            s16_pk[0] = _cvt_rs_x2_f32_26;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 88))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_27;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_27) : "f"(s16_hi[15]), "f"(s16_hi[14]), "r"(s16_rb[3]));
                            s16_pk[0] = _cvt_rs_x2_f32_27;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1112))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            unsigned int key0_0 = s16_key_lo;
                            unsigned int key1_0 = s16_key_hi;
                            long long o_4 = s16_base + 32ll;
                            unsigned long long p0_4 = (unsigned long long)(unsigned int)o_4 * 3528531795ull;
                            s16_w[0] = (unsigned int)(o_4 >> 32ll) ^ key0_0;
                            s16_w[2] = (unsigned int)(p0_4 >> 32ull) ^ key1_0;
                            s16_w[3] = (unsigned int)p0_4;
                            long long o_5 = s16_base + 40ll;
                            unsigned long long p0_5 = (unsigned long long)(unsigned int)o_5 * 3528531795ull;
                            s16_w[4] = (unsigned int)(o_5 >> 32ll) ^ key0_0;
                            s16_w[6] = (unsigned int)(p0_5 >> 32ull) ^ key1_0;
                            s16_w[7] = (unsigned int)p0_5;
                            long long o_6 = s16_base + 48ll;
                            unsigned long long p0_6 = (unsigned long long)(unsigned int)o_6 * 3528531795ull;
                            s16_w[8] = (unsigned int)(o_6 >> 32ll) ^ key0_0;
                            s16_w[10] = (unsigned int)(p0_6 >> 32ull) ^ key1_0;
                            s16_w[11] = (unsigned int)p0_6;
                            long long o_7 = s16_base + 56ll;
                            unsigned long long p0_7 = (unsigned long long)(unsigned int)o_7 * 3528531795ull;
                            s16_w[12] = (unsigned int)(o_7 >> 32ll) ^ key0_0;
                            s16_w[14] = (unsigned int)(p0_7 >> 32ull) ^ key1_0;
                            s16_w[15] = (unsigned int)p0_7;
                            key0_0 = key0_0 + 2654435769u;
                            key1_0 = key1_0 + 3144134277u;
                            unsigned long long q0_4 = (unsigned long long)s16_w[0] * 3528531795ull;
                            unsigned long long q1_4 = (unsigned long long)s16_w[2] * 3449720151ull;
                            s16_w[0] = (unsigned int)(q1_4 >> 32ull) ^ key0_0;
                            s16_w[1] = (unsigned int)q1_4;
                            s16_w[2] = (unsigned int)(q0_4 >> 32ull) ^ s16_w[3] ^ key1_0;
                            s16_w[3] = (unsigned int)q0_4;
                            unsigned long long q0_5 = (unsigned long long)s16_w[4] * 3528531795ull;
                            unsigned long long q1_5 = (unsigned long long)s16_w[6] * 3449720151ull;
                            s16_w[4] = (unsigned int)(q1_5 >> 32ull) ^ key0_0;
                            s16_w[5] = (unsigned int)q1_5;
                            s16_w[6] = (unsigned int)(q0_5 >> 32ull) ^ s16_w[7] ^ key1_0;
                            s16_w[7] = (unsigned int)q0_5;
                            unsigned long long q0_6 = (unsigned long long)s16_w[8] * 3528531795ull;
                            unsigned long long q1_6 = (unsigned long long)s16_w[10] * 3449720151ull;
                            s16_w[8] = (unsigned int)(q1_6 >> 32ull) ^ key0_0;
                            s16_w[9] = (unsigned int)q1_6;
                            s16_w[10] = (unsigned int)(q0_6 >> 32ull) ^ s16_w[11] ^ key1_0;
                            s16_w[11] = (unsigned int)q0_6;
                            unsigned long long q0_7 = (unsigned long long)s16_w[12] * 3528531795ull;
                            unsigned long long q1_7 = (unsigned long long)s16_w[14] * 3449720151ull;
                            s16_w[12] = (unsigned int)(q1_7 >> 32ull) ^ key0_0;
                            s16_w[13] = (unsigned int)q1_7;
                            s16_w[14] = (unsigned int)(q0_7 >> 32ull) ^ s16_w[15] ^ key1_0;
                            s16_w[15] = (unsigned int)q0_7;
                            key0_0 = key0_0 + 2654435769u;
                            key1_0 = key1_0 + 3144134277u;
                            unsigned long long s0_12 = (unsigned long long)s16_w[0] * 3528531795ull;
                            unsigned long long s1_12 = (unsigned long long)s16_w[2] * 3449720151ull;
                            s16_w[0] = (unsigned int)(s1_12 >> 32ull) ^ s16_w[1] ^ key0_0;
                            s16_w[1] = (unsigned int)s1_12;
                            s16_w[2] = (unsigned int)(s0_12 >> 32ull) ^ s16_w[3] ^ key1_0;
                            s16_w[3] = (unsigned int)s0_12;
                            unsigned long long s0_13 = (unsigned long long)s16_w[4] * 3528531795ull;
                            unsigned long long s1_13 = (unsigned long long)s16_w[6] * 3449720151ull;
                            s16_w[4] = (unsigned int)(s1_13 >> 32ull) ^ s16_w[5] ^ key0_0;
                            s16_w[5] = (unsigned int)s1_13;
                            s16_w[6] = (unsigned int)(s0_13 >> 32ull) ^ s16_w[7] ^ key1_0;
                            s16_w[7] = (unsigned int)s0_13;
                            unsigned long long s0_14 = (unsigned long long)s16_w[8] * 3528531795ull;
                            unsigned long long s1_14 = (unsigned long long)s16_w[10] * 3449720151ull;
                            s16_w[8] = (unsigned int)(s1_14 >> 32ull) ^ s16_w[9] ^ key0_0;
                            s16_w[9] = (unsigned int)s1_14;
                            s16_w[10] = (unsigned int)(s0_14 >> 32ull) ^ s16_w[11] ^ key1_0;
                            s16_w[11] = (unsigned int)s0_14;
                            unsigned long long s0_15 = (unsigned long long)s16_w[12] * 3528531795ull;
                            unsigned long long s1_15 = (unsigned long long)s16_w[14] * 3449720151ull;
                            s16_w[12] = (unsigned int)(s1_15 >> 32ull) ^ s16_w[13] ^ key0_0;
                            s16_w[13] = (unsigned int)s1_15;
                            s16_w[14] = (unsigned int)(s0_15 >> 32ull) ^ s16_w[15] ^ key1_0;
                            s16_w[15] = (unsigned int)s0_15;
                            key0_0 = key0_0 + 2654435769u;
                            key1_0 = key1_0 + 3144134277u;
                            unsigned long long s0_16 = (unsigned long long)s16_w[0] * 3528531795ull;
                            unsigned long long s1_16 = (unsigned long long)s16_w[2] * 3449720151ull;
                            s16_w[0] = (unsigned int)(s1_16 >> 32ull) ^ s16_w[1] ^ key0_0;
                            s16_w[1] = (unsigned int)s1_16;
                            s16_w[2] = (unsigned int)(s0_16 >> 32ull) ^ s16_w[3] ^ key1_0;
                            s16_w[3] = (unsigned int)s0_16;
                            unsigned long long s0_17 = (unsigned long long)s16_w[4] * 3528531795ull;
                            unsigned long long s1_17 = (unsigned long long)s16_w[6] * 3449720151ull;
                            s16_w[4] = (unsigned int)(s1_17 >> 32ull) ^ s16_w[5] ^ key0_0;
                            s16_w[5] = (unsigned int)s1_17;
                            s16_w[6] = (unsigned int)(s0_17 >> 32ull) ^ s16_w[7] ^ key1_0;
                            s16_w[7] = (unsigned int)s0_17;
                            unsigned long long s0_18 = (unsigned long long)s16_w[8] * 3528531795ull;
                            unsigned long long s1_18 = (unsigned long long)s16_w[10] * 3449720151ull;
                            s16_w[8] = (unsigned int)(s1_18 >> 32ull) ^ s16_w[9] ^ key0_0;
                            s16_w[9] = (unsigned int)s1_18;
                            s16_w[10] = (unsigned int)(s0_18 >> 32ull) ^ s16_w[11] ^ key1_0;
                            s16_w[11] = (unsigned int)s0_18;
                            unsigned long long s0_19 = (unsigned long long)s16_w[12] * 3528531795ull;
                            unsigned long long s1_19 = (unsigned long long)s16_w[14] * 3449720151ull;
                            s16_w[12] = (unsigned int)(s1_19 >> 32ull) ^ s16_w[13] ^ key0_0;
                            s16_w[13] = (unsigned int)s1_19;
                            s16_w[14] = (unsigned int)(s0_19 >> 32ull) ^ s16_w[15] ^ key1_0;
                            s16_w[15] = (unsigned int)s0_19;
                            key0_0 = key0_0 + 2654435769u;
                            key1_0 = key1_0 + 3144134277u;
                            unsigned long long s0_20 = (unsigned long long)s16_w[0] * 3528531795ull;
                            unsigned long long s1_20 = (unsigned long long)s16_w[2] * 3449720151ull;
                            s16_w[0] = (unsigned int)(s1_20 >> 32ull) ^ s16_w[1] ^ key0_0;
                            s16_w[1] = (unsigned int)s1_20;
                            s16_w[2] = (unsigned int)(s0_20 >> 32ull) ^ s16_w[3] ^ key1_0;
                            s16_w[3] = (unsigned int)s0_20;
                            unsigned long long s0_21 = (unsigned long long)s16_w[4] * 3528531795ull;
                            unsigned long long s1_21 = (unsigned long long)s16_w[6] * 3449720151ull;
                            s16_w[4] = (unsigned int)(s1_21 >> 32ull) ^ s16_w[5] ^ key0_0;
                            s16_w[5] = (unsigned int)s1_21;
                            s16_w[6] = (unsigned int)(s0_21 >> 32ull) ^ s16_w[7] ^ key1_0;
                            s16_w[7] = (unsigned int)s0_21;
                            unsigned long long s0_22 = (unsigned long long)s16_w[8] * 3528531795ull;
                            unsigned long long s1_22 = (unsigned long long)s16_w[10] * 3449720151ull;
                            s16_w[8] = (unsigned int)(s1_22 >> 32ull) ^ s16_w[9] ^ key0_0;
                            s16_w[9] = (unsigned int)s1_22;
                            s16_w[10] = (unsigned int)(s0_22 >> 32ull) ^ s16_w[11] ^ key1_0;
                            s16_w[11] = (unsigned int)s0_22;
                            unsigned long long s0_23 = (unsigned long long)s16_w[12] * 3528531795ull;
                            unsigned long long s1_23 = (unsigned long long)s16_w[14] * 3449720151ull;
                            s16_w[12] = (unsigned int)(s1_23 >> 32ull) ^ s16_w[13] ^ key0_0;
                            s16_w[13] = (unsigned int)s1_23;
                            s16_w[14] = (unsigned int)(s0_23 >> 32ull) ^ s16_w[15] ^ key1_0;
                            s16_w[15] = (unsigned int)s0_23;
                            key0_0 = key0_0 + 2654435769u;
                            key1_0 = key1_0 + 3144134277u;
                            #pragma unroll 1
                            for (int _round_1 = 5; _round_1 < philox_rounds; _round_1++) {
                                unsigned long long t0_4 = (unsigned long long)s16_w[0] * 3528531795ull;
                                unsigned long long t1_4 = (unsigned long long)s16_w[2] * 3449720151ull;
                                s16_w[0] = (unsigned int)(t1_4 >> 32ull) ^ s16_w[1] ^ key0_0;
                                s16_w[1] = (unsigned int)t1_4;
                                s16_w[2] = (unsigned int)(t0_4 >> 32ull) ^ s16_w[3] ^ key1_0;
                                s16_w[3] = (unsigned int)t0_4;
                                unsigned long long t0_5 = (unsigned long long)s16_w[4] * 3528531795ull;
                                unsigned long long t1_5 = (unsigned long long)s16_w[6] * 3449720151ull;
                                s16_w[4] = (unsigned int)(t1_5 >> 32ull) ^ s16_w[5] ^ key0_0;
                                s16_w[5] = (unsigned int)t1_5;
                                s16_w[6] = (unsigned int)(t0_5 >> 32ull) ^ s16_w[7] ^ key1_0;
                                s16_w[7] = (unsigned int)t0_5;
                                unsigned long long t0_6 = (unsigned long long)s16_w[8] * 3528531795ull;
                                unsigned long long t1_6 = (unsigned long long)s16_w[10] * 3449720151ull;
                                s16_w[8] = (unsigned int)(t1_6 >> 32ull) ^ s16_w[9] ^ key0_0;
                                s16_w[9] = (unsigned int)t1_6;
                                s16_w[10] = (unsigned int)(t0_6 >> 32ull) ^ s16_w[11] ^ key1_0;
                                s16_w[11] = (unsigned int)t0_6;
                                unsigned long long t0_7 = (unsigned long long)s16_w[12] * 3528531795ull;
                                unsigned long long t1_7 = (unsigned long long)s16_w[14] * 3449720151ull;
                                s16_w[12] = (unsigned int)(t1_7 >> 32ull) ^ s16_w[13] ^ key0_0;
                                s16_w[13] = (unsigned int)t1_7;
                                s16_w[14] = (unsigned int)(t0_7 >> 32ull) ^ s16_w[15] ^ key1_0;
                                s16_w[15] = (unsigned int)t0_7;
                                key0_0 = key0_0 + 2654435769u;
                                key1_0 = key1_0 + 3144134277u;
                            }
                            unsigned int s16_give0_4 = ((s16_par != 0) ? s16_w[0] : s16_w[1]);
                            unsigned int s16_give1_4 = ((s16_par != 0) ? s16_w[2] : s16_w[3]);
                            unsigned int s16_own0_4 = ((s16_par != 0) ? s16_w[1] : s16_w[0]);
                            unsigned int s16_own1_4 = ((s16_par != 0) ? s16_w[3] : s16_w[2]);
                            unsigned int s16_got0_4 = __shfl_xor_sync(0xFFFFFFFF, s16_give0_4, 1);
                            unsigned int s16_got1_4 = __shfl_xor_sync(0xFFFFFFFF, s16_give1_4, 1);
                            s16_rb[0] = ((s16_par != 0) ? s16_got0_4 : s16_own0_4);
                            s16_rb[1] = ((s16_par != 0) ? s16_own0_4 : s16_got0_4);
                            s16_rb[2] = ((s16_par != 0) ? s16_got1_4 : s16_own1_4);
                            s16_rb[3] = ((s16_par != 0) ? s16_own1_4 : s16_got1_4);
                            uint32_t _cvt_rs_x2_f32_28;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_28) : "f"(s16_lo[17]), "f"(s16_lo[16]), "r"(s16_rb[0]));
                            s16_pk[0] = _cvt_rs_x2_f32_28;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 32))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_29;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_29) : "f"(s16_lo[19]), "f"(s16_lo[18]), "r"(s16_rb[1]));
                            s16_pk[0] = _cvt_rs_x2_f32_29;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1056))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_30;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_30) : "f"(s16_hi[17]), "f"(s16_hi[16]), "r"(s16_rb[2]));
                            s16_pk[0] = _cvt_rs_x2_f32_30;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 96))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_31;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_31) : "f"(s16_hi[19]), "f"(s16_hi[18]), "r"(s16_rb[3]));
                            s16_pk[0] = _cvt_rs_x2_f32_31;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1120))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            unsigned int s16_give0_5 = ((s16_par != 0) ? s16_w[4] : s16_w[5]);
                            unsigned int s16_give1_5 = ((s16_par != 0) ? s16_w[6] : s16_w[7]);
                            unsigned int s16_own0_5 = ((s16_par != 0) ? s16_w[5] : s16_w[4]);
                            unsigned int s16_own1_5 = ((s16_par != 0) ? s16_w[7] : s16_w[6]);
                            unsigned int s16_got0_5 = __shfl_xor_sync(0xFFFFFFFF, s16_give0_5, 1);
                            unsigned int s16_got1_5 = __shfl_xor_sync(0xFFFFFFFF, s16_give1_5, 1);
                            s16_rb[0] = ((s16_par != 0) ? s16_got0_5 : s16_own0_5);
                            s16_rb[1] = ((s16_par != 0) ? s16_own0_5 : s16_got0_5);
                            s16_rb[2] = ((s16_par != 0) ? s16_got1_5 : s16_own1_5);
                            s16_rb[3] = ((s16_par != 0) ? s16_own1_5 : s16_got1_5);
                            uint32_t _cvt_rs_x2_f32_32;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_32) : "f"(s16_lo[21]), "f"(s16_lo[20]), "r"(s16_rb[0]));
                            s16_pk[0] = _cvt_rs_x2_f32_32;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 40))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_33;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_33) : "f"(s16_lo[23]), "f"(s16_lo[22]), "r"(s16_rb[1]));
                            s16_pk[0] = _cvt_rs_x2_f32_33;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1064))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_34;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_34) : "f"(s16_hi[21]), "f"(s16_hi[20]), "r"(s16_rb[2]));
                            s16_pk[0] = _cvt_rs_x2_f32_34;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 104))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_35;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_35) : "f"(s16_hi[23]), "f"(s16_hi[22]), "r"(s16_rb[3]));
                            s16_pk[0] = _cvt_rs_x2_f32_35;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1128))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            unsigned int s16_give0_6 = ((s16_par != 0) ? s16_w[8] : s16_w[9]);
                            unsigned int s16_give1_6 = ((s16_par != 0) ? s16_w[10] : s16_w[11]);
                            unsigned int s16_own0_6 = ((s16_par != 0) ? s16_w[9] : s16_w[8]);
                            unsigned int s16_own1_6 = ((s16_par != 0) ? s16_w[11] : s16_w[10]);
                            unsigned int s16_got0_6 = __shfl_xor_sync(0xFFFFFFFF, s16_give0_6, 1);
                            unsigned int s16_got1_6 = __shfl_xor_sync(0xFFFFFFFF, s16_give1_6, 1);
                            s16_rb[0] = ((s16_par != 0) ? s16_got0_6 : s16_own0_6);
                            s16_rb[1] = ((s16_par != 0) ? s16_own0_6 : s16_got0_6);
                            s16_rb[2] = ((s16_par != 0) ? s16_got1_6 : s16_own1_6);
                            s16_rb[3] = ((s16_par != 0) ? s16_own1_6 : s16_got1_6);
                            uint32_t _cvt_rs_x2_f32_36;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_36) : "f"(s16_lo[25]), "f"(s16_lo[24]), "r"(s16_rb[0]));
                            s16_pk[0] = _cvt_rs_x2_f32_36;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 48))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_37;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_37) : "f"(s16_lo[27]), "f"(s16_lo[26]), "r"(s16_rb[1]));
                            s16_pk[0] = _cvt_rs_x2_f32_37;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1072))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_38;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_38) : "f"(s16_hi[25]), "f"(s16_hi[24]), "r"(s16_rb[2]));
                            s16_pk[0] = _cvt_rs_x2_f32_38;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 112))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_39;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_39) : "f"(s16_hi[27]), "f"(s16_hi[26]), "r"(s16_rb[3]));
                            s16_pk[0] = _cvt_rs_x2_f32_39;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1136))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            unsigned int s16_give0_7 = ((s16_par != 0) ? s16_w[12] : s16_w[13]);
                            unsigned int s16_give1_7 = ((s16_par != 0) ? s16_w[14] : s16_w[15]);
                            unsigned int s16_own0_7 = ((s16_par != 0) ? s16_w[13] : s16_w[12]);
                            unsigned int s16_own1_7 = ((s16_par != 0) ? s16_w[15] : s16_w[14]);
                            unsigned int s16_got0_7 = __shfl_xor_sync(0xFFFFFFFF, s16_give0_7, 1);
                            unsigned int s16_got1_7 = __shfl_xor_sync(0xFFFFFFFF, s16_give1_7, 1);
                            s16_rb[0] = ((s16_par != 0) ? s16_got0_7 : s16_own0_7);
                            s16_rb[1] = ((s16_par != 0) ? s16_own0_7 : s16_got0_7);
                            s16_rb[2] = ((s16_par != 0) ? s16_got1_7 : s16_own1_7);
                            s16_rb[3] = ((s16_par != 0) ? s16_own1_7 : s16_got1_7);
                            uint32_t _cvt_rs_x2_f32_40;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_40) : "f"(s16_lo[29]), "f"(s16_lo[28]), "r"(s16_rb[0]));
                            s16_pk[0] = _cvt_rs_x2_f32_40;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 56))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_41;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_41) : "f"(s16_lo[31]), "f"(s16_lo[30]), "r"(s16_rb[1]));
                            s16_pk[0] = _cvt_rs_x2_f32_41;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1080))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_42;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_42) : "f"(s16_hi[29]), "f"(s16_hi[28]), "r"(s16_rb[2]));
                            s16_pk[0] = _cvt_rs_x2_f32_42;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 120))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                            uint32_t _cvt_rs_x2_f32_43;
                            asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_43) : "f"(s16_hi[31]), "f"(s16_hi[30]), "r"(s16_rb[3]));
                            s16_pk[0] = _cvt_rs_x2_f32_43;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(output_state + (s16_top + (long long)(lane_quad1 * 2 + 1144))), "r"(s16_pk[(0) + 0]), "r"((unsigned int)(wr_ok != 0)) : "memory");
                        }
                        if (abs0_1 != 0 && (flags & 1) != 0) {
                            if (warp1 == 0 && lane == 0) {
                                asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(split_ws)) + (seq_1 * num_heads + head_1))), "r"(static_cast<unsigned int>(0)) : "memory");
                            }
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                }
                if (nch_1 == 0) {
                    if (owns_end_1 != 0 && (flags & 2) != 0) {
                        int wr_s_1 = output_state_indices[seq_1];
                        int wr_ok_1 = ((wr_s_1 >= 0 && (long long)wr_s_1 < output_state_size_p0) ? 1 : 0);
                        long long wr_head_1 = (long long)wr_s_1 * output_state_stride_slot + (long long)head_1 * 8192ll;
                        int rd_s_1 = state_indices[seq_1];
                        int rd_ok_1 = ((rd_s_1 >= 0 && (long long)rd_s_1 < initial_state_size_p0) ? 1 : 0);
                        long long rd_head_1 = (long long)rd_s_1 * initial_state_stride_slot + (long long)head_1 * 8192ll;
                        if (wr_ok_1 != 0) {
                            long long cs_rtop = rd_head_1 + (long long)row_top1 * 128ll;
                            long long cs_wtop = wr_head_1 + (long long)row_top1 * 128ll;
                            #pragma unroll
                            for (int cs_half = 0; cs_half < 2; cs_half++) {
                                unsigned int cs_frag[16];
                                cs_frag[0] = 0;
                                cs_frag[1] = 0;
                                cs_frag[2] = 0;
                                cs_frag[3] = 0;
                                cs_frag[4] = 0;
                                cs_frag[5] = 0;
                                cs_frag[6] = 0;
                                cs_frag[7] = 0;
                                cs_frag[8] = 0;
                                cs_frag[9] = 0;
                                cs_frag[10] = 0;
                                cs_frag[11] = 0;
                                cs_frag[12] = 0;
                                cs_frag[13] = 0;
                                cs_frag[14] = 0;
                                cs_frag[15] = 0;
                                if (seed_init1 != 0) {
                                    #pragma unroll
                                    for (int cs_group = 0; cs_group < 8; cs_group++) {
                                        long long cs_col = (long long)(cs_half * 64 + cs_group * 8 + lane_quad1 * 2);
                                        {
                                            uint32_t _bf16x2_bits_44;
                                            _bf16x2_bits_44 = *reinterpret_cast<const uint32_t*>(initial_state + cs_rtop + cs_col);
                                            cs_frag[2 * cs_group] = _bf16x2_bits_44;
                                        }
                                        {
                                            uint32_t _bf16x2_bits_45;
                                            _bf16x2_bits_45 = *reinterpret_cast<const uint32_t*>(initial_state + cs_rtop + 1024ll + cs_col);
                                            cs_frag[2 * cs_group + 1] = _bf16x2_bits_45;
                                        }
                                    }
                                }
                                #pragma unroll
                                for (int cs_group2 = 0; cs_group2 < 8; cs_group2++) {
                                    long long cs_col2 = (long long)(cs_half * 64 + cs_group2 * 8 + lane_quad1 * 2);
                                    reinterpret_cast<unsigned*>(output_state + (cs_wtop + cs_col2))[0] = reinterpret_cast<unsigned*>(cs_frag + (2 * cs_group2))[0];
                                    reinterpret_cast<unsigned*>(output_state + (cs_wtop + 1024ll + cs_col2))[0] = reinterpret_cast<unsigned*>(cs_frag + (2 * cs_group2 + 1))[0];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: gate_w ----
    if (warp == 8) {
        { // gate_w_main
            unsigned int g_cur8 = 0;
            int lane8 = lane;
            int w8_cur = 0;
            int ord8 = 0;
            asm volatile("barrier.sync 3, 384;" ::: "memory");
            unsigned int _phase_gate_empty_cg0 = 1;
            #pragma unroll 1
            for (unsigned int it8 = ((smem_scan[0] == 0) ? (unsigned int)smem_scan[2] : (unsigned int)smem_scan[1]); it8 < (unsigned int)smem_scan[1]; it8++) {
                int xh_2 = 0;
                int seq_2 = items[(int)it8 * 4];
                int head_2 = items[(int)it8 * 4 + 1];
                int cut_2 = items[(int)it8 * 4 + 2];
                int span_2 = items[(int)it8 * 4 + 3];
                int rs_2 = cu_seqlens[seq_2];
                int re_2 = cu_seqlens[seq_2 + 1];
                int start_2 = ((rs_2 < 0) ? 0 : ((rs_2 > total_tokens) ? total_tokens : rs_2));
                int end_2 = ((re_2 < start_2) ? start_2 : ((re_2 > total_tokens) ? total_tokens : re_2));
                int tok0_2 = start_2 + cut_2 * 64;
                int tokend_2 = start_2 + (cut_2 + span_2) * 64;
                tokend_2 = ((end_2 < tokend_2) ? end_2 : tokend_2);
                int sp_at_2 = num_seqs * num_heads + (int)it8 * 2;
                int sp_lo_2 = (int)reinterpret_cast<unsigned int*>(split_ws)[sp_at_2];
                int sp_num_2 = (int)reinterpret_cast<unsigned int*>(split_ws)[sp_at_2 + 1];
                int sp_wend_2 = sp_lo_2 + sp_num_2;
                int start_tok_2 = start_2;
                tokend_2 = start_2 + (sp_wend_2 + num_hh - 1) / num_hh * 64;
                tokend_2 = ((end_2 < tokend_2) ? end_2 : tokend_2);
                int cstart_2 = sp_lo_2;
                int abs0_2 = cut_2;
                int nch_2 = ((sp_num_2 == 0) ? 0 : (tokend_2 - start_2 + 63) / 64 * num_hh - sp_lo_2);
                int wstart_2 = cut_2;
                int owns_end_2 = (((sp_num_2 > 0 || cut_2 == 0) && sp_wend_2 == 2 * (((end_2 - start_2) * num_hh + 127) / 128)) ? 1 : 0);
                float la = 0.0f;
                float lb = 0.0f;
                int vt8 = 0;
                #pragma unroll 1
                for (int i8 = 0; i8 < nch_2; i8++) {
                    int x8 = cstart_2 + i8;
                    int blk8 = x8 / num_hh;
                    int cb8 = x8 - blk8 * num_hh;
                    int tok0_blk8 = start_tok_2 + blk8 * 64;
                    if (i8 == 0 || cb8 == 0) {
                        vt8 = tokend_2 - tok0_blk8;
                        vt8 = ((vt8 > 64) ? 64 : ((vt8 < 0) ? 0 : vt8));
                        int ta = lane8;
                        int tb = lane8 + 32;
                        float ga = 1.0f;
                        float gb = 1.0f;
                        if (ta < vt8) {
                            ga = gate[(tok0_blk8 + ta) * num_heads + head_2];
                        }
                        if (tb < vt8) {
                            gb = gate[(tok0_blk8 + tb) * num_heads + head_2];
                        }
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(ga + 1e-10f));
                        la = ((ta < vt8) ? _log2_0 : 0.0f);
                        float _log2_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(gb + 1e-10f));
                        lb = ((tb < vt8) ? _log2_1 : 0.0f);
                        float ua_0 = __shfl_up_sync(0xFFFFFFFF, la, 1, 32);
                        float ub8_0 = __shfl_up_sync(0xFFFFFFFF, lb, 1, 32);
                        if (lane8 >= 1) {
                            la = la + ua_0;
                            lb = lb + ub8_0;
                        }
                        float ua_1 = __shfl_up_sync(0xFFFFFFFF, la, 2, 32);
                        float ub8_1 = __shfl_up_sync(0xFFFFFFFF, lb, 2, 32);
                        if (lane8 >= 2) {
                            la = la + ua_1;
                            lb = lb + ub8_1;
                        }
                        float ua_2 = __shfl_up_sync(0xFFFFFFFF, la, 4, 32);
                        float ub8_2 = __shfl_up_sync(0xFFFFFFFF, lb, 4, 32);
                        if (lane8 >= 4) {
                            la = la + ua_2;
                            lb = lb + ub8_2;
                        }
                        float ua_3 = __shfl_up_sync(0xFFFFFFFF, la, 8, 32);
                        float ub8_3 = __shfl_up_sync(0xFFFFFFFF, lb, 8, 32);
                        if (lane8 >= 8) {
                            la = la + ua_3;
                            lb = lb + ub8_3;
                        }
                        float ua_4 = __shfl_up_sync(0xFFFFFFFF, la, 16, 32);
                        float ub8_4 = __shfl_up_sync(0xFFFFFFFF, lb, 16, 32);
                        if (lane8 >= 16) {
                            la = la + ua_4;
                            lb = lb + ub8_4;
                        }
                        float _shfl_sync_2;
                        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xFFFFFFFF;" : "=f"(_shfl_sync_2) : "f"(la), "r"(31));
                        lb = lb + _shfl_sync_2;
                    }
                    int exp_row0_8 = start_tok_2 * num_hh + x8 * 64;
                    mbarrier_wait(gate_empty_cg0_addr + (g_cur8) * 8, _phase_gate_empty_cg0);
                    mbarrier_wait(gate_empty_cg1_addr + (g_cur8) * 8, _phase_gate_empty_cg0);
                    int gbase8 = g_cur8 * 64;
                    smem_ct[gbase8 + lane8] = la;
                    smem_ct[gbase8 + lane8 + 32] = lb;
                    __syncwarp();
                    float base8 = 0.0f;
                    if (cb8 > 0) {
                        base8 = smem_ct[gbase8 + (cb8 * 64 - 1) / num_hh];
                    }
                    int vr8 = vt8 * num_hh - cb8 * 64;
                    vr8 = ((vr8 > 64) ? 64 : ((vr8 < 0) ? 0 : vr8));
                    float cs63h8 = 0.0f;
                    cs63h8 = smem_ct[gbase8 + (cb8 * 64 + 63) / num_hh] - base8;
                    #pragma unroll
                    for (int hh = 0; hh < 2; hh++) {
                        int m8 = lane8 + hh * 32;
                        int tm8 = (cb8 * 64 + m8) / num_hh;
                        float bl8 = smem_ct[gbase8 + tm8];
                        float be8 = 0.0f;
                        if (m8 < vr8) {
                            be8 = beta[(exp_row0_8 + m8) * num_heads + head_2];
                        }
                        smem_blk[gbase8 + m8] = bl8;
                        smem_cs[gbase8 + m8] = bl8 - base8;
                        smem_beta[gbase8 + m8] = be8;
                        float cs_m8 = bl8 - base8;
                        float cs63_8 = cs63h8;
                        float ct_m8 = smem_ct[gbase8 + m8];
                        float _exp2_3 = approx_exp2(cs_m8);
                        smem_ecs[gbase8 + m8] = _exp2_3;
                        float _exp2_4 = approx_exp2(cs63_8 - cs_m8);
                        smem_edec[gbase8 + m8] = ((m8 < vr8) ? _exp2_4 : 0.0f);
                        float _exp2_5 = approx_exp2(ct_m8);
                        smem_ect[gbase8 + m8] = _exp2_5 * scale;
                    }
                    mbarrier_arrive(gate_full_addr + (g_cur8) * 8);
                    g_cur8 += 1;
                    if (g_cur8 == 4) { g_cur8 = 0; _phase_gate_empty_cg0 ^= 1; }
                }
                ord8 = ord8 + 1;
            }
        }
    }
    // ---- Role: tma_w ----
    if (warp == 9) {
        { // tma_w_main
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");
            unsigned int kq_cur9 = 0;
            unsigned int v_cur9 = 0;
            asm volatile("barrier.sync 3, 384;" ::: "memory");
            unsigned int _phase_kq_empty = 1;
            unsigned int _phase_v_empty = 1;
            #pragma unroll 1
            for (unsigned int it9 = ((smem_scan[0] == 0) ? (unsigned int)smem_scan[2] : (unsigned int)smem_scan[1]); it9 < (unsigned int)smem_scan[1]; it9++) {
                int w9 = 0;
                int xh_3 = 0;
                int seq_3 = items[(int)it9 * 4];
                int head_3 = items[(int)it9 * 4 + 1];
                int cut_3 = items[(int)it9 * 4 + 2];
                int span_3 = items[(int)it9 * 4 + 3];
                int rs_3 = cu_seqlens[seq_3];
                int re_3 = cu_seqlens[seq_3 + 1];
                int start_3 = ((rs_3 < 0) ? 0 : ((rs_3 > total_tokens) ? total_tokens : rs_3));
                int end_3 = ((re_3 < start_3) ? start_3 : ((re_3 > total_tokens) ? total_tokens : re_3));
                int tok0_3 = start_3 + cut_3 * 64;
                int tokend_3 = start_3 + (cut_3 + span_3) * 64;
                tokend_3 = ((end_3 < tokend_3) ? end_3 : tokend_3);
                int sp_at_3 = num_seqs * num_heads + (int)it9 * 2;
                int sp_lo_3 = (int)reinterpret_cast<unsigned int*>(split_ws)[sp_at_3];
                int sp_num_3 = (int)reinterpret_cast<unsigned int*>(split_ws)[sp_at_3 + 1];
                int sp_wend_3 = sp_lo_3 + sp_num_3;
                int start_tok_3 = start_3;
                tokend_3 = start_3 + (sp_wend_3 + num_hh - 1) / num_hh * 64;
                tokend_3 = ((end_3 < tokend_3) ? end_3 : tokend_3);
                int cstart_3 = sp_lo_3;
                int abs0_3 = cut_3;
                int nch_3 = ((sp_num_3 == 0) ? 0 : (tokend_3 - start_3 + 63) / 64 * num_hh - sp_lo_3);
                int wstart_3 = cut_3;
                int owns_end_3 = (((sp_num_3 > 0 || cut_3 == 0) && sp_wend_3 == 2 * (((end_3 - start_3) * num_hh + 127) / 128)) ? 1 : 0);
                int blk9c = cstart_3 / num_hh;
                int cb9c = cstart_3 - blk9c * num_hh;
                #pragma unroll 1
                for (int i9 = 0; i9 < nch_3; i9++) {
                    int x9 = cstart_3 + i9;
                    int blk9 = blk9c;
                    int exp_row9 = start_tok_3 * num_hh + x9 * 64;
                    int tok_blk9 = start_tok_3 + blk9 * 64;
                    int m9 = i9 & 1;
                    mbarrier_wait_hint(kq_empty_addr + (kq_cur9) * 8, _phase_kq_empty, 10000000);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kq_full_addr + (kq_cur9) * 8, 32768);
                        #pragma unroll
                        for (int kh = 0; kh < 2; kh++) {
                            unsigned int k_dst9 = smem_kq_addr + kq_cur9 * 32768 + (unsigned int)(kh * 16384 + m9 * 8192);
                            unsigned int q_dst9 = smem_kq_addr + kq_cur9 * 32768 + (unsigned int)(kh * 16384 + (1 - m9) * 8192);
                            tma_3d_gmem2smem(k_dst9, (&K), kh * 64, exp_row9, head_3, kq_full_addr + (kq_cur9) * 8);
                            tma_3d_gmem2smem(q_dst9, (&Q), kh * 64, tok_blk9, head_3, kq_full_addr + (kq_cur9) * 8);
                        }
                    }
                    kq_cur9 += 1;
                    if (kq_cur9 == 4) { kq_cur9 = 0; _phase_kq_empty ^= 1; }
                    mbarrier_wait_hint(v_empty_addr + (v_cur9) * 8, _phase_v_empty, 10000000);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(v_full_addr + (v_cur9) * 8, 8192);
                        tma_3d_gmem2smem(smem_v_addr + v_cur9 * 8192, (&V), 0, exp_row9, head_3, v_full_addr + (v_cur9) * 8);
                    }
                    v_cur9 += 1;
                    if (v_cur9 == 2) { v_cur9 = 0; _phase_v_empty ^= 1; }
                    cb9c = cb9c + 1;
                    if (cb9c == num_hh) {
                        cb9c = 0;
                        blk9c = blk9c + 1;
                    }
                }
            }
        }
    }
    // ---- Role: issuer ----
    if (warp == 10) {
        { // issuer_main
            unsigned int kq_cur10 = 0;
            unsigned int tinv_cur10 = 0;
            unsigned int a_cur10 = 0;
            unsigned int ad_cur10 = 0;
            unsigned int ar_cur10 = 0;
            asm volatile("barrier.sync 3, 384;" ::: "memory");
            unsigned int _phase_kq_full = 0;
            unsigned int _phase_acc_done = 1;
            unsigned int _phase_img_ready_0 = 0;
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_tinv_ready = 0;
            unsigned int _phase_y_ready_0 = 0;
            unsigned int _phase_du_ready_0 = 0;
            unsigned int _phase_a_ready = 0;
            unsigned int _phase_uimg_ready_0 = 0;
            unsigned int _phase_qs_scaled_0 = 0;
            #pragma unroll 1
            for (unsigned int it10 = ((smem_scan[0] == 0) ? (unsigned int)smem_scan[2] : (unsigned int)smem_scan[1]); it10 < (unsigned int)smem_scan[1]; it10++) {
                int w10 = 0;
                int xh_4 = 0;
                int seq_4 = items[(int)it10 * 4];
                int head_4 = items[(int)it10 * 4 + 1];
                int cut_4 = items[(int)it10 * 4 + 2];
                int span_4 = items[(int)it10 * 4 + 3];
                int rs_4 = cu_seqlens[seq_4];
                int re_4 = cu_seqlens[seq_4 + 1];
                int start_4 = ((rs_4 < 0) ? 0 : ((rs_4 > total_tokens) ? total_tokens : rs_4));
                int end_4 = ((re_4 < start_4) ? start_4 : ((re_4 > total_tokens) ? total_tokens : re_4));
                int tok0_4 = start_4 + cut_4 * 64;
                int tokend_4 = start_4 + (cut_4 + span_4) * 64;
                tokend_4 = ((end_4 < tokend_4) ? end_4 : tokend_4);
                int sp_at_4 = num_seqs * num_heads + (int)it10 * 2;
                int sp_lo_4 = (int)reinterpret_cast<unsigned int*>(split_ws)[sp_at_4];
                int sp_num_4 = (int)reinterpret_cast<unsigned int*>(split_ws)[sp_at_4 + 1];
                int sp_wend_4 = sp_lo_4 + sp_num_4;
                int start_tok_4 = start_4;
                tokend_4 = start_4 + (sp_wend_4 + num_hh - 1) / num_hh * 64;
                tokend_4 = ((end_4 < tokend_4) ? end_4 : tokend_4);
                int cstart_4 = sp_lo_4;
                int abs0_4 = cut_4;
                int nch_4 = ((sp_num_4 == 0) ? 0 : (tokend_4 - start_4 + 63) / 64 * num_hh - sp_lo_4);
                int wstart_4 = cut_4;
                int owns_end_4 = (((sp_num_4 > 0 || cut_4 == 0) && sp_wend_4 == 2 * (((end_4 - start_4) * num_hh + 127) / 128)) ? 1 : 0);
                int cb10c = cstart_4 - cstart_4 / num_hh * num_hh;
                if (0 < nch_4) {
                    mbarrier_wait(kq_full_addr + (kq_cur10) * 8, _phase_kq_full);
                    mbarrier_wait(acc_done_addr + (ad_cur10) * 8, _phase_acc_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int a_lo = make_warp_uniform((((smem_kq_addr) >> 4) & 0x3FFF) + (kq_cur10) * 2048);
                    int bk_lo = make_warp_uniform((((smem_kq_addr) >> 4) & 0x3FFF) + (kq_cur10) * 2048);
                    uint32_t _elected_0 = elect_sync();
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135267472;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(a_lo), "r"(bk_lo), "r"(tmem_acc), "r"(0), "r"(_elected_0) : "memory");
                    tcgen05_commit_leader(acc_ready_addr + (ar_cur10) * 8, _elected_0);
                    ad_cur10 += 1;
                    if (ad_cur10 == 2) { ad_cur10 = 0; _phase_acc_done ^= 1; }
                    ar_cur10 += 1;
                    if (ar_cur10 == 2) { ar_cur10 = 0; }
                }
                if (1 < nch_4) {
                    mbarrier_wait(kq_full_addr + (((kq_cur10 + 1 < 4) ? kq_cur10 + 1 : (unsigned int)0)) * 8, ((kq_cur10 + 1 < 4) ? _phase_kq_full : _phase_kq_full ^ 1));
                    mbarrier_wait(acc_done_addr + (ad_cur10) * 8, _phase_acc_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int a_lo_1 = make_warp_uniform((((smem_kq_addr) >> 4) & 0x3FFF) + (((kq_cur10 + 1 < 4) ? kq_cur10 + 1 : (unsigned int)0)) * 2048);
                    int bk_lo_1 = make_warp_uniform((((smem_kq_addr + 8192) >> 4) & 0x3FFF) + (((kq_cur10 + 1 < 4) ? kq_cur10 + 1 : (unsigned int)0)) * 2048);
                    uint32_t _elected_1 = elect_sync();
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135267472;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(a_lo_1), "r"(bk_lo_1), "r"((tmem_acc + (64))), "r"(0), "r"(_elected_1) : "memory");
                    tcgen05_commit_leader(acc_ready_addr + (ar_cur10) * 8, _elected_1);
                    ad_cur10 += 1;
                    if (ad_cur10 == 2) { ad_cur10 = 0; _phase_acc_done ^= 1; }
                    ar_cur10 += 1;
                    if (ar_cur10 == 2) { ar_cur10 = 0; }
                }
                #pragma unroll 1
                for (int c10 = 0; c10 < nch_4; c10++) {
                    int cb10 = cb10c;
                    int m10 = c10 & 1;
                    int is_head10 = ((c10 == 0 || cb10 == 0) ? 1 : 0);
                    int last10 = ((cb10 == num_hh - 1 || c10 == nch_4 - 1) ? 1 : 0);
                    if (m10 != 0 && c10 + 2 < nch_4) {
                        mbarrier_wait(kq_full_addr + (((kq_cur10 + 2 >= 4) ? kq_cur10 + 2 - 4 : kq_cur10 + 2)) * 8, _phase_kq_full ^ (unsigned int)(kq_cur10 + 2 >= 4));
                        mbarrier_wait(acc_done_addr + (ad_cur10) * 8, _phase_acc_done);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int a_lo_2 = make_warp_uniform((((smem_kq_addr) >> 4) & 0x3FFF) + (((kq_cur10 + 2 >= 4) ? kq_cur10 + 2 - 4 : kq_cur10 + 2)) * 2048);
                        int bk_lo_2 = make_warp_uniform((((smem_kq_addr + 8192) >> 4) & 0x3FFF) + (((kq_cur10 + 2 >= 4) ? kq_cur10 + 2 - 4 : kq_cur10 + 2)) * 2048);
                        uint32_t _elected_2 = elect_sync();
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135267472;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(a_lo_2), "r"(bk_lo_2), "r"((tmem_acc + (64))), "r"(0), "r"(_elected_2) : "memory");
                        tcgen05_commit_leader(acc_ready_addr + (ar_cur10) * 8, _elected_2);
                        ad_cur10 += 1;
                        if (ad_cur10 == 2) { ad_cur10 = 0; _phase_acc_done ^= 1; }
                        ar_cur10 += 1;
                        if (ar_cur10 == 2) { ar_cur10 = 0; }
                    }
                    mbarrier_wait(img_ready_addr, _phase_img_ready_0);
                    _phase_img_ready_0 ^= 1;
                    mbarrier_wait(kq_full_addr + (kq_cur10) * 8, _phase_kq_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int k_lo10e = make_warp_uniform((((smem_kq_addr) >> 4) & 0x3FFF) + (kq_cur10) * 2048);
                    int k_lo10o = make_warp_uniform((((smem_kq_addr + 8192) >> 4) & 0x3FFF) + (kq_cur10) * 2048);
                    int kq_s10 = ((m10 != 0) ? kq_cur10 * 4 + 1 : kq_cur10 * 4);
                    int kq_t10 = ((m10 != 0) ? kq_cur10 * 4 : kq_cur10 * 4 + 1);
                    int k_lo10s = make_warp_uniform((((smem_kq_h_addr) >> 4) & 0x3FFF) + (kq_s10) * 512);
                    int q_lo10s = make_warp_uniform((((smem_kq_h_addr) >> 4) & 0x3FFF) + (kq_t10) * 512);
                    uint32_t _elected_3 = elect_sync();
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_ks), "r"(k_lo10s), "r"(tmem_img), "r"(0), "r"(_elected_3) : "memory");
                    tcgen05_commit_leader(ks_full_addr, _elected_3);
                    if (is_head10 != 0) {
                        mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                        _phase_o_empty_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_4 = elect_sync();
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o), "r"(q_lo10s), "r"(tmem_img), "r"(0), "r"(_elected_4) : "memory");
                        tcgen05_commit_leader(qs_full_addr, _elected_4);
                    }
                    mbarrier_wait(tinv_ready_addr + (tinv_cur10) * 8, _phase_tinv_ready);
                    mbarrier_wait(y_ready_addr, _phase_y_ready_0);
                    _phase_y_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int t_lo10 = make_warp_uniform((((smem_tinv_addr) >> 4) & 0x3FFF) + (tinv_cur10) * 512);
                    uint32_t _elected_5 = elect_sync();
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_ks), "r"(t_lo10), "r"(tmem_y), "r"(0), "r"(_elected_5) : "memory");
                    tcgen05_commit_leader(u_full_addr, _elected_5);
                    tcgen05_commit_leader(tinv_empty_addr + (tinv_cur10) * 8, _elected_5);
                    tinv_cur10 += 1;
                    if (tinv_cur10 == 3) { tinv_cur10 = 0; _phase_tinv_ready ^= 1; }
                    if (m10 == 0 && c10 + 2 < nch_4) {
                        mbarrier_wait(kq_full_addr + (((kq_cur10 + 2 >= 4) ? kq_cur10 + 2 - 4 : kq_cur10 + 2)) * 8, _phase_kq_full ^ (unsigned int)(kq_cur10 + 2 >= 4));
                        mbarrier_wait(acc_done_addr + (ad_cur10) * 8, _phase_acc_done);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int a_lo_3 = make_warp_uniform((((smem_kq_addr) >> 4) & 0x3FFF) + (((kq_cur10 + 2 >= 4) ? kq_cur10 + 2 - 4 : kq_cur10 + 2)) * 2048);
                        int bk_lo_3 = make_warp_uniform((((smem_kq_addr) >> 4) & 0x3FFF) + (((kq_cur10 + 2 >= 4) ? kq_cur10 + 2 - 4 : kq_cur10 + 2)) * 2048);
                        uint32_t _elected_6 = elect_sync();
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135267472;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(a_lo_3), "r"(bk_lo_3), "r"(tmem_acc), "r"(0), "r"(_elected_6) : "memory");
                        tcgen05_commit_leader(acc_ready_addr + (ar_cur10) * 8, _elected_6);
                        ad_cur10 += 1;
                        if (ad_cur10 == 2) { ad_cur10 = 0; _phase_acc_done ^= 1; }
                        ar_cur10 += 1;
                        if (ar_cur10 == 2) { ar_cur10 = 0; }
                    }
                    if (last10 == 0) {
                        mbarrier_wait(du_ready_addr, _phase_du_ready_0);
                        _phase_du_ready_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int kmn_s10a = ((m10 != 0) ? kq_cur10 * 4 + 1 : kq_cur10 * 4);
                        int kmn_lo10a = make_warp_uniform(((((smem_kq_mn_h_addr) >> 4) & 0x3FFF) | 0x4000000) + (kmn_s10a) * 512);
                        uint32_t _elected_7 = elect_sync();
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
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
                    :: "r"(tmem_state), "r"(kmn_lo10a), "r"(tmem_du), "r"(1), "r"(_elected_7) : "memory");
                        tcgen05_commit_leader(state_full_addr, _elected_7);
                        tcgen05_commit_leader(kq_empty_addr + (kq_cur10) * 8, _elected_7);
                        kq_cur10 += 1;
                        if (kq_cur10 == 4) { kq_cur10 = 0; _phase_kq_full ^= 1; }
                    }
                    mbarrier_wait(a_ready_addr + (a_cur10) * 8, _phase_a_ready);
                    mbarrier_wait(uimg_ready_addr, _phase_uimg_ready_0);
                    _phase_uimg_ready_0 ^= 1;
                    if (is_head10 != 0) {
                        mbarrier_wait(qs_scaled_addr, _phase_qs_scaled_0);
                        _phase_qs_scaled_0 ^= 1;
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int a_lo10o = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (a_cur10) * 512);
                    uint32_t _elected_8 = elect_sync();
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o), "r"(a_lo10o), "r"(tmem_y), "r"(1), "r"(_elected_8) : "memory");
                    tcgen05_commit_leader(a_empty_addr + (a_cur10) * 8, _elected_8);
                    a_cur10 += 1;
                    if (a_cur10 == 3) { a_cur10 = 0; _phase_a_ready ^= 1; }
                    if (last10 != 0) {
                        elect_commit(o_full_addr);
                    }
                    if (last10 != 0) {
                        mbarrier_wait(du_ready_addr, _phase_du_ready_0);
                        _phase_du_ready_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int kmn_s10b = ((m10 != 0) ? kq_cur10 * 4 + 1 : kq_cur10 * 4);
                        int kmn_lo10b = make_warp_uniform(((((smem_kq_mn_h_addr) >> 4) & 0x3FFF) | 0x4000000) + (kmn_s10b) * 512);
                        uint32_t _elected_9 = elect_sync();
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
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
                    :: "r"(tmem_state), "r"(kmn_lo10b), "r"(tmem_du), "r"(1), "r"(_elected_9) : "memory");
                        tcgen05_commit_leader(state_full_addr, _elected_9);
                        tcgen05_commit_leader(kq_empty_addr + (kq_cur10) * 8, _elected_9);
                        kq_cur10 += 1;
                        if (kq_cur10 == 4) { kq_cur10 = 0; _phase_kq_full ^= 1; }
                    }
                    cb10c = ((cb10c == num_hh - 1) ? 0 : cb10c + 1);
                }
                if ((nch_4 & 1) != 0) {
                    mbarrier_wait(acc_done_addr + (ad_cur10) * 8, _phase_acc_done);
                    elect_commit(acc_ready_addr + (ar_cur10) * 8);
                    ad_cur10 += 1;
                    if (ad_cur10 == 2) { ad_cur10 = 0; _phase_acc_done ^= 1; }
                    ar_cur10 += 1;
                    if (ar_cur10 == 2) { ar_cur10 = 0; }
                }
            }
        }
    }
    // ---- Role: store_w ----
    if (warp == 11) {
        { // store_w_main
            int lane11 = lane;
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Omap))) : "memory");
            int gq_at = num_items * 4;
            gq_at = gq_at + (int)num_bids + 1;
            int gq_bad = 0;
            if (lane11 == 0) {
                if (items[gq_at] != num_seqs) {
                    gq_bad = 1;
                }
            }
            #pragma unroll 1
            for (int gq_i = lane11; gq_i < num_seqs + 1; gq_i += 32) {
                if (cu_seqlens[gq_i] != items[gq_at + 1 + gq_i]) {
                    gq_bad = 1;
                }
            }
            int _warp_vote_0 = __any_sync(0xFFFFFFFF, gq_bad != 0);
            if (_warp_vote_0 != 0) {
                if (lane11 == 0) {
                    smem_scan[0] = 1;
                }
            }
            asm volatile("barrier.sync 3, 384;" ::: "memory");
            unsigned int _phase_o_stage_ready_0 = 0;
            #pragma unroll 1
            for (unsigned int it11 = ((smem_scan[0] == 0) ? (unsigned int)smem_scan[2] : (unsigned int)smem_scan[1]); it11 < (unsigned int)smem_scan[1]; it11++) {
                int w11 = 0;
                int xh_5 = 0;
                int seq_5 = items[(int)it11 * 4];
                int head_5 = items[(int)it11 * 4 + 1];
                int cut_5 = items[(int)it11 * 4 + 2];
                int span_5 = items[(int)it11 * 4 + 3];
                int rs_5 = cu_seqlens[seq_5];
                int re_5 = cu_seqlens[seq_5 + 1];
                int start_5 = ((rs_5 < 0) ? 0 : ((rs_5 > total_tokens) ? total_tokens : rs_5));
                int end_5 = ((re_5 < start_5) ? start_5 : ((re_5 > total_tokens) ? total_tokens : re_5));
                int tok0_5 = start_5 + cut_5 * 64;
                int tokend_5 = start_5 + (cut_5 + span_5) * 64;
                tokend_5 = ((end_5 < tokend_5) ? end_5 : tokend_5);
                int sp_at_5 = num_seqs * num_heads + (int)it11 * 2;
                int sp_lo_5 = (int)reinterpret_cast<unsigned int*>(split_ws)[sp_at_5];
                int sp_num_5 = (int)reinterpret_cast<unsigned int*>(split_ws)[sp_at_5 + 1];
                int sp_wend_5 = sp_lo_5 + sp_num_5;
                int start_tok_5 = start_5;
                tokend_5 = start_5 + (sp_wend_5 + num_hh - 1) / num_hh * 64;
                tokend_5 = ((end_5 < tokend_5) ? end_5 : tokend_5);
                int cstart_5 = sp_lo_5;
                int abs0_5 = cut_5;
                int nch_5 = ((sp_num_5 == 0) ? 0 : (tokend_5 - start_5 + 63) / 64 * num_hh - sp_lo_5);
                int wstart_5 = cut_5;
                int owns_end_5 = (((sp_num_5 > 0 || cut_5 == 0) && sp_wend_5 == 2 * (((end_5 - start_5) * num_hh + 127) / 128)) ? 1 : 0);
                nch_5 = ((nch_5 > 0) ? nch_5 - (wstart_5 - cstart_5) : 0);
                cstart_5 = wstart_5;
                if (elect_sync()) {
                    {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>((&Omap));
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tmap_ws + (long long)bid * 128ll));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tmap_ws + (long long)bid * 128ll)), "r"((uint32_t)((unsigned int)tokend_5)) : "memory");
                    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
                }
                __syncwarp();
                #pragma unroll 1
                for (int i11 = 0; i11 < nch_5; i11++) {
                    int x11 = cstart_5 + i11;
                    int blk11 = x11 / num_hh;
                    int cb11 = x11 - blk11 * num_hh;
                    int last_in_blk11 = ((cb11 == num_hh - 1 || i11 == nch_5 - 1) ? 1 : 0);
                    int write11 = ((blk11 * num_hh >= wstart_5) ? 1 : 0);
                    if (last_in_blk11 != 0 && write11 != 0) {
                        int tok0_blk11 = start_tok_5 + blk11 * 64;
                        int vt11 = tokend_5 - tok0_blk11;
                        vt11 = ((vt11 > 64) ? 64 : ((vt11 < 0) ? 0 : vt11));
                        mbarrier_wait(o_stage_ready_addr, _phase_o_stage_ready_0);
                        _phase_o_stage_ready_0 ^= 1;
                        if (elect_sync()) {
                            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tmap_ws + (long long)bid * 128ll)) : "memory");
                            tma_store_3d(tmap_ws + (long long)bid * 128ll, 0, tok0_blk11, head_5, smem_o_addr);
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        __syncwarp();
                        mbarrier_arrive(o_stage_empty_addr);
                    }
                }
            }
            if (bid == 0 && lane11 == 0) {
                if (smem_scan[0] != 0) {
                    *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(O)) + (0)) = 2143322048;
                }
            }
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
