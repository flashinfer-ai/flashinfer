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
#define NUM_K_PIPE_STAGES 3
#define NUM_Q_PIPE_STAGES 2
#define NUM_V_PIPE_STAGES 2
#define NUM_GATE_PIPE_STAGES 5
#define NUM_AB_PIPE_STAGES 3
#define NUM_QK_PIPE_STAGES 3
#define NUM_O_PIPE_STAGES 1
#define SMEM_SMEM_K_OFF 1024
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_SMEM_K_MN_OFF 1024
#define SMEM_SMEM_K_MN_STAGE_BYTES 16384
#define SMEM_SMEM_K_MN_STRIDE 16384
#define SMEM_SMEM_Q_OFF 50176
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_V_OFF 82944
#define SMEM_SMEM_V_STAGE_BYTES 8192
#define SMEM_SMEM_V_STRIDE 8192
#define SMEM_SMEM_AB_OFF 99328
#define SMEM_SMEM_AB_STAGE_BYTES 8192
#define SMEM_SMEM_AB_STRIDE 8192
#define SMEM_SMEM_ABO_OFF 123904
#define SMEM_SMEM_ABO_STAGE_BYTES 8192
#define SMEM_SMEM_ABO_STRIDE 8192
#define SMEM_SMEM_QK_OFF 148480
#define SMEM_SMEM_QK_STAGE_BYTES 8192
#define SMEM_SMEM_QK_STRIDE 8192
#define SMEM_SMEM_O_OFF 173056
#define SMEM_SMEM_O_STAGE_BYTES 8192
#define SMEM_SMEM_O_STRIDE 8192
#define SMEM_SMEM_PRIME_OFF 181248
#define SMEM_SMEM_PRIME_STAGE_BYTES 8192
#define SMEM_SMEM_PRIME_STRIDE 8192
#define SMEM_SMEM_GATE_OFF 189440
#define SMEM_SMEM_GATE_STAGE_BYTES 1536
#define SMEM_SMEM_GATE_STRIDE 1536
#define SMEM_SMEM_CT_OFF 197120
#define SMEM_SMEM_CT_STAGE_BYTES 256
#define SMEM_SMEM_CT_STRIDE 256
#define SMEM_TOTAL 197376
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


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdp_0de8fd0ee57885709a1c(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap O, float* __restrict__ gate, float* __restrict__ beta, int* __restrict__ cu_seqlens, int* __restrict__ state_indices, int* __restrict__ output_slot_indices, float* __restrict__ initial_state, float* __restrict__ output_state, uint8_t* __restrict__ tensormap_workspace, long long initial_state_stride_slot, long long output_state_stride_slot, float scale, int num_seqs, int total_tokens, int num_heads, int num_householder, int total_tiles, int metadata_i64_mask, uint8_t* __restrict__ ainv_workspace, long long ainv_chunk_stride, long long initial_state_size_p0, long long output_state_size_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define load_k_full_addr (mbar_base + 0)
    #define load_k_empty_addr (mbar_base + 24)
    #define load_q_full_addr (mbar_base + 48)
    #define q_empty_addr (mbar_base + 64)
    #define load_v_full_addr (mbar_base + 80)
    #define v_empty_addr (mbar_base + 96)
    #define alpha_full_addr (mbar_base + 112)
    #define beta_full_addr (mbar_base + 152)
    #define gate_empty_addr (mbar_base + 192)
    #define ab_ready_addr (mbar_base + 232)
    #define ab_empty_addr (mbar_base + 256)
    #define qk_ready_addr (mbar_base + 280)
    #define qk_empty_addr (mbar_base + 304)
    #define o_ready_addr (mbar_base + 328)
    #define o_empty_addr (mbar_base + 336)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_k_addr = smem + 1024;
    __nv_bfloat16* smem_k_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_k_mn_addr = smem + 1024;
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int smem_q_addr = smem + 50176;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int smem_v_addr = smem + 82944;
    __nv_bfloat16* smem_ab = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int smem_ab_addr = smem + 99328;
    __nv_bfloat16* smem_abo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 123904);
    const int smem_abo_addr = smem + 123904;
    __nv_bfloat16* smem_qk = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int smem_qk_addr = smem + 148480;
    __nv_bfloat16* smem_o = reinterpret_cast<__nv_bfloat16*>(smem_raw + 173056);
    const int smem_o_addr = smem + 173056;
    __nv_bfloat16* smem_prime = reinterpret_cast<__nv_bfloat16*>(smem_raw + 181248);
    const int smem_prime_addr = smem + 181248;
    float* smem_gate = reinterpret_cast<float*>(smem_raw + 189440);
    const int smem_gate_addr = smem + 189440;
    float* smem_ct = reinterpret_cast<float*>(smem_raw + 197120);
    const int __unused = smem + 197120;

    // Mbarrier init (15 pipeline groups, 0 ordered-sequence groups, 43 barriers)
    // Mbarriers at smem_raw[0..344)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // load_k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // load_k_empty: 3 barriers, init_count=256
            mbarrier_init(smem + 24, 256);
            mbarrier_init(smem + 32, 256);
            mbarrier_init(smem + 40, 256);
            // --- pipeline 'q_pipe' ---
            // load_q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // q_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 64, 256);
            mbarrier_init(smem + 72, 256);
            // --- pipeline 'v_pipe' ---
            // load_v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // v_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 96, 128);
            mbarrier_init(smem + 104, 128);
            // --- pipeline 'gate_pipe' ---
            // alpha_full: 5 barriers, init_count=32
            mbarrier_init(smem + 112, 32);
            mbarrier_init(smem + 120, 32);
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            // beta_full: 5 barriers, init_count=32
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            // gate_empty: 5 barriers, init_count=256
            mbarrier_init(smem + 192, 256);
            mbarrier_init(smem + 200, 256);
            mbarrier_init(smem + 208, 256);
            mbarrier_init(smem + 216, 256);
            mbarrier_init(smem + 224, 256);
            // --- pipeline 'ab_pipe' ---
            // ab_ready: 3 barriers, init_count=128
            mbarrier_init(smem + 232, 128);
            mbarrier_init(smem + 240, 128);
            mbarrier_init(smem + 248, 128);
            // ab_empty: 3 barriers, init_count=128
            mbarrier_init(smem + 256, 128);
            mbarrier_init(smem + 264, 128);
            mbarrier_init(smem + 272, 128);
            // --- pipeline 'qk_pipe' ---
            // qk_ready: 3 barriers, init_count=128
            mbarrier_init(smem + 280, 128);
            mbarrier_init(smem + 288, 128);
            mbarrier_init(smem + 296, 128);
            // qk_empty: 3 barriers, init_count=128
            mbarrier_init(smem + 304, 128);
            mbarrier_init(smem + 312, 128);
            mbarrier_init(smem + 320, 128);
            // --- pipeline 'o_pipe' ---
            // o_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 328, 128);
            // o_empty: 1 barriers, init_count=32
            mbarrier_init(smem + 336, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // Kernel post-init ops
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
    }

    // ---- Role: state_wg ----
    if (warp <= 3) {
        { // state_wg_main
            unsigned int rs_magic = (unsigned int)((1048576 + num_householder - 1) / num_householder);
            asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
            float scale_eff = ((scale != 0.0f) ? scale : 0.088388346f);
            unsigned int k_cur = 0;
            unsigned int q_cur = 0;
            unsigned int v_cur = 0;
            unsigned int g_cur = 0;
            unsigned int ab_cur = 0;
            unsigned int qk_cur = 0;
            float state[64] = {0};
            float q_state[32] = {0};
            float ks[32] = {0};
            int warp_id_in_role = warp - 0;
            int pr_t = warp_id_in_role * 32 + lane;
            unsigned int pr_zero[4];
            pr_zero[0] = 0;
            pr_zero[1] = 0;
            pr_zero[2] = 0;
            pr_zero[3] = 0;
            #pragma unroll
            for (int pr_i = 0; pr_i < 4; pr_i++) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_prime_addr + (unsigned int)((pr_i * 128 + pr_t) * 16)), "r"(pr_zero[0]), "r"(pr_zero[1]), "r"(pr_zero[2]), "r"(pr_zero[3]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 2, 128;" ::: "memory");
            unsigned int prime_a[16];
            prime_a[0] = 0;
            prime_a[1] = 0;
            prime_a[2] = 0;
            prime_a[3] = 0;
            prime_a[4] = 0;
            prime_a[5] = 0;
            prime_a[6] = 0;
            prime_a[7] = 0;
            prime_a[8] = 0;
            prime_a[9] = 0;
            prime_a[10] = 0;
            prime_a[11] = 0;
            prime_a[12] = 0;
            prime_a[13] = 0;
            prime_a[14] = 0;
            prime_a[15] = 0;
            int pz_t = warp_id_in_role * 32 + lane;
            int pz_row = pz_t / 4;
            int pz_col = 32 + (pz_t & 3) * 8;
            unsigned int pz_zero[4];
            pz_zero[0] = 0;
            pz_zero[1] = 0;
            pz_zero[2] = 0;
            pz_zero[3] = 0;
            #pragma unroll
            for (int pz_st = 0; pz_st < 3; pz_st++) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_abo_addr + (unsigned int)(pz_st * 8192) + (unsigned int)(pz_row * 128 + pz_col * 2 ^ (pz_row * 128 + pz_col * 2 >> 7 & 7) << 4)), "r"(pz_zero[0]), "r"(pz_zero[1]), "r"(pz_zero[2]), "r"(pz_zero[3]) : "memory");
            }
            int cg = 0;
            unsigned int _phase_alpha_full = 0;
            unsigned int _phase_load_k_full = 0;
            unsigned int _phase_o_empty = 1;
            unsigned int _phase_load_q_full = 0;
            unsigned int _phase_load_v_full = 0;
            unsigned int _phase_ab_ready = 0;
            unsigned int _phase_qk_ready = 0;
            #pragma unroll 1
            for (unsigned int tile = bid; tile < total_tiles; tile += num_bids) {
                int tile_i = (int)tile;
                int batch_idx = tile_i / num_heads;
                int head_idx = tile_i - batch_idx * num_heads;
                int raw_start = cu_seqlens[batch_idx];
                int raw_end = cu_seqlens[batch_idx + 1];
                int start_floor = ((raw_start > 0) ? raw_start : 0);
                int batch_start = ((start_floor < total_tokens) ? start_floor : total_tokens);
                int end_floor = ((raw_end > batch_start) ? raw_end : batch_start);
                int batch_end = ((end_floor < total_tokens) ? end_floor : total_tokens);
                int exp_len = (batch_end - batch_start) * num_householder;
                int num_chunks = (exp_len + 63) / 64;
                long long output_head_base = (long long)batch_idx * output_state_stride_slot + (long long)head_idx * 8192;
                state[0] = 0.0f;
                state[1] = 0.0f;
                state[2] = 0.0f;
                state[3] = 0.0f;
                state[4] = 0.0f;
                state[5] = 0.0f;
                state[6] = 0.0f;
                state[7] = 0.0f;
                state[8] = 0.0f;
                state[9] = 0.0f;
                state[10] = 0.0f;
                state[11] = 0.0f;
                state[12] = 0.0f;
                state[13] = 0.0f;
                state[14] = 0.0f;
                state[15] = 0.0f;
                state[16] = 0.0f;
                state[17] = 0.0f;
                state[18] = 0.0f;
                state[19] = 0.0f;
                state[20] = 0.0f;
                state[21] = 0.0f;
                state[22] = 0.0f;
                state[23] = 0.0f;
                state[24] = 0.0f;
                state[25] = 0.0f;
                state[26] = 0.0f;
                state[27] = 0.0f;
                state[28] = 0.0f;
                state[29] = 0.0f;
                state[30] = 0.0f;
                state[31] = 0.0f;
                state[32] = 0.0f;
                state[33] = 0.0f;
                state[34] = 0.0f;
                state[35] = 0.0f;
                state[36] = 0.0f;
                state[37] = 0.0f;
                state[38] = 0.0f;
                state[39] = 0.0f;
                state[40] = 0.0f;
                state[41] = 0.0f;
                state[42] = 0.0f;
                state[43] = 0.0f;
                state[44] = 0.0f;
                state[45] = 0.0f;
                state[46] = 0.0f;
                state[47] = 0.0f;
                state[48] = 0.0f;
                state[49] = 0.0f;
                state[50] = 0.0f;
                state[51] = 0.0f;
                state[52] = 0.0f;
                state[53] = 0.0f;
                state[54] = 0.0f;
                state[55] = 0.0f;
                state[56] = 0.0f;
                state[57] = 0.0f;
                state[58] = 0.0f;
                state[59] = 0.0f;
                state[60] = 0.0f;
                state[61] = 0.0f;
                state[62] = 0.0f;
                state[63] = 0.0f;
                asm volatile("" : "+f"(q_state[0]) :: "memory");
                asm volatile("" : "+f"(q_state[1]) :: "memory");
                asm volatile("" : "+f"(q_state[2]) :: "memory");
                asm volatile("" : "+f"(q_state[3]) :: "memory");
                asm volatile("" : "+f"(q_state[4]) :: "memory");
                asm volatile("" : "+f"(q_state[5]) :: "memory");
                asm volatile("" : "+f"(q_state[6]) :: "memory");
                asm volatile("" : "+f"(q_state[7]) :: "memory");
                asm volatile("" : "+f"(q_state[8]) :: "memory");
                asm volatile("" : "+f"(q_state[9]) :: "memory");
                asm volatile("" : "+f"(q_state[10]) :: "memory");
                asm volatile("" : "+f"(q_state[11]) :: "memory");
                asm volatile("" : "+f"(q_state[12]) :: "memory");
                asm volatile("" : "+f"(q_state[13]) :: "memory");
                asm volatile("" : "+f"(q_state[14]) :: "memory");
                asm volatile("" : "+f"(q_state[15]) :: "memory");
                asm volatile("" : "+f"(q_state[16]) :: "memory");
                asm volatile("" : "+f"(q_state[17]) :: "memory");
                asm volatile("" : "+f"(q_state[18]) :: "memory");
                asm volatile("" : "+f"(q_state[19]) :: "memory");
                asm volatile("" : "+f"(q_state[20]) :: "memory");
                asm volatile("" : "+f"(q_state[21]) :: "memory");
                asm volatile("" : "+f"(q_state[22]) :: "memory");
                asm volatile("" : "+f"(q_state[23]) :: "memory");
                asm volatile("" : "+f"(q_state[24]) :: "memory");
                asm volatile("" : "+f"(q_state[25]) :: "memory");
                asm volatile("" : "+f"(q_state[26]) :: "memory");
                asm volatile("" : "+f"(q_state[27]) :: "memory");
                asm volatile("" : "+f"(q_state[28]) :: "memory");
                asm volatile("" : "+f"(q_state[29]) :: "memory");
                asm volatile("" : "+f"(q_state[30]) :: "memory");
                asm volatile("" : "+f"(q_state[31]) :: "memory");
                asm volatile("" : "+r"(prime_a[0]) :: "memory");
                asm volatile("" : "+r"(prime_a[1]) :: "memory");
                asm volatile("" : "+r"(prime_a[2]) :: "memory");
                asm volatile("" : "+r"(prime_a[3]) :: "memory");
                asm volatile("" : "+r"(prime_a[4]) :: "memory");
                asm volatile("" : "+r"(prime_a[5]) :: "memory");
                asm volatile("" : "+r"(prime_a[6]) :: "memory");
                asm volatile("" : "+r"(prime_a[7]) :: "memory");
                asm volatile("" : "+r"(prime_a[8]) :: "memory");
                asm volatile("" : "+r"(prime_a[9]) :: "memory");
                asm volatile("" : "+r"(prime_a[10]) :: "memory");
                asm volatile("" : "+r"(prime_a[11]) :: "memory");
                asm volatile("" : "+r"(prime_a[12]) :: "memory");
                asm volatile("" : "+r"(prime_a[13]) :: "memory");
                asm volatile("" : "+r"(prime_a[14]) :: "memory");
                asm volatile("" : "+r"(prime_a[15]) :: "memory");
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                uint64_t _wgmma_desc_0 = (((uint64_t)(((smem_prime_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t prime_desc = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n}\n"
                    : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                    : "r"(prime_a[0]), "r"(prime_a[1]), "r"(prime_a[2]), "r"(prime_a[3]), "l"(prime_desc)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                    : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                    : "r"(prime_a[4]), "r"(prime_a[(4) + 1]), "r"(prime_a[(4) + 2]), "r"(prime_a[(4) + 3]), "l"(prime_desc + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                    : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                    : "r"(prime_a[8]), "r"(prime_a[(8) + 1]), "r"(prime_a[(8) + 2]), "r"(prime_a[(8) + 3]), "l"(prime_desc + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                    : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                    : "r"(prime_a[12]), "r"(prime_a[(12) + 1]), "r"(prime_a[(12) + 2]), "r"(prime_a[(12) + 3]), "l"(prime_desc + 6)
                    : "memory");
                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                #pragma unroll 1
                for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                    int s_warp = warp_id_in_role;
                    int s_quad = lane & 3;
                    int s_gb = (int)g_cur * 384;
                    int s_mtx = lane / 8;
                    int s_row_in_mtx = lane & 7;
                    int s_ldm_i_off = lane >> 4 & 1;
                    int s_ldm_v0 = s_warp * 16 + (lane >> 3 & 1) * 8;
                    uint32_t smid_s;
                    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid_s));
                    int bp_blk = (int)((unsigned int)chunk_idx * rs_magic / 1048576);
                    int bp_i_in = chunk_idx - bp_blk * num_householder;
                    int bp_roff = bp_i_in * 64;
                    int bp_blk_t0 = batch_start + bp_blk * 64;
                    int bp_rem_tok = batch_end - bp_blk_t0;
                    int bp_tib = ((bp_rem_tok < 64) ? bp_rem_tok : 64);
                    int bp_is_head = ((bp_i_in == 0) ? 1 : 0);
                    int bp_is_tail = ((bp_i_in == num_householder - 1) ? 1 : ((chunk_idx == num_chunks - 1) ? 1 : 0));
                    int cp_blk = (int)((unsigned int)chunk_idx * rs_magic / 1048576);
                    int cp_i_in = chunk_idx - cp_blk * num_householder;
                    int cp_sub = (int)((unsigned int)(cp_i_in * 64) * rs_magic / 1048576);
                    int cp_tok_off = cp_blk * 64 + cp_sub;
                    int cp_rem = cp_i_in * 64 - cp_sub * num_householder;
                    int cp_t0 = batch_start + cp_tok_off;
                    int cp_b = num_householder - 1 - cp_rem;
                    int cp_cnt = (int)((unsigned int)(63 - cp_b) * rs_magic / 1048576) + 1;
                    int s_is_head = bp_is_head;
                    int s_is_tail = bp_is_tail;
                    mbarrier_wait(alpha_full_addr + (g_cur) * 8, _phase_alpha_full);
                    mbarrier_wait(beta_full_addr + (g_cur) * 8, _phase_alpha_full);
                    unsigned int s_img[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state[_lp*2 + 0], state[_lp*2+1 + 0]));
                        s_img[_lp] = *(uint32_t*)&_bf2;
                    }
                    mbarrier_wait(load_k_full_addr + (k_cur) * 8, _phase_load_k_full);
                    if (s_is_head != 0) {
                        asm volatile("" : "+f"(q_state[0]) :: "memory");
                        asm volatile("" : "+f"(q_state[1]) :: "memory");
                        asm volatile("" : "+f"(q_state[2]) :: "memory");
                        asm volatile("" : "+f"(q_state[3]) :: "memory");
                        asm volatile("" : "+f"(q_state[4]) :: "memory");
                        asm volatile("" : "+f"(q_state[5]) :: "memory");
                        asm volatile("" : "+f"(q_state[6]) :: "memory");
                        asm volatile("" : "+f"(q_state[7]) :: "memory");
                        asm volatile("" : "+f"(q_state[8]) :: "memory");
                        asm volatile("" : "+f"(q_state[9]) :: "memory");
                        asm volatile("" : "+f"(q_state[10]) :: "memory");
                        asm volatile("" : "+f"(q_state[11]) :: "memory");
                        asm volatile("" : "+f"(q_state[12]) :: "memory");
                        asm volatile("" : "+f"(q_state[13]) :: "memory");
                        asm volatile("" : "+f"(q_state[14]) :: "memory");
                        asm volatile("" : "+f"(q_state[15]) :: "memory");
                        asm volatile("" : "+f"(q_state[16]) :: "memory");
                        asm volatile("" : "+f"(q_state[17]) :: "memory");
                        asm volatile("" : "+f"(q_state[18]) :: "memory");
                        asm volatile("" : "+f"(q_state[19]) :: "memory");
                        asm volatile("" : "+f"(q_state[20]) :: "memory");
                        asm volatile("" : "+f"(q_state[21]) :: "memory");
                        asm volatile("" : "+f"(q_state[22]) :: "memory");
                        asm volatile("" : "+f"(q_state[23]) :: "memory");
                        asm volatile("" : "+f"(q_state[24]) :: "memory");
                        asm volatile("" : "+f"(q_state[25]) :: "memory");
                        asm volatile("" : "+f"(q_state[26]) :: "memory");
                        asm volatile("" : "+f"(q_state[27]) :: "memory");
                        asm volatile("" : "+f"(q_state[28]) :: "memory");
                        asm volatile("" : "+f"(q_state[29]) :: "memory");
                        asm volatile("" : "+f"(q_state[30]) :: "memory");
                        asm volatile("" : "+f"(q_state[31]) :: "memory");
                        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                        if (chunk_idx != 0) {
                            int obt_warp = warp_id_in_role;
                            int obt_mtx = lane / 8;
                            int obt_row = lane & 7;
                            mbarrier_wait(o_empty_addr, _phase_o_empty);
                            unsigned int obt_img[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_state[_lp*2 + 0], q_state[_lp*2+1 + 0]));
                                obt_img[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int obt_base = smem_o_addr + 0 * 8192;
                            #pragma unroll
                            for (int tg = 0; tg < 4; tg++) {
                                int obt_vb = obt_warp * 16 + (obt_mtx & 1) * 8;
                                int obt_tok = tg * 16 + obt_mtx / 2 * 8 + obt_row;
                                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>((unsigned long long)(obt_base + (unsigned int)(obt_tok * 128 + obt_vb * 2 ^ (obt_tok * 128 + obt_vb * 2 >> 7 & 7) << 4)));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&obt_img[tg * 4])), "r"(*reinterpret_cast<const uint32_t*>(&obt_img[tg * 4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&obt_img[tg * 4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&obt_img[tg * 4 + 3]))
                                    : "memory");
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_arrive(o_ready_addr);
                            _phase_o_empty ^= 1;
                            mbarrier_arrive(qk_empty_addr + (((qk_cur + 2 >= 3) ? qk_cur + 2 - 3 : qk_cur + 2)) * 8);
                        }
                        mbarrier_wait(load_q_full_addr + (q_cur) * 8, _phase_load_q_full);
                        asm volatile("" : "+f"(ks[0]) :: "memory");
                        asm volatile("" : "+f"(ks[1]) :: "memory");
                        asm volatile("" : "+f"(ks[2]) :: "memory");
                        asm volatile("" : "+f"(ks[3]) :: "memory");
                        asm volatile("" : "+f"(ks[4]) :: "memory");
                        asm volatile("" : "+f"(ks[5]) :: "memory");
                        asm volatile("" : "+f"(ks[6]) :: "memory");
                        asm volatile("" : "+f"(ks[7]) :: "memory");
                        asm volatile("" : "+f"(ks[8]) :: "memory");
                        asm volatile("" : "+f"(ks[9]) :: "memory");
                        asm volatile("" : "+f"(ks[10]) :: "memory");
                        asm volatile("" : "+f"(ks[11]) :: "memory");
                        asm volatile("" : "+f"(ks[12]) :: "memory");
                        asm volatile("" : "+f"(ks[13]) :: "memory");
                        asm volatile("" : "+f"(ks[14]) :: "memory");
                        asm volatile("" : "+f"(ks[15]) :: "memory");
                        asm volatile("" : "+f"(ks[16]) :: "memory");
                        asm volatile("" : "+f"(ks[17]) :: "memory");
                        asm volatile("" : "+f"(ks[18]) :: "memory");
                        asm volatile("" : "+f"(ks[19]) :: "memory");
                        asm volatile("" : "+f"(ks[20]) :: "memory");
                        asm volatile("" : "+f"(ks[21]) :: "memory");
                        asm volatile("" : "+f"(ks[22]) :: "memory");
                        asm volatile("" : "+f"(ks[23]) :: "memory");
                        asm volatile("" : "+f"(ks[24]) :: "memory");
                        asm volatile("" : "+f"(ks[25]) :: "memory");
                        asm volatile("" : "+f"(ks[26]) :: "memory");
                        asm volatile("" : "+f"(ks[27]) :: "memory");
                        asm volatile("" : "+f"(ks[28]) :: "memory");
                        asm volatile("" : "+f"(ks[29]) :: "memory");
                        asm volatile("" : "+f"(ks[30]) :: "memory");
                        asm volatile("" : "+f"(ks[31]) :: "memory");
                        asm volatile("" : "+f"(q_state[0]) :: "memory");
                        asm volatile("" : "+f"(q_state[1]) :: "memory");
                        asm volatile("" : "+f"(q_state[2]) :: "memory");
                        asm volatile("" : "+f"(q_state[3]) :: "memory");
                        asm volatile("" : "+f"(q_state[4]) :: "memory");
                        asm volatile("" : "+f"(q_state[5]) :: "memory");
                        asm volatile("" : "+f"(q_state[6]) :: "memory");
                        asm volatile("" : "+f"(q_state[7]) :: "memory");
                        asm volatile("" : "+f"(q_state[8]) :: "memory");
                        asm volatile("" : "+f"(q_state[9]) :: "memory");
                        asm volatile("" : "+f"(q_state[10]) :: "memory");
                        asm volatile("" : "+f"(q_state[11]) :: "memory");
                        asm volatile("" : "+f"(q_state[12]) :: "memory");
                        asm volatile("" : "+f"(q_state[13]) :: "memory");
                        asm volatile("" : "+f"(q_state[14]) :: "memory");
                        asm volatile("" : "+f"(q_state[15]) :: "memory");
                        asm volatile("" : "+f"(q_state[16]) :: "memory");
                        asm volatile("" : "+f"(q_state[17]) :: "memory");
                        asm volatile("" : "+f"(q_state[18]) :: "memory");
                        asm volatile("" : "+f"(q_state[19]) :: "memory");
                        asm volatile("" : "+f"(q_state[20]) :: "memory");
                        asm volatile("" : "+f"(q_state[21]) :: "memory");
                        asm volatile("" : "+f"(q_state[22]) :: "memory");
                        asm volatile("" : "+f"(q_state[23]) :: "memory");
                        asm volatile("" : "+f"(q_state[24]) :: "memory");
                        asm volatile("" : "+f"(q_state[25]) :: "memory");
                        asm volatile("" : "+f"(q_state[26]) :: "memory");
                        asm volatile("" : "+f"(q_state[27]) :: "memory");
                        asm volatile("" : "+f"(q_state[28]) :: "memory");
                        asm volatile("" : "+f"(q_state[29]) :: "memory");
                        asm volatile("" : "+f"(q_state[30]) :: "memory");
                        asm volatile("" : "+f"(q_state[31]) :: "memory");
                        asm volatile("" : "+r"(s_img[0]) :: "memory");
                        asm volatile("" : "+r"(s_img[1]) :: "memory");
                        asm volatile("" : "+r"(s_img[2]) :: "memory");
                        asm volatile("" : "+r"(s_img[3]) :: "memory");
                        asm volatile("" : "+r"(s_img[4]) :: "memory");
                        asm volatile("" : "+r"(s_img[5]) :: "memory");
                        asm volatile("" : "+r"(s_img[6]) :: "memory");
                        asm volatile("" : "+r"(s_img[7]) :: "memory");
                        asm volatile("" : "+r"(s_img[8]) :: "memory");
                        asm volatile("" : "+r"(s_img[9]) :: "memory");
                        asm volatile("" : "+r"(s_img[10]) :: "memory");
                        asm volatile("" : "+r"(s_img[11]) :: "memory");
                        asm volatile("" : "+r"(s_img[12]) :: "memory");
                        asm volatile("" : "+r"(s_img[13]) :: "memory");
                        asm volatile("" : "+r"(s_img[14]) :: "memory");
                        asm volatile("" : "+r"(s_img[15]) :: "memory");
                        asm volatile("" : "+r"(s_img[16]) :: "memory");
                        asm volatile("" : "+r"(s_img[17]) :: "memory");
                        asm volatile("" : "+r"(s_img[18]) :: "memory");
                        asm volatile("" : "+r"(s_img[19]) :: "memory");
                        asm volatile("" : "+r"(s_img[20]) :: "memory");
                        asm volatile("" : "+r"(s_img[21]) :: "memory");
                        asm volatile("" : "+r"(s_img[22]) :: "memory");
                        asm volatile("" : "+r"(s_img[23]) :: "memory");
                        asm volatile("" : "+r"(s_img[24]) :: "memory");
                        asm volatile("" : "+r"(s_img[25]) :: "memory");
                        asm volatile("" : "+r"(s_img[26]) :: "memory");
                        asm volatile("" : "+r"(s_img[27]) :: "memory");
                        asm volatile("" : "+r"(s_img[28]) :: "memory");
                        asm volatile("" : "+r"(s_img[29]) :: "memory");
                        asm volatile("" : "+r"(s_img[30]) :: "memory");
                        asm volatile("" : "+r"(s_img[31]) :: "memory");
                        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                        uint64_t _wgmma_desc_2 = (((uint64_t)(((smem_q_addr + q_cur * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t q_lo = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
                        uint64_t _wgmma_desc_3 = (((uint64_t)(((smem_q_addr + q_cur * 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t q_hi = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_3 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_3);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[0]), "r"(s_img[1]), "r"(s_img[2]), "r"(s_img[3]), "l"(q_lo)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[4]), "r"(s_img[(4) + 1]), "r"(s_img[(4) + 2]), "r"(s_img[(4) + 3]), "l"(q_lo + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[8]), "r"(s_img[(8) + 1]), "r"(s_img[(8) + 2]), "r"(s_img[(8) + 3]), "l"(q_lo + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[12]), "r"(s_img[(12) + 1]), "r"(s_img[(12) + 2]), "r"(s_img[(12) + 3]), "l"(q_lo + 6)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[16]), "r"(s_img[(16) + 1]), "r"(s_img[(16) + 2]), "r"(s_img[(16) + 3]), "l"(q_hi)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[20]), "r"(s_img[(20) + 1]), "r"(s_img[(20) + 2]), "r"(s_img[(20) + 3]), "l"(q_hi + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[24]), "r"(s_img[(24) + 1]), "r"(s_img[(24) + 2]), "r"(s_img[(24) + 3]), "l"(q_hi + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[28]), "r"(s_img[(28) + 1]), "r"(s_img[(28) + 2]), "r"(s_img[(28) + 3]), "l"(q_hi + 6)
                            : "memory");
                        uint64_t _wgmma_desc_4 = (((uint64_t)(((smem_k_addr + k_cur * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t kh_lo = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_4 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_4);
                        uint64_t _wgmma_desc_5 = (((uint64_t)(((smem_k_addr + k_cur * 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t kh_hi = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_5 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_5);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[0]), "r"(s_img[1]), "r"(s_img[2]), "r"(s_img[3]), "l"(kh_lo)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[4]), "r"(s_img[(4) + 1]), "r"(s_img[(4) + 2]), "r"(s_img[(4) + 3]), "l"(kh_lo + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[8]), "r"(s_img[(8) + 1]), "r"(s_img[(8) + 2]), "r"(s_img[(8) + 3]), "l"(kh_lo + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[12]), "r"(s_img[(12) + 1]), "r"(s_img[(12) + 2]), "r"(s_img[(12) + 3]), "l"(kh_lo + 6)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[16]), "r"(s_img[(16) + 1]), "r"(s_img[(16) + 2]), "r"(s_img[(16) + 3]), "l"(kh_hi)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[20]), "r"(s_img[(20) + 1]), "r"(s_img[(20) + 2]), "r"(s_img[(20) + 3]), "l"(kh_hi + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[24]), "r"(s_img[(24) + 1]), "r"(s_img[(24) + 2]), "r"(s_img[(24) + 3]), "l"(kh_hi + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[28]), "r"(s_img[(28) + 1]), "r"(s_img[(28) + 2]), "r"(s_img[(28) + 3]), "l"(kh_hi + 6)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        asm volatile("" : "+f"(ks[0]) :: "memory");
                        asm volatile("" : "+f"(ks[1]) :: "memory");
                        asm volatile("" : "+f"(ks[2]) :: "memory");
                        asm volatile("" : "+f"(ks[3]) :: "memory");
                        asm volatile("" : "+f"(ks[4]) :: "memory");
                        asm volatile("" : "+f"(ks[5]) :: "memory");
                        asm volatile("" : "+f"(ks[6]) :: "memory");
                        asm volatile("" : "+f"(ks[7]) :: "memory");
                        asm volatile("" : "+f"(ks[8]) :: "memory");
                        asm volatile("" : "+f"(ks[9]) :: "memory");
                        asm volatile("" : "+f"(ks[10]) :: "memory");
                        asm volatile("" : "+f"(ks[11]) :: "memory");
                        asm volatile("" : "+f"(ks[12]) :: "memory");
                        asm volatile("" : "+f"(ks[13]) :: "memory");
                        asm volatile("" : "+f"(ks[14]) :: "memory");
                        asm volatile("" : "+f"(ks[15]) :: "memory");
                        asm volatile("" : "+f"(ks[16]) :: "memory");
                        asm volatile("" : "+f"(ks[17]) :: "memory");
                        asm volatile("" : "+f"(ks[18]) :: "memory");
                        asm volatile("" : "+f"(ks[19]) :: "memory");
                        asm volatile("" : "+f"(ks[20]) :: "memory");
                        asm volatile("" : "+f"(ks[21]) :: "memory");
                        asm volatile("" : "+f"(ks[22]) :: "memory");
                        asm volatile("" : "+f"(ks[23]) :: "memory");
                        asm volatile("" : "+f"(ks[24]) :: "memory");
                        asm volatile("" : "+f"(ks[25]) :: "memory");
                        asm volatile("" : "+f"(ks[26]) :: "memory");
                        asm volatile("" : "+f"(ks[27]) :: "memory");
                        asm volatile("" : "+f"(ks[28]) :: "memory");
                        asm volatile("" : "+f"(ks[29]) :: "memory");
                        asm volatile("" : "+f"(ks[30]) :: "memory");
                        asm volatile("" : "+f"(ks[31]) :: "memory");
                        asm volatile("" : "+f"(q_state[0]) :: "memory");
                        asm volatile("" : "+f"(q_state[1]) :: "memory");
                        asm volatile("" : "+f"(q_state[2]) :: "memory");
                        asm volatile("" : "+f"(q_state[3]) :: "memory");
                        asm volatile("" : "+f"(q_state[4]) :: "memory");
                        asm volatile("" : "+f"(q_state[5]) :: "memory");
                        asm volatile("" : "+f"(q_state[6]) :: "memory");
                        asm volatile("" : "+f"(q_state[7]) :: "memory");
                        asm volatile("" : "+f"(q_state[8]) :: "memory");
                        asm volatile("" : "+f"(q_state[9]) :: "memory");
                        asm volatile("" : "+f"(q_state[10]) :: "memory");
                        asm volatile("" : "+f"(q_state[11]) :: "memory");
                        asm volatile("" : "+f"(q_state[12]) :: "memory");
                        asm volatile("" : "+f"(q_state[13]) :: "memory");
                        asm volatile("" : "+f"(q_state[14]) :: "memory");
                        asm volatile("" : "+f"(q_state[15]) :: "memory");
                        asm volatile("" : "+f"(q_state[16]) :: "memory");
                        asm volatile("" : "+f"(q_state[17]) :: "memory");
                        asm volatile("" : "+f"(q_state[18]) :: "memory");
                        asm volatile("" : "+f"(q_state[19]) :: "memory");
                        asm volatile("" : "+f"(q_state[20]) :: "memory");
                        asm volatile("" : "+f"(q_state[21]) :: "memory");
                        asm volatile("" : "+f"(q_state[22]) :: "memory");
                        asm volatile("" : "+f"(q_state[23]) :: "memory");
                        asm volatile("" : "+f"(q_state[24]) :: "memory");
                        asm volatile("" : "+f"(q_state[25]) :: "memory");
                        asm volatile("" : "+f"(q_state[26]) :: "memory");
                        asm volatile("" : "+f"(q_state[27]) :: "memory");
                        asm volatile("" : "+f"(q_state[28]) :: "memory");
                        asm volatile("" : "+f"(q_state[29]) :: "memory");
                        asm volatile("" : "+f"(q_state[30]) :: "memory");
                        asm volatile("" : "+f"(q_state[31]) :: "memory");
                        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                    } else {
                        asm volatile("" : "+f"(ks[0]) :: "memory");
                        asm volatile("" : "+f"(ks[1]) :: "memory");
                        asm volatile("" : "+f"(ks[2]) :: "memory");
                        asm volatile("" : "+f"(ks[3]) :: "memory");
                        asm volatile("" : "+f"(ks[4]) :: "memory");
                        asm volatile("" : "+f"(ks[5]) :: "memory");
                        asm volatile("" : "+f"(ks[6]) :: "memory");
                        asm volatile("" : "+f"(ks[7]) :: "memory");
                        asm volatile("" : "+f"(ks[8]) :: "memory");
                        asm volatile("" : "+f"(ks[9]) :: "memory");
                        asm volatile("" : "+f"(ks[10]) :: "memory");
                        asm volatile("" : "+f"(ks[11]) :: "memory");
                        asm volatile("" : "+f"(ks[12]) :: "memory");
                        asm volatile("" : "+f"(ks[13]) :: "memory");
                        asm volatile("" : "+f"(ks[14]) :: "memory");
                        asm volatile("" : "+f"(ks[15]) :: "memory");
                        asm volatile("" : "+f"(ks[16]) :: "memory");
                        asm volatile("" : "+f"(ks[17]) :: "memory");
                        asm volatile("" : "+f"(ks[18]) :: "memory");
                        asm volatile("" : "+f"(ks[19]) :: "memory");
                        asm volatile("" : "+f"(ks[20]) :: "memory");
                        asm volatile("" : "+f"(ks[21]) :: "memory");
                        asm volatile("" : "+f"(ks[22]) :: "memory");
                        asm volatile("" : "+f"(ks[23]) :: "memory");
                        asm volatile("" : "+f"(ks[24]) :: "memory");
                        asm volatile("" : "+f"(ks[25]) :: "memory");
                        asm volatile("" : "+f"(ks[26]) :: "memory");
                        asm volatile("" : "+f"(ks[27]) :: "memory");
                        asm volatile("" : "+f"(ks[28]) :: "memory");
                        asm volatile("" : "+f"(ks[29]) :: "memory");
                        asm volatile("" : "+f"(ks[30]) :: "memory");
                        asm volatile("" : "+f"(ks[31]) :: "memory");
                        asm volatile("" : "+r"(s_img[0]) :: "memory");
                        asm volatile("" : "+r"(s_img[1]) :: "memory");
                        asm volatile("" : "+r"(s_img[2]) :: "memory");
                        asm volatile("" : "+r"(s_img[3]) :: "memory");
                        asm volatile("" : "+r"(s_img[4]) :: "memory");
                        asm volatile("" : "+r"(s_img[5]) :: "memory");
                        asm volatile("" : "+r"(s_img[6]) :: "memory");
                        asm volatile("" : "+r"(s_img[7]) :: "memory");
                        asm volatile("" : "+r"(s_img[8]) :: "memory");
                        asm volatile("" : "+r"(s_img[9]) :: "memory");
                        asm volatile("" : "+r"(s_img[10]) :: "memory");
                        asm volatile("" : "+r"(s_img[11]) :: "memory");
                        asm volatile("" : "+r"(s_img[12]) :: "memory");
                        asm volatile("" : "+r"(s_img[13]) :: "memory");
                        asm volatile("" : "+r"(s_img[14]) :: "memory");
                        asm volatile("" : "+r"(s_img[15]) :: "memory");
                        asm volatile("" : "+r"(s_img[16]) :: "memory");
                        asm volatile("" : "+r"(s_img[17]) :: "memory");
                        asm volatile("" : "+r"(s_img[18]) :: "memory");
                        asm volatile("" : "+r"(s_img[19]) :: "memory");
                        asm volatile("" : "+r"(s_img[20]) :: "memory");
                        asm volatile("" : "+r"(s_img[21]) :: "memory");
                        asm volatile("" : "+r"(s_img[22]) :: "memory");
                        asm volatile("" : "+r"(s_img[23]) :: "memory");
                        asm volatile("" : "+r"(s_img[24]) :: "memory");
                        asm volatile("" : "+r"(s_img[25]) :: "memory");
                        asm volatile("" : "+r"(s_img[26]) :: "memory");
                        asm volatile("" : "+r"(s_img[27]) :: "memory");
                        asm volatile("" : "+r"(s_img[28]) :: "memory");
                        asm volatile("" : "+r"(s_img[29]) :: "memory");
                        asm volatile("" : "+r"(s_img[30]) :: "memory");
                        asm volatile("" : "+r"(s_img[31]) :: "memory");
                        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                        uint64_t _wgmma_desc_6 = (((uint64_t)(((smem_k_addr + k_cur * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t k_lo = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_6 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_6);
                        uint64_t _wgmma_desc_7 = (((uint64_t)(((smem_k_addr + k_cur * 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t k_hi = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_7 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_7);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[0]), "r"(s_img[1]), "r"(s_img[2]), "r"(s_img[3]), "l"(k_lo)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[4]), "r"(s_img[(4) + 1]), "r"(s_img[(4) + 2]), "r"(s_img[(4) + 3]), "l"(k_lo + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[8]), "r"(s_img[(8) + 1]), "r"(s_img[(8) + 2]), "r"(s_img[(8) + 3]), "l"(k_lo + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[12]), "r"(s_img[(12) + 1]), "r"(s_img[(12) + 2]), "r"(s_img[(12) + 3]), "l"(k_lo + 6)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[16]), "r"(s_img[(16) + 1]), "r"(s_img[(16) + 2]), "r"(s_img[(16) + 3]), "l"(k_hi)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[20]), "r"(s_img[(20) + 1]), "r"(s_img[(20) + 2]), "r"(s_img[(20) + 3]), "l"(k_hi + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[24]), "r"(s_img[(24) + 1]), "r"(s_img[(24) + 2]), "r"(s_img[(24) + 3]), "l"(k_hi + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[28]), "r"(s_img[(28) + 1]), "r"(s_img[(28) + 2]), "r"(s_img[(28) + 3]), "l"(k_hi + 6)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        asm volatile("" : "+f"(ks[0]) :: "memory");
                        asm volatile("" : "+f"(ks[1]) :: "memory");
                        asm volatile("" : "+f"(ks[2]) :: "memory");
                        asm volatile("" : "+f"(ks[3]) :: "memory");
                        asm volatile("" : "+f"(ks[4]) :: "memory");
                        asm volatile("" : "+f"(ks[5]) :: "memory");
                        asm volatile("" : "+f"(ks[6]) :: "memory");
                        asm volatile("" : "+f"(ks[7]) :: "memory");
                        asm volatile("" : "+f"(ks[8]) :: "memory");
                        asm volatile("" : "+f"(ks[9]) :: "memory");
                        asm volatile("" : "+f"(ks[10]) :: "memory");
                        asm volatile("" : "+f"(ks[11]) :: "memory");
                        asm volatile("" : "+f"(ks[12]) :: "memory");
                        asm volatile("" : "+f"(ks[13]) :: "memory");
                        asm volatile("" : "+f"(ks[14]) :: "memory");
                        asm volatile("" : "+f"(ks[15]) :: "memory");
                        asm volatile("" : "+f"(ks[16]) :: "memory");
                        asm volatile("" : "+f"(ks[17]) :: "memory");
                        asm volatile("" : "+f"(ks[18]) :: "memory");
                        asm volatile("" : "+f"(ks[19]) :: "memory");
                        asm volatile("" : "+f"(ks[20]) :: "memory");
                        asm volatile("" : "+f"(ks[21]) :: "memory");
                        asm volatile("" : "+f"(ks[22]) :: "memory");
                        asm volatile("" : "+f"(ks[23]) :: "memory");
                        asm volatile("" : "+f"(ks[24]) :: "memory");
                        asm volatile("" : "+f"(ks[25]) :: "memory");
                        asm volatile("" : "+f"(ks[26]) :: "memory");
                        asm volatile("" : "+f"(ks[27]) :: "memory");
                        asm volatile("" : "+f"(ks[28]) :: "memory");
                        asm volatile("" : "+f"(ks[29]) :: "memory");
                        asm volatile("" : "+f"(ks[30]) :: "memory");
                        asm volatile("" : "+f"(ks[31]) :: "memory");
                        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                        asm volatile("" : "+f"(q_state[0]) :: "memory");
                        asm volatile("" : "+f"(q_state[1]) :: "memory");
                        asm volatile("" : "+f"(q_state[2]) :: "memory");
                        asm volatile("" : "+f"(q_state[3]) :: "memory");
                        asm volatile("" : "+f"(q_state[4]) :: "memory");
                        asm volatile("" : "+f"(q_state[5]) :: "memory");
                        asm volatile("" : "+f"(q_state[6]) :: "memory");
                        asm volatile("" : "+f"(q_state[7]) :: "memory");
                        asm volatile("" : "+f"(q_state[8]) :: "memory");
                        asm volatile("" : "+f"(q_state[9]) :: "memory");
                        asm volatile("" : "+f"(q_state[10]) :: "memory");
                        asm volatile("" : "+f"(q_state[11]) :: "memory");
                        asm volatile("" : "+f"(q_state[12]) :: "memory");
                        asm volatile("" : "+f"(q_state[13]) :: "memory");
                        asm volatile("" : "+f"(q_state[14]) :: "memory");
                        asm volatile("" : "+f"(q_state[15]) :: "memory");
                        asm volatile("" : "+f"(q_state[16]) :: "memory");
                        asm volatile("" : "+f"(q_state[17]) :: "memory");
                        asm volatile("" : "+f"(q_state[18]) :: "memory");
                        asm volatile("" : "+f"(q_state[19]) :: "memory");
                        asm volatile("" : "+f"(q_state[20]) :: "memory");
                        asm volatile("" : "+f"(q_state[21]) :: "memory");
                        asm volatile("" : "+f"(q_state[22]) :: "memory");
                        asm volatile("" : "+f"(q_state[23]) :: "memory");
                        asm volatile("" : "+f"(q_state[24]) :: "memory");
                        asm volatile("" : "+f"(q_state[25]) :: "memory");
                        asm volatile("" : "+f"(q_state[26]) :: "memory");
                        asm volatile("" : "+f"(q_state[27]) :: "memory");
                        asm volatile("" : "+f"(q_state[28]) :: "memory");
                        asm volatile("" : "+f"(q_state[29]) :: "memory");
                        asm volatile("" : "+f"(q_state[30]) :: "memory");
                        asm volatile("" : "+f"(q_state[31]) :: "memory");
                        mbarrier_arrive(qk_empty_addr + (((qk_cur + 2 >= 3) ? qk_cur + 2 - 3 : qk_cur + 2)) * 8);
                    }
                    if (s_is_head != 0) {
                        #pragma unroll
                        for (int j = 0; j < 32; j++) {
                            int s_col = j / 4 * 8 + s_quad * 2 + (j & 1);
                            q_state[j] = q_state[j] * smem_gate[s_gb + 256 + s_col];
                        }
                        mbarrier_arrive(q_empty_addr + (q_cur) * 8);
                        q_cur += 1;
                        if (q_cur == 2) { q_cur = 0; _phase_load_q_full ^= 1; }
                    }
                    mbarrier_wait(load_v_full_addr + (v_cur) * 8, _phase_load_v_full);
                    unsigned int v_base = smem_v_addr + v_cur * 8192;
                    unsigned int vraw[16];
                    #pragma unroll
                    for (int nn = 0; nn < 4; nn++) {
                        int m_row = (2 * nn + s_ldm_i_off) * 8 + s_row_in_mtx;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(vraw[4 * nn]), "=r"(vraw[4 * nn + 1]), "=r"(vraw[4 * nn + 2]), "=r"(vraw[4 * nn + 3])
                            : "r"(v_base + (unsigned int)(m_row * 128 + s_ldm_v0 * 2 ^ (m_row * 128 + s_ldm_v0 * 2 >> 7 & 7) << 4))
                            : "memory");
                    }
                    float cp16[16];
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        #pragma unroll
                        for (int e = 0; e < 2; e++) {
                            cp16[2 * i + e] = smem_gate[s_gb + 64 + i * 8 + s_quad * 2 + e];
                        }
                    }
                    float vf[32];
                    #pragma unroll
                    for (int _pair = 0; _pair < 16; _pair++) {
                        (&vf[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(vraw[_pair]) << 16);
                        (&vf[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(vraw[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 32; j_1++) {
                        vf[j_1] = __fmaf_rn(-cp16[j_1 / 4 * 2 + (j_1 & 1)], ks[j_1], vf[j_1]);
                    }
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 32; j_2++) {
                        ks[j_2] = ks[j_2] * cp16[j_2 / 4 * 2 + (j_2 & 1)];
                    }
                    unsigned int ks_img[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks[_lp*2 + 0], ks[_lp*2+1 + 0]));
                        ks_img[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int y_img[16];
                    #pragma unroll
                    for (int w = 0; w < 16; w++) {
                        uint32_t y_w;
                        asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(y_w) : "r"(vraw[w]), "r"(ks_img[w]));
                        y_img[w] = y_w;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(v_empty_addr + (v_cur) * 8);
                    float gamma = smem_gate[s_gb + 64 + 63];
                    float bpair[16];
                    #pragma unroll
                    for (int i_1 = 0; i_1 < 8; i_1++) {
                        #pragma unroll
                        for (int e_1 = 0; e_1 < 2; e_1++) {
                            bpair[2 * i_1 + e_1] = smem_gate[s_gb + 320 + i_1 * 8 + s_quad * 2 + e_1];
                        }
                    }
                    mbarrier_wait(ab_ready_addr + (ab_cur) * 8, _phase_ab_ready);
                    unsigned int s_tile_inv = smem_ab_addr + ab_cur * 8192;
                    int s_thread = s_warp * 32 + lane;
                    unsigned int s_x21[8];
                    unsigned int s_pw[16];
                    int s_prow = lane % 16;
                    int s_pcol = lane / 16 * 8;
                    unsigned int s_abase = smem_abo_addr + ab_cur * 8192;
                    if (s_warp >= 2) {
                        int f64_y = s_warp & 1;
                        int f64_lane_row = lane % 16;
                        int f64_lane_col = lane / 16 * 8;
                        unsigned int f64_d[4];
                        unsigned int f64_c[4];
                        float f64_dc_acc[16];
                        f64_dc_acc[0] = 0.0f;
                        f64_dc_acc[1] = 0.0f;
                        f64_dc_acc[2] = 0.0f;
                        f64_dc_acc[3] = 0.0f;
                        f64_dc_acc[4] = 0.0f;
                        f64_dc_acc[5] = 0.0f;
                        f64_dc_acc[6] = 0.0f;
                        f64_dc_acc[7] = 0.0f;
                        f64_dc_acc[8] = 0.0f;
                        f64_dc_acc[9] = 0.0f;
                        f64_dc_acc[10] = 0.0f;
                        f64_dc_acc[11] = 0.0f;
                        f64_dc_acc[12] = 0.0f;
                        f64_dc_acc[13] = 0.0f;
                        f64_dc_acc[14] = 0.0f;
                        f64_dc_acc[15] = 0.0f;
                        #pragma unroll
                        for (int f64_k = 0; f64_k < 2; f64_k++) {
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(f64_d[0]), "=r"(f64_d[1]), "=r"(f64_d[2]), "=r"(f64_d[3])
                                : "r"(s_tile_inv + (unsigned int)((32 + f64_k * 16 + f64_lane_col) / 8 * 1024 + (32 + f64_y * 16 + f64_lane_row) * 16 + (32 + f64_k * 16 + f64_lane_col) % 8 * 2))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(f64_c[0]), "=r"(f64_c[1]), "=r"(f64_c[2]), "=r"(f64_c[3])
                                : "r"(s_tile_inv + (unsigned int)(f64_lane_col / 8 * 1024 + (32 + f64_k * 16 + f64_lane_row) * 16 + f64_lane_col % 8 * 2))
                                : "memory");
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(f64_dc_acc[0]), "+f"(f64_dc_acc[1]), "+f"(f64_dc_acc[2]), "+f"(f64_dc_acc[3])
                                : "r"(f64_d[0]), "r"(f64_d[1]), "r"(f64_d[2]), "r"(f64_d[3]), "r"(f64_c[0]), "r"(f64_c[1]));
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(f64_dc_acc[4]), "+f"(f64_dc_acc[(4) + 1]), "+f"(f64_dc_acc[(4) + 2]), "+f"(f64_dc_acc[(4) + 3])
                                : "r"(f64_d[0]), "r"(f64_d[1]), "r"(f64_d[2]), "r"(f64_d[3]), "r"(f64_c[2]), "r"(f64_c[(2) + 1]));
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(f64_c[0]), "=r"(f64_c[1]), "=r"(f64_c[2]), "=r"(f64_c[3])
                                : "r"(s_tile_inv + (unsigned int)((16 + f64_lane_col) / 8 * 1024 + (32 + f64_k * 16 + f64_lane_row) * 16 + (16 + f64_lane_col) % 8 * 2))
                                : "memory");
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(f64_dc_acc[8]), "+f"(f64_dc_acc[(8) + 1]), "+f"(f64_dc_acc[(8) + 2]), "+f"(f64_dc_acc[(8) + 3])
                                : "r"(f64_d[0]), "r"(f64_d[1]), "r"(f64_d[2]), "r"(f64_d[3]), "r"(f64_c[0]), "r"(f64_c[1]));
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(f64_dc_acc[12]), "+f"(f64_dc_acc[(12) + 1]), "+f"(f64_dc_acc[(12) + 2]), "+f"(f64_dc_acc[(12) + 3])
                                : "r"(f64_d[0]), "r"(f64_d[1]), "r"(f64_d[2]), "r"(f64_d[3]), "r"(f64_c[2]), "r"(f64_c[(2) + 1]));
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++) {
                            f64_dc_acc[_ls] = f64_dc_acc[_ls] * -1.0f;
                        }
                        unsigned int f64_dc_f16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __half2 _h2 = __float22half2_rn(make_float2(f64_dc_acc[_lp*2 + 0], f64_dc_acc[_lp*2+1 + 0]));
                            f64_dc_f16[_lp] = *(uint32_t*)&_h2;
                        }
                        unsigned int f64_a[8];
                        float f64_out0_acc[8];
                        float f64_out1_acc[8];
                        f64_out0_acc[0] = 0.0f;
                        f64_out1_acc[0] = 0.0f;
                        f64_out0_acc[1] = 0.0f;
                        f64_out1_acc[1] = 0.0f;
                        f64_out0_acc[2] = 0.0f;
                        f64_out1_acc[2] = 0.0f;
                        f64_out0_acc[3] = 0.0f;
                        f64_out1_acc[3] = 0.0f;
                        f64_out0_acc[4] = 0.0f;
                        f64_out1_acc[4] = 0.0f;
                        f64_out0_acc[5] = 0.0f;
                        f64_out1_acc[5] = 0.0f;
                        f64_out0_acc[6] = 0.0f;
                        f64_out1_acc[6] = 0.0f;
                        f64_out0_acc[7] = 0.0f;
                        f64_out1_acc[7] = 0.0f;
                        unsigned int f64_dc_lo[4];
                        unsigned int f64_dc_hi[4];
                        f64_dc_lo[0] = f64_dc_f16[0];
                        f64_dc_hi[0] = f64_dc_f16[4];
                        f64_dc_lo[1] = f64_dc_f16[1];
                        f64_dc_hi[1] = f64_dc_f16[5];
                        f64_dc_lo[2] = f64_dc_f16[2];
                        f64_dc_hi[2] = f64_dc_f16[6];
                        f64_dc_lo[3] = f64_dc_f16[3];
                        f64_dc_hi[3] = f64_dc_f16[7];
                        #pragma unroll
                        for (int f64_a_repeat = 0; f64_a_repeat < 4; f64_a_repeat++) {
                            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                : "=r"(f64_a[f64_a_repeat * 2]), "=r"(f64_a[f64_a_repeat * 2 + 1])
                                : "r"(s_tile_inv + (unsigned int)(f64_a_repeat * 8 / 8 * 1024 + f64_lane_row * 16 + f64_a_repeat * 8 % 8 * 2))
                                : "memory");
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(f64_out0_acc[0]), "+f"(f64_out0_acc[1]), "+f"(f64_out0_acc[2]), "+f"(f64_out0_acc[3])
                            : "r"(f64_dc_lo[0]), "r"(f64_dc_lo[1]), "r"(f64_dc_lo[2]), "r"(f64_dc_lo[3]), "r"(f64_a[0]), "r"(f64_a[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(f64_out0_acc[4]), "+f"(f64_out0_acc[(4) + 1]), "+f"(f64_out0_acc[(4) + 2]), "+f"(f64_out0_acc[(4) + 3])
                            : "r"(f64_dc_lo[0]), "r"(f64_dc_lo[1]), "r"(f64_dc_lo[2]), "r"(f64_dc_lo[3]), "r"(f64_a[2]), "r"(f64_a[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(f64_out1_acc[0]), "+f"(f64_out1_acc[1]), "+f"(f64_out1_acc[2]), "+f"(f64_out1_acc[3])
                            : "r"(f64_dc_lo[0]), "r"(f64_dc_lo[1]), "r"(f64_dc_lo[2]), "r"(f64_dc_lo[3]), "r"(f64_a[4]), "r"(f64_a[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(f64_out1_acc[4]), "+f"(f64_out1_acc[(4) + 1]), "+f"(f64_out1_acc[(4) + 2]), "+f"(f64_out1_acc[(4) + 3])
                            : "r"(f64_dc_lo[0]), "r"(f64_dc_lo[1]), "r"(f64_dc_lo[2]), "r"(f64_dc_lo[3]), "r"(f64_a[6]), "r"(f64_a[(6) + 1]));
                        #pragma unroll
                        for (int f64_b_repeat = 0; f64_b_repeat < 4; f64_b_repeat++) {
                            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                : "=r"(f64_a[f64_b_repeat * 2]), "=r"(f64_a[f64_b_repeat * 2 + 1])
                                : "r"(s_tile_inv + (unsigned int)(f64_b_repeat * 8 / 8 * 1024 + (16 + f64_lane_row) * 16 + f64_b_repeat * 8 % 8 * 2))
                                : "memory");
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(f64_out0_acc[0]), "+f"(f64_out0_acc[1]), "+f"(f64_out0_acc[2]), "+f"(f64_out0_acc[3])
                            : "r"(f64_dc_hi[0]), "r"(f64_dc_hi[1]), "r"(f64_dc_hi[2]), "r"(f64_dc_hi[3]), "r"(f64_a[0]), "r"(f64_a[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(f64_out0_acc[4]), "+f"(f64_out0_acc[(4) + 1]), "+f"(f64_out0_acc[(4) + 2]), "+f"(f64_out0_acc[(4) + 3])
                            : "r"(f64_dc_hi[0]), "r"(f64_dc_hi[1]), "r"(f64_dc_hi[2]), "r"(f64_dc_hi[3]), "r"(f64_a[2]), "r"(f64_a[(2) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(f64_out1_acc[0]), "+f"(f64_out1_acc[1]), "+f"(f64_out1_acc[2]), "+f"(f64_out1_acc[3])
                            : "r"(f64_dc_hi[0]), "r"(f64_dc_hi[1]), "r"(f64_dc_hi[2]), "r"(f64_dc_hi[3]), "r"(f64_a[4]), "r"(f64_a[(4) + 1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(f64_out1_acc[4]), "+f"(f64_out1_acc[(4) + 1]), "+f"(f64_out1_acc[(4) + 2]), "+f"(f64_out1_acc[(4) + 3])
                            : "r"(f64_dc_hi[0]), "r"(f64_dc_hi[1]), "r"(f64_dc_hi[2]), "r"(f64_dc_hi[3]), "r"(f64_a[6]), "r"(f64_a[(6) + 1]));
                        unsigned int f64_out0_f16[4];
                        unsigned int f64_out1_f16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __half2 _h2 = __float22half2_rn(make_float2(f64_out0_acc[_lp*2 + 0], f64_out0_acc[_lp*2+1 + 0]));
                            f64_out0_f16[_lp] = *(uint32_t*)&_h2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __half2 _h2 = __float22half2_rn(make_float2(f64_out1_acc[_lp*2 + 0], f64_out1_acc[_lp*2+1 + 0]));
                            f64_out1_f16[_lp] = *(uint32_t*)&_h2;
                        }
                        s_x21[0] = f64_out0_f16[0];
                        s_x21[4] = f64_out1_f16[0];
                        s_x21[1] = f64_out0_f16[1];
                        s_x21[5] = f64_out1_f16[1];
                        s_x21[2] = f64_out0_f16[2];
                        s_x21[6] = f64_out1_f16[2];
                        s_x21[3] = f64_out0_f16[3];
                        s_x21[7] = f64_out1_f16[3];
                    }
                    if (s_warp >= 2) {
                        s_pw[0] = s_x21[0];
                        s_pw[1] = s_x21[1];
                        s_pw[2] = s_x21[2];
                        s_pw[3] = s_x21[3];
                        s_pw[4] = s_x21[4];
                        s_pw[5] = s_x21[5];
                        s_pw[6] = s_x21[6];
                        s_pw[7] = s_x21[7];
                        int pq_r0 = s_warp * 16 + lane / 4;
                        int pq_r1 = pq_r0 + 8;
                        int pq_srow = s_warp * 16 + (lane & 7) + (lane & 8);
                        int pq_scol = (lane & 16) / 2;
                        #pragma unroll
                        for (int gn = 0; gn < 2; gn++) {
                            float pq_vals[8];
                            unsigned int pq_blk[4];
                            #pragma unroll
                            for (int w_1 = 0; w_1 < 4; w_1++) {
                                pq_blk[w_1] = s_pw[gn * 4 + w_1];
                            }
                            #pragma unroll
                            for (int word = 0; word < 4; word++) {
                                unsigned int hi_bits = pq_blk[word] >> 16;
                                __half lo_f16 = 0.0f;
                                __half hi_f16 = 0.0f;
                                unsigned int _bit_cast_src_8 = pq_blk[word];
                                lo_f16 = reinterpret_cast<__half*>(&_bit_cast_src_8)[0];
                                hi_f16 = reinterpret_cast<__half*>(&hi_bits)[0];
                                pq_vals[word * 2] = (float)lo_f16;
                                pq_vals[word * 2 + 1] = (float)hi_f16;
                            }
                            int pq_c0 = gn * 16 + (lane & 3) * 2;
                            float pq_out[8];
                            pq_out[0] = ((pq_r0 > pq_c0) ? pq_vals[0] * bpair[gn * 4] : 0.0f);
                            pq_out[1] = ((pq_r0 > pq_c0 + 1) ? pq_vals[1] * bpair[gn * 4 + 1] : 0.0f);
                            pq_out[2] = ((pq_r1 > pq_c0) ? pq_vals[2] * bpair[gn * 4] : 0.0f);
                            pq_out[3] = ((pq_r1 > pq_c0 + 1) ? pq_vals[3] * bpair[gn * 4 + 1] : 0.0f);
                            pq_out[4] = ((pq_r0 > pq_c0 + 8) ? pq_vals[4] * bpair[gn * 4 + 2] : 0.0f);
                            pq_out[5] = ((pq_r0 > pq_c0 + 9) ? pq_vals[5] * bpair[gn * 4 + 3] : 0.0f);
                            pq_out[6] = ((pq_r1 > pq_c0 + 8) ? pq_vals[6] * bpair[gn * 4 + 2] : 0.0f);
                            pq_out[7] = ((pq_r1 > pq_c0 + 9) ? pq_vals[7] * bpair[gn * 4 + 3] : 0.0f);
                            unsigned int pq_packed[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pq_out[_lp*2 + 0], pq_out[_lp*2+1 + 0]));
                                pq_packed[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_9 = static_cast<uint32_t>((unsigned long long)(s_abase + (unsigned int)(pq_srow * 128 + (gn * 16 + pq_scol) * 2 ^ (pq_srow * 128 + (gn * 16 + pq_scol) * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed[3]))
                                : "memory");
                        }
                    } else {
                        #pragma unroll
                        for (int gn_1 = 0; gn_1 < 2; gn_1++) {
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(s_pw[gn_1 * 4]), "=r"(s_pw[gn_1 * 4 + 1]), "=r"(s_pw[gn_1 * 4 + 2]), "=r"(s_pw[gn_1 * 4 + 3])
                                : "r"(s_tile_inv + (unsigned int)((gn_1 * 16 + s_pcol) / 8 * 1024 + (s_warp * 16 + s_prow) * 16 + (gn_1 * 16 + s_pcol) % 8 * 2))
                                : "memory");
                        }
                        int pq_r0_1 = s_warp * 16 + lane / 4;
                        int pq_r1_1 = pq_r0_1 + 8;
                        int pq_srow_1 = s_warp * 16 + (lane & 7) + (lane & 8);
                        int pq_scol_1 = (lane & 16) / 2;
                        #pragma unroll
                        for (int gn_2 = 0; gn_2 < 2; gn_2++) {
                            float pq_vals_1[8];
                            unsigned int pq_blk_1[4];
                            #pragma unroll
                            for (int w_2 = 0; w_2 < 4; w_2++) {
                                pq_blk_1[w_2] = s_pw[gn_2 * 4 + w_2];
                            }
                            #pragma unroll
                            for (int word_1 = 0; word_1 < 4; word_1++) {
                                unsigned int hi_bits_1 = pq_blk_1[word_1] >> 16;
                                __half lo_f16_1 = 0.0f;
                                __half hi_f16_1 = 0.0f;
                                unsigned int _bit_cast_src_10 = pq_blk_1[word_1];
                                lo_f16_1 = reinterpret_cast<__half*>(&_bit_cast_src_10)[0];
                                hi_f16_1 = reinterpret_cast<__half*>(&hi_bits_1)[0];
                                pq_vals_1[word_1 * 2] = (float)lo_f16_1;
                                pq_vals_1[word_1 * 2 + 1] = (float)hi_f16_1;
                            }
                            int pq_c0_1 = gn_2 * 16 + (lane & 3) * 2;
                            float pq_out_1[8];
                            pq_out_1[0] = ((pq_r0_1 > pq_c0_1) ? pq_vals_1[0] * bpair[gn_2 * 4] : 0.0f);
                            pq_out_1[1] = ((pq_r0_1 > pq_c0_1 + 1) ? pq_vals_1[1] * bpair[gn_2 * 4 + 1] : 0.0f);
                            pq_out_1[2] = ((pq_r1_1 > pq_c0_1) ? pq_vals_1[2] * bpair[gn_2 * 4] : 0.0f);
                            pq_out_1[3] = ((pq_r1_1 > pq_c0_1 + 1) ? pq_vals_1[3] * bpair[gn_2 * 4 + 1] : 0.0f);
                            pq_out_1[4] = ((pq_r0_1 > pq_c0_1 + 8) ? pq_vals_1[4] * bpair[gn_2 * 4 + 2] : 0.0f);
                            pq_out_1[5] = ((pq_r0_1 > pq_c0_1 + 9) ? pq_vals_1[5] * bpair[gn_2 * 4 + 3] : 0.0f);
                            pq_out_1[6] = ((pq_r1_1 > pq_c0_1 + 8) ? pq_vals_1[6] * bpair[gn_2 * 4 + 2] : 0.0f);
                            pq_out_1[7] = ((pq_r1_1 > pq_c0_1 + 9) ? pq_vals_1[7] * bpair[gn_2 * 4 + 3] : 0.0f);
                            unsigned int pq_packed_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pq_out_1[_lp*2 + 0], pq_out_1[_lp*2+1 + 0]));
                                pq_packed_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_11 = static_cast<uint32_t>((unsigned long long)(s_abase + (unsigned int)(pq_srow_1 * 128 + (gn_2 * 16 + pq_scol_1) * 2 ^ (pq_srow_1 * 128 + (gn_2 * 16 + pq_scol_1) * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_1[3]))
                                : "memory");
                        }
                        #pragma unroll
                        for (int gn_3 = 2; gn_3 < 4; gn_3++) {
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(s_pw[gn_3 * 4]), "=r"(s_pw[gn_3 * 4 + 1]), "=r"(s_pw[gn_3 * 4 + 2]), "=r"(s_pw[gn_3 * 4 + 3])
                                : "r"(s_tile_inv + (unsigned int)((gn_3 * 16 + s_pcol) / 8 * 1024 + (32 + s_warp * 16 + s_prow) * 16 + (gn_3 * 16 + s_pcol) % 8 * 2))
                                : "memory");
                        }
                        int pq_r0_0 = 32 + s_warp * 16 + lane / 4;
                        int pq_r1_0 = pq_r0_0 + 8;
                        int pq_srow_0 = 32 + s_warp * 16 + (lane & 7) + (lane & 8);
                        int pq_scol_0 = (lane & 16) / 2;
                        #pragma unroll
                        for (int gn_4 = 2; gn_4 < 4; gn_4++) {
                            float pq_vals_2[8];
                            unsigned int pq_blk_2[4];
                            #pragma unroll
                            for (int w_3 = 0; w_3 < 4; w_3++) {
                                pq_blk_2[w_3] = s_pw[gn_4 * 4 + w_3];
                            }
                            #pragma unroll
                            for (int word_2 = 0; word_2 < 4; word_2++) {
                                unsigned int hi_bits_2 = pq_blk_2[word_2] >> 16;
                                __half lo_f16_2 = 0.0f;
                                __half hi_f16_2 = 0.0f;
                                unsigned int _bit_cast_src_12 = pq_blk_2[word_2];
                                lo_f16_2 = reinterpret_cast<__half*>(&_bit_cast_src_12)[0];
                                hi_f16_2 = reinterpret_cast<__half*>(&hi_bits_2)[0];
                                pq_vals_2[word_2 * 2] = (float)lo_f16_2;
                                pq_vals_2[word_2 * 2 + 1] = (float)hi_f16_2;
                            }
                            int pq_c0_2 = gn_4 * 16 + (lane & 3) * 2;
                            float pq_out_2[8];
                            pq_out_2[0] = ((pq_r0_0 > pq_c0_2) ? pq_vals_2[0] * bpair[gn_4 * 4] : 0.0f);
                            pq_out_2[1] = ((pq_r0_0 > pq_c0_2 + 1) ? pq_vals_2[1] * bpair[gn_4 * 4 + 1] : 0.0f);
                            pq_out_2[2] = ((pq_r1_0 > pq_c0_2) ? pq_vals_2[2] * bpair[gn_4 * 4] : 0.0f);
                            pq_out_2[3] = ((pq_r1_0 > pq_c0_2 + 1) ? pq_vals_2[3] * bpair[gn_4 * 4 + 1] : 0.0f);
                            pq_out_2[4] = ((pq_r0_0 > pq_c0_2 + 8) ? pq_vals_2[4] * bpair[gn_4 * 4 + 2] : 0.0f);
                            pq_out_2[5] = ((pq_r0_0 > pq_c0_2 + 9) ? pq_vals_2[5] * bpair[gn_4 * 4 + 3] : 0.0f);
                            pq_out_2[6] = ((pq_r1_0 > pq_c0_2 + 8) ? pq_vals_2[6] * bpair[gn_4 * 4 + 2] : 0.0f);
                            pq_out_2[7] = ((pq_r1_0 > pq_c0_2 + 9) ? pq_vals_2[7] * bpair[gn_4 * 4 + 3] : 0.0f);
                            unsigned int pq_packed_2[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pq_out_2[_lp*2 + 0], pq_out_2[_lp*2+1 + 0]));
                                pq_packed_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_13 = static_cast<uint32_t>((unsigned long long)(s_abase + (unsigned int)(pq_srow_0 * 128 + (gn_4 * 16 + pq_scol_0) * 2 ^ (pq_srow_0 * 128 + (gn_4 * 16 + pq_scol_0) * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_2[3]))
                                : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    asm volatile("" : "+f"(ks[0]) :: "memory");
                    asm volatile("" : "+f"(ks[1]) :: "memory");
                    asm volatile("" : "+f"(ks[2]) :: "memory");
                    asm volatile("" : "+f"(ks[3]) :: "memory");
                    asm volatile("" : "+f"(ks[4]) :: "memory");
                    asm volatile("" : "+f"(ks[5]) :: "memory");
                    asm volatile("" : "+f"(ks[6]) :: "memory");
                    asm volatile("" : "+f"(ks[7]) :: "memory");
                    asm volatile("" : "+f"(ks[8]) :: "memory");
                    asm volatile("" : "+f"(ks[9]) :: "memory");
                    asm volatile("" : "+f"(ks[10]) :: "memory");
                    asm volatile("" : "+f"(ks[11]) :: "memory");
                    asm volatile("" : "+f"(ks[12]) :: "memory");
                    asm volatile("" : "+f"(ks[13]) :: "memory");
                    asm volatile("" : "+f"(ks[14]) :: "memory");
                    asm volatile("" : "+f"(ks[15]) :: "memory");
                    asm volatile("" : "+f"(ks[16]) :: "memory");
                    asm volatile("" : "+f"(ks[17]) :: "memory");
                    asm volatile("" : "+f"(ks[18]) :: "memory");
                    asm volatile("" : "+f"(ks[19]) :: "memory");
                    asm volatile("" : "+f"(ks[20]) :: "memory");
                    asm volatile("" : "+f"(ks[21]) :: "memory");
                    asm volatile("" : "+f"(ks[22]) :: "memory");
                    asm volatile("" : "+f"(ks[23]) :: "memory");
                    asm volatile("" : "+f"(ks[24]) :: "memory");
                    asm volatile("" : "+f"(ks[25]) :: "memory");
                    asm volatile("" : "+f"(ks[26]) :: "memory");
                    asm volatile("" : "+f"(ks[27]) :: "memory");
                    asm volatile("" : "+f"(ks[28]) :: "memory");
                    asm volatile("" : "+f"(ks[29]) :: "memory");
                    asm volatile("" : "+f"(ks[30]) :: "memory");
                    asm volatile("" : "+f"(ks[31]) :: "memory");
                    asm volatile("" : "+r"(y_img[0]) :: "memory");
                    asm volatile("" : "+r"(y_img[1]) :: "memory");
                    asm volatile("" : "+r"(y_img[2]) :: "memory");
                    asm volatile("" : "+r"(y_img[3]) :: "memory");
                    asm volatile("" : "+r"(y_img[4]) :: "memory");
                    asm volatile("" : "+r"(y_img[5]) :: "memory");
                    asm volatile("" : "+r"(y_img[6]) :: "memory");
                    asm volatile("" : "+r"(y_img[7]) :: "memory");
                    asm volatile("" : "+r"(y_img[8]) :: "memory");
                    asm volatile("" : "+r"(y_img[9]) :: "memory");
                    asm volatile("" : "+r"(y_img[10]) :: "memory");
                    asm volatile("" : "+r"(y_img[11]) :: "memory");
                    asm volatile("" : "+r"(y_img[12]) :: "memory");
                    asm volatile("" : "+r"(y_img[13]) :: "memory");
                    asm volatile("" : "+r"(y_img[14]) :: "memory");
                    asm volatile("" : "+r"(y_img[15]) :: "memory");
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_14 = (((uint64_t)(((smem_abo_addr + ab_cur * 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t ab_desc = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_14 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_14);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n}\n"
                        : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                        : "r"(y_img[0]), "r"(y_img[1]), "r"(y_img[2]), "r"(y_img[3]), "l"(ab_desc)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                        : "r"(y_img[4]), "r"(y_img[(4) + 1]), "r"(y_img[(4) + 2]), "r"(y_img[(4) + 3]), "l"(ab_desc + 2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                        : "r"(y_img[8]), "r"(y_img[(8) + 1]), "r"(y_img[(8) + 2]), "r"(y_img[(8) + 3]), "l"(ab_desc + 4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                        : "r"(y_img[12]), "r"(y_img[(12) + 1]), "r"(y_img[(12) + 2]), "r"(y_img[(12) + 3]), "l"(ab_desc + 6)
                        : "memory");
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    state[0] = state[0] * gamma;
                    state[1] = state[1] * gamma;
                    state[2] = state[2] * gamma;
                    state[3] = state[3] * gamma;
                    state[4] = state[4] * gamma;
                    state[5] = state[5] * gamma;
                    state[6] = state[6] * gamma;
                    state[7] = state[7] * gamma;
                    state[8] = state[8] * gamma;
                    state[9] = state[9] * gamma;
                    state[10] = state[10] * gamma;
                    state[11] = state[11] * gamma;
                    state[12] = state[12] * gamma;
                    state[13] = state[13] * gamma;
                    state[14] = state[14] * gamma;
                    state[15] = state[15] * gamma;
                    state[16] = state[16] * gamma;
                    state[17] = state[17] * gamma;
                    state[18] = state[18] * gamma;
                    state[19] = state[19] * gamma;
                    state[20] = state[20] * gamma;
                    state[21] = state[21] * gamma;
                    state[22] = state[22] * gamma;
                    state[23] = state[23] * gamma;
                    state[24] = state[24] * gamma;
                    state[25] = state[25] * gamma;
                    state[26] = state[26] * gamma;
                    state[27] = state[27] * gamma;
                    state[28] = state[28] * gamma;
                    state[29] = state[29] * gamma;
                    state[30] = state[30] * gamma;
                    state[31] = state[31] * gamma;
                    state[32] = state[32] * gamma;
                    state[33] = state[33] * gamma;
                    state[34] = state[34] * gamma;
                    state[35] = state[35] * gamma;
                    state[36] = state[36] * gamma;
                    state[37] = state[37] * gamma;
                    state[38] = state[38] * gamma;
                    state[39] = state[39] * gamma;
                    state[40] = state[40] * gamma;
                    state[41] = state[41] * gamma;
                    state[42] = state[42] * gamma;
                    state[43] = state[43] * gamma;
                    state[44] = state[44] * gamma;
                    state[45] = state[45] * gamma;
                    state[46] = state[46] * gamma;
                    state[47] = state[47] * gamma;
                    state[48] = state[48] * gamma;
                    state[49] = state[49] * gamma;
                    state[50] = state[50] * gamma;
                    state[51] = state[51] * gamma;
                    state[52] = state[52] * gamma;
                    state[53] = state[53] * gamma;
                    state[54] = state[54] * gamma;
                    state[55] = state[55] * gamma;
                    state[56] = state[56] * gamma;
                    state[57] = state[57] * gamma;
                    state[58] = state[58] * gamma;
                    state[59] = state[59] * gamma;
                    state[60] = state[60] * gamma;
                    state[61] = state[61] * gamma;
                    state[62] = state[62] * gamma;
                    state[63] = state[63] * gamma;
                    asm volatile("" : "+f"(state[0]) :: "memory");
                    asm volatile("" : "+f"(state[1]) :: "memory");
                    asm volatile("" : "+f"(state[2]) :: "memory");
                    asm volatile("" : "+f"(state[3]) :: "memory");
                    asm volatile("" : "+f"(state[4]) :: "memory");
                    asm volatile("" : "+f"(state[5]) :: "memory");
                    asm volatile("" : "+f"(state[6]) :: "memory");
                    asm volatile("" : "+f"(state[7]) :: "memory");
                    asm volatile("" : "+f"(state[8]) :: "memory");
                    asm volatile("" : "+f"(state[9]) :: "memory");
                    asm volatile("" : "+f"(state[10]) :: "memory");
                    asm volatile("" : "+f"(state[11]) :: "memory");
                    asm volatile("" : "+f"(state[12]) :: "memory");
                    asm volatile("" : "+f"(state[13]) :: "memory");
                    asm volatile("" : "+f"(state[14]) :: "memory");
                    asm volatile("" : "+f"(state[15]) :: "memory");
                    asm volatile("" : "+f"(state[16]) :: "memory");
                    asm volatile("" : "+f"(state[17]) :: "memory");
                    asm volatile("" : "+f"(state[18]) :: "memory");
                    asm volatile("" : "+f"(state[19]) :: "memory");
                    asm volatile("" : "+f"(state[20]) :: "memory");
                    asm volatile("" : "+f"(state[21]) :: "memory");
                    asm volatile("" : "+f"(state[22]) :: "memory");
                    asm volatile("" : "+f"(state[23]) :: "memory");
                    asm volatile("" : "+f"(state[24]) :: "memory");
                    asm volatile("" : "+f"(state[25]) :: "memory");
                    asm volatile("" : "+f"(state[26]) :: "memory");
                    asm volatile("" : "+f"(state[27]) :: "memory");
                    asm volatile("" : "+f"(state[28]) :: "memory");
                    asm volatile("" : "+f"(state[29]) :: "memory");
                    asm volatile("" : "+f"(state[30]) :: "memory");
                    asm volatile("" : "+f"(state[31]) :: "memory");
                    asm volatile("" : "+f"(state[32]) :: "memory");
                    asm volatile("" : "+f"(state[33]) :: "memory");
                    asm volatile("" : "+f"(state[34]) :: "memory");
                    asm volatile("" : "+f"(state[35]) :: "memory");
                    asm volatile("" : "+f"(state[36]) :: "memory");
                    asm volatile("" : "+f"(state[37]) :: "memory");
                    asm volatile("" : "+f"(state[38]) :: "memory");
                    asm volatile("" : "+f"(state[39]) :: "memory");
                    asm volatile("" : "+f"(state[40]) :: "memory");
                    asm volatile("" : "+f"(state[41]) :: "memory");
                    asm volatile("" : "+f"(state[42]) :: "memory");
                    asm volatile("" : "+f"(state[43]) :: "memory");
                    asm volatile("" : "+f"(state[44]) :: "memory");
                    asm volatile("" : "+f"(state[45]) :: "memory");
                    asm volatile("" : "+f"(state[46]) :: "memory");
                    asm volatile("" : "+f"(state[47]) :: "memory");
                    asm volatile("" : "+f"(state[48]) :: "memory");
                    asm volatile("" : "+f"(state[49]) :: "memory");
                    asm volatile("" : "+f"(state[50]) :: "memory");
                    asm volatile("" : "+f"(state[51]) :: "memory");
                    asm volatile("" : "+f"(state[52]) :: "memory");
                    asm volatile("" : "+f"(state[53]) :: "memory");
                    asm volatile("" : "+f"(state[54]) :: "memory");
                    asm volatile("" : "+f"(state[55]) :: "memory");
                    asm volatile("" : "+f"(state[56]) :: "memory");
                    asm volatile("" : "+f"(state[57]) :: "memory");
                    asm volatile("" : "+f"(state[58]) :: "memory");
                    asm volatile("" : "+f"(state[59]) :: "memory");
                    asm volatile("" : "+f"(state[60]) :: "memory");
                    asm volatile("" : "+f"(state[61]) :: "memory");
                    asm volatile("" : "+f"(state[62]) :: "memory");
                    asm volatile("" : "+f"(state[63]) :: "memory");
                    asm volatile("" : "+f"(ks[0]) :: "memory");
                    asm volatile("" : "+f"(ks[1]) :: "memory");
                    asm volatile("" : "+f"(ks[2]) :: "memory");
                    asm volatile("" : "+f"(ks[3]) :: "memory");
                    asm volatile("" : "+f"(ks[4]) :: "memory");
                    asm volatile("" : "+f"(ks[5]) :: "memory");
                    asm volatile("" : "+f"(ks[6]) :: "memory");
                    asm volatile("" : "+f"(ks[7]) :: "memory");
                    asm volatile("" : "+f"(ks[8]) :: "memory");
                    asm volatile("" : "+f"(ks[9]) :: "memory");
                    asm volatile("" : "+f"(ks[10]) :: "memory");
                    asm volatile("" : "+f"(ks[11]) :: "memory");
                    asm volatile("" : "+f"(ks[12]) :: "memory");
                    asm volatile("" : "+f"(ks[13]) :: "memory");
                    asm volatile("" : "+f"(ks[14]) :: "memory");
                    asm volatile("" : "+f"(ks[15]) :: "memory");
                    asm volatile("" : "+f"(ks[16]) :: "memory");
                    asm volatile("" : "+f"(ks[17]) :: "memory");
                    asm volatile("" : "+f"(ks[18]) :: "memory");
                    asm volatile("" : "+f"(ks[19]) :: "memory");
                    asm volatile("" : "+f"(ks[20]) :: "memory");
                    asm volatile("" : "+f"(ks[21]) :: "memory");
                    asm volatile("" : "+f"(ks[22]) :: "memory");
                    asm volatile("" : "+f"(ks[23]) :: "memory");
                    asm volatile("" : "+f"(ks[24]) :: "memory");
                    asm volatile("" : "+f"(ks[25]) :: "memory");
                    asm volatile("" : "+f"(ks[26]) :: "memory");
                    asm volatile("" : "+f"(ks[27]) :: "memory");
                    asm volatile("" : "+f"(ks[28]) :: "memory");
                    asm volatile("" : "+f"(ks[29]) :: "memory");
                    asm volatile("" : "+f"(ks[30]) :: "memory");
                    asm volatile("" : "+f"(ks[31]) :: "memory");
                    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                    mbarrier_arrive(ab_empty_addr + (ab_cur) * 8);
                    float dpair[16];
                    #pragma unroll
                    for (int i_2 = 0; i_2 < 8; i_2++) {
                        #pragma unroll
                        for (int e_2 = 0; e_2 < 2; e_2++) {
                            dpair[2 * i_2 + e_2] = smem_gate[s_gb + 128 + i_2 * 8 + s_quad * 2 + e_2];
                        }
                    }
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 32; j_3++) {
                        ks[j_3] = ks[j_3] + bpair[j_3 / 4 * 2 + (j_3 & 1)] * vf[j_3];
                    }
                    unsigned int nv_img[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks[_lp*2 + 0], ks[_lp*2+1 + 0]));
                        nv_img[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 32; j_4++) {
                        ks[j_4] = ks[j_4] * dpair[j_4 / 4 * 2 + (j_4 & 1)];
                    }
                    unsigned int du_img[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks[_lp*2 + 0], ks[_lp*2+1 + 0]));
                        du_img[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int du_lo[16];
                    float du_hi_f[32];
                    #pragma unroll
                    for (int _pair = 0; _pair < 16; _pair++) {
                        (&du_hi_f[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(du_img[_pair]) << 16);
                        (&du_hi_f[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(du_img[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_5 = 0; j_5 < 32; j_5++) {
                        du_hi_f[j_5] = ks[j_5] - du_hi_f[j_5];
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(du_hi_f[_lp*2 + 0], du_hi_f[_lp*2+1 + 0]));
                        du_lo[_lp] = *(uint32_t*)&_bf2;
                    }
                    mbarrier_arrive(gate_empty_addr + (g_cur) * 8);
                    mbarrier_wait(qk_ready_addr + (qk_cur) * 8, _phase_qk_ready);
                    asm volatile("" : "+f"(state[0]) :: "memory");
                    asm volatile("" : "+f"(state[1]) :: "memory");
                    asm volatile("" : "+f"(state[2]) :: "memory");
                    asm volatile("" : "+f"(state[3]) :: "memory");
                    asm volatile("" : "+f"(state[4]) :: "memory");
                    asm volatile("" : "+f"(state[5]) :: "memory");
                    asm volatile("" : "+f"(state[6]) :: "memory");
                    asm volatile("" : "+f"(state[7]) :: "memory");
                    asm volatile("" : "+f"(state[8]) :: "memory");
                    asm volatile("" : "+f"(state[9]) :: "memory");
                    asm volatile("" : "+f"(state[10]) :: "memory");
                    asm volatile("" : "+f"(state[11]) :: "memory");
                    asm volatile("" : "+f"(state[12]) :: "memory");
                    asm volatile("" : "+f"(state[13]) :: "memory");
                    asm volatile("" : "+f"(state[14]) :: "memory");
                    asm volatile("" : "+f"(state[15]) :: "memory");
                    asm volatile("" : "+f"(state[16]) :: "memory");
                    asm volatile("" : "+f"(state[17]) :: "memory");
                    asm volatile("" : "+f"(state[18]) :: "memory");
                    asm volatile("" : "+f"(state[19]) :: "memory");
                    asm volatile("" : "+f"(state[20]) :: "memory");
                    asm volatile("" : "+f"(state[21]) :: "memory");
                    asm volatile("" : "+f"(state[22]) :: "memory");
                    asm volatile("" : "+f"(state[23]) :: "memory");
                    asm volatile("" : "+f"(state[24]) :: "memory");
                    asm volatile("" : "+f"(state[25]) :: "memory");
                    asm volatile("" : "+f"(state[26]) :: "memory");
                    asm volatile("" : "+f"(state[27]) :: "memory");
                    asm volatile("" : "+f"(state[28]) :: "memory");
                    asm volatile("" : "+f"(state[29]) :: "memory");
                    asm volatile("" : "+f"(state[30]) :: "memory");
                    asm volatile("" : "+f"(state[31]) :: "memory");
                    asm volatile("" : "+f"(state[32]) :: "memory");
                    asm volatile("" : "+f"(state[33]) :: "memory");
                    asm volatile("" : "+f"(state[34]) :: "memory");
                    asm volatile("" : "+f"(state[35]) :: "memory");
                    asm volatile("" : "+f"(state[36]) :: "memory");
                    asm volatile("" : "+f"(state[37]) :: "memory");
                    asm volatile("" : "+f"(state[38]) :: "memory");
                    asm volatile("" : "+f"(state[39]) :: "memory");
                    asm volatile("" : "+f"(state[40]) :: "memory");
                    asm volatile("" : "+f"(state[41]) :: "memory");
                    asm volatile("" : "+f"(state[42]) :: "memory");
                    asm volatile("" : "+f"(state[43]) :: "memory");
                    asm volatile("" : "+f"(state[44]) :: "memory");
                    asm volatile("" : "+f"(state[45]) :: "memory");
                    asm volatile("" : "+f"(state[46]) :: "memory");
                    asm volatile("" : "+f"(state[47]) :: "memory");
                    asm volatile("" : "+f"(state[48]) :: "memory");
                    asm volatile("" : "+f"(state[49]) :: "memory");
                    asm volatile("" : "+f"(state[50]) :: "memory");
                    asm volatile("" : "+f"(state[51]) :: "memory");
                    asm volatile("" : "+f"(state[52]) :: "memory");
                    asm volatile("" : "+f"(state[53]) :: "memory");
                    asm volatile("" : "+f"(state[54]) :: "memory");
                    asm volatile("" : "+f"(state[55]) :: "memory");
                    asm volatile("" : "+f"(state[56]) :: "memory");
                    asm volatile("" : "+f"(state[57]) :: "memory");
                    asm volatile("" : "+f"(state[58]) :: "memory");
                    asm volatile("" : "+f"(state[59]) :: "memory");
                    asm volatile("" : "+f"(state[60]) :: "memory");
                    asm volatile("" : "+f"(state[61]) :: "memory");
                    asm volatile("" : "+f"(state[62]) :: "memory");
                    asm volatile("" : "+f"(state[63]) :: "memory");
                    asm volatile("" : "+f"(q_state[0]) :: "memory");
                    asm volatile("" : "+f"(q_state[1]) :: "memory");
                    asm volatile("" : "+f"(q_state[2]) :: "memory");
                    asm volatile("" : "+f"(q_state[3]) :: "memory");
                    asm volatile("" : "+f"(q_state[4]) :: "memory");
                    asm volatile("" : "+f"(q_state[5]) :: "memory");
                    asm volatile("" : "+f"(q_state[6]) :: "memory");
                    asm volatile("" : "+f"(q_state[7]) :: "memory");
                    asm volatile("" : "+f"(q_state[8]) :: "memory");
                    asm volatile("" : "+f"(q_state[9]) :: "memory");
                    asm volatile("" : "+f"(q_state[10]) :: "memory");
                    asm volatile("" : "+f"(q_state[11]) :: "memory");
                    asm volatile("" : "+f"(q_state[12]) :: "memory");
                    asm volatile("" : "+f"(q_state[13]) :: "memory");
                    asm volatile("" : "+f"(q_state[14]) :: "memory");
                    asm volatile("" : "+f"(q_state[15]) :: "memory");
                    asm volatile("" : "+f"(q_state[16]) :: "memory");
                    asm volatile("" : "+f"(q_state[17]) :: "memory");
                    asm volatile("" : "+f"(q_state[18]) :: "memory");
                    asm volatile("" : "+f"(q_state[19]) :: "memory");
                    asm volatile("" : "+f"(q_state[20]) :: "memory");
                    asm volatile("" : "+f"(q_state[21]) :: "memory");
                    asm volatile("" : "+f"(q_state[22]) :: "memory");
                    asm volatile("" : "+f"(q_state[23]) :: "memory");
                    asm volatile("" : "+f"(q_state[24]) :: "memory");
                    asm volatile("" : "+f"(q_state[25]) :: "memory");
                    asm volatile("" : "+f"(q_state[26]) :: "memory");
                    asm volatile("" : "+f"(q_state[27]) :: "memory");
                    asm volatile("" : "+f"(q_state[28]) :: "memory");
                    asm volatile("" : "+f"(q_state[29]) :: "memory");
                    asm volatile("" : "+f"(q_state[30]) :: "memory");
                    asm volatile("" : "+f"(q_state[31]) :: "memory");
                    asm volatile("" : "+r"(nv_img[0]) :: "memory");
                    asm volatile("" : "+r"(nv_img[1]) :: "memory");
                    asm volatile("" : "+r"(nv_img[2]) :: "memory");
                    asm volatile("" : "+r"(nv_img[3]) :: "memory");
                    asm volatile("" : "+r"(nv_img[4]) :: "memory");
                    asm volatile("" : "+r"(nv_img[5]) :: "memory");
                    asm volatile("" : "+r"(nv_img[6]) :: "memory");
                    asm volatile("" : "+r"(nv_img[7]) :: "memory");
                    asm volatile("" : "+r"(nv_img[8]) :: "memory");
                    asm volatile("" : "+r"(nv_img[9]) :: "memory");
                    asm volatile("" : "+r"(nv_img[10]) :: "memory");
                    asm volatile("" : "+r"(nv_img[11]) :: "memory");
                    asm volatile("" : "+r"(nv_img[12]) :: "memory");
                    asm volatile("" : "+r"(nv_img[13]) :: "memory");
                    asm volatile("" : "+r"(nv_img[14]) :: "memory");
                    asm volatile("" : "+r"(nv_img[15]) :: "memory");
                    asm volatile("" : "+r"(du_img[0]) :: "memory");
                    asm volatile("" : "+r"(du_img[1]) :: "memory");
                    asm volatile("" : "+r"(du_img[2]) :: "memory");
                    asm volatile("" : "+r"(du_img[3]) :: "memory");
                    asm volatile("" : "+r"(du_img[4]) :: "memory");
                    asm volatile("" : "+r"(du_img[5]) :: "memory");
                    asm volatile("" : "+r"(du_img[6]) :: "memory");
                    asm volatile("" : "+r"(du_img[7]) :: "memory");
                    asm volatile("" : "+r"(du_img[8]) :: "memory");
                    asm volatile("" : "+r"(du_img[9]) :: "memory");
                    asm volatile("" : "+r"(du_img[10]) :: "memory");
                    asm volatile("" : "+r"(du_img[11]) :: "memory");
                    asm volatile("" : "+r"(du_img[12]) :: "memory");
                    asm volatile("" : "+r"(du_img[13]) :: "memory");
                    asm volatile("" : "+r"(du_img[14]) :: "memory");
                    asm volatile("" : "+r"(du_img[15]) :: "memory");
                    asm volatile("" : "+r"(du_lo[0]) :: "memory");
                    asm volatile("" : "+r"(du_lo[1]) :: "memory");
                    asm volatile("" : "+r"(du_lo[2]) :: "memory");
                    asm volatile("" : "+r"(du_lo[3]) :: "memory");
                    asm volatile("" : "+r"(du_lo[4]) :: "memory");
                    asm volatile("" : "+r"(du_lo[5]) :: "memory");
                    asm volatile("" : "+r"(du_lo[6]) :: "memory");
                    asm volatile("" : "+r"(du_lo[7]) :: "memory");
                    asm volatile("" : "+r"(du_lo[8]) :: "memory");
                    asm volatile("" : "+r"(du_lo[9]) :: "memory");
                    asm volatile("" : "+r"(du_lo[10]) :: "memory");
                    asm volatile("" : "+r"(du_lo[11]) :: "memory");
                    asm volatile("" : "+r"(du_lo[12]) :: "memory");
                    asm volatile("" : "+r"(du_lo[13]) :: "memory");
                    asm volatile("" : "+r"(du_lo[14]) :: "memory");
                    asm volatile("" : "+r"(du_lo[15]) :: "memory");
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_15 = (((uint64_t)(((smem_k_mn_addr + k_cur * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t kmn_desc = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_15 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_15);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_img[0]), "r"(du_img[1]), "r"(du_img[2]), "r"(du_img[3]), "l"(kmn_desc)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_img[4]), "r"(du_img[(4) + 1]), "r"(du_img[(4) + 2]), "r"(du_img[(4) + 3]), "l"(kmn_desc + 128)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_img[8]), "r"(du_img[(8) + 1]), "r"(du_img[(8) + 2]), "r"(du_img[(8) + 3]), "l"(kmn_desc + 256)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_img[12]), "r"(du_img[(12) + 1]), "r"(du_img[(12) + 2]), "r"(du_img[(12) + 3]), "l"(kmn_desc + 384)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_lo[0]), "r"(du_lo[1]), "r"(du_lo[2]), "r"(du_lo[3]), "l"(kmn_desc)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_lo[4]), "r"(du_lo[(4) + 1]), "r"(du_lo[(4) + 2]), "r"(du_lo[(4) + 3]), "l"(kmn_desc + 128)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_lo[8]), "r"(du_lo[(8) + 1]), "r"(du_lo[(8) + 2]), "r"(du_lo[(8) + 3]), "l"(kmn_desc + 256)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_lo[12]), "r"(du_lo[(12) + 1]), "r"(du_lo[(12) + 2]), "r"(du_lo[(12) + 3]), "l"(kmn_desc + 384)
                        : "memory");
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_16 = (((uint64_t)(((smem_qk_addr + qk_cur * 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t qk_desc = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_16 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_16);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                        : "r"(nv_img[0]), "r"(nv_img[1]), "r"(nv_img[2]), "r"(nv_img[3]), "l"(qk_desc)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                        : "r"(nv_img[4]), "r"(nv_img[(4) + 1]), "r"(nv_img[(4) + 2]), "r"(nv_img[(4) + 3]), "l"(qk_desc + 2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                        : "r"(nv_img[8]), "r"(nv_img[(8) + 1]), "r"(nv_img[(8) + 2]), "r"(nv_img[(8) + 3]), "l"(qk_desc + 4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                        : "r"(nv_img[12]), "r"(nv_img[(12) + 1]), "r"(nv_img[(12) + 2]), "r"(nv_img[(12) + 3]), "l"(qk_desc + 6)
                        : "memory");
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    asm volatile("" : "+f"(state[0]) :: "memory");
                    asm volatile("" : "+f"(state[1]) :: "memory");
                    asm volatile("" : "+f"(state[2]) :: "memory");
                    asm volatile("" : "+f"(state[3]) :: "memory");
                    asm volatile("" : "+f"(state[4]) :: "memory");
                    asm volatile("" : "+f"(state[5]) :: "memory");
                    asm volatile("" : "+f"(state[6]) :: "memory");
                    asm volatile("" : "+f"(state[7]) :: "memory");
                    asm volatile("" : "+f"(state[8]) :: "memory");
                    asm volatile("" : "+f"(state[9]) :: "memory");
                    asm volatile("" : "+f"(state[10]) :: "memory");
                    asm volatile("" : "+f"(state[11]) :: "memory");
                    asm volatile("" : "+f"(state[12]) :: "memory");
                    asm volatile("" : "+f"(state[13]) :: "memory");
                    asm volatile("" : "+f"(state[14]) :: "memory");
                    asm volatile("" : "+f"(state[15]) :: "memory");
                    asm volatile("" : "+f"(state[16]) :: "memory");
                    asm volatile("" : "+f"(state[17]) :: "memory");
                    asm volatile("" : "+f"(state[18]) :: "memory");
                    asm volatile("" : "+f"(state[19]) :: "memory");
                    asm volatile("" : "+f"(state[20]) :: "memory");
                    asm volatile("" : "+f"(state[21]) :: "memory");
                    asm volatile("" : "+f"(state[22]) :: "memory");
                    asm volatile("" : "+f"(state[23]) :: "memory");
                    asm volatile("" : "+f"(state[24]) :: "memory");
                    asm volatile("" : "+f"(state[25]) :: "memory");
                    asm volatile("" : "+f"(state[26]) :: "memory");
                    asm volatile("" : "+f"(state[27]) :: "memory");
                    asm volatile("" : "+f"(state[28]) :: "memory");
                    asm volatile("" : "+f"(state[29]) :: "memory");
                    asm volatile("" : "+f"(state[30]) :: "memory");
                    asm volatile("" : "+f"(state[31]) :: "memory");
                    asm volatile("" : "+f"(state[32]) :: "memory");
                    asm volatile("" : "+f"(state[33]) :: "memory");
                    asm volatile("" : "+f"(state[34]) :: "memory");
                    asm volatile("" : "+f"(state[35]) :: "memory");
                    asm volatile("" : "+f"(state[36]) :: "memory");
                    asm volatile("" : "+f"(state[37]) :: "memory");
                    asm volatile("" : "+f"(state[38]) :: "memory");
                    asm volatile("" : "+f"(state[39]) :: "memory");
                    asm volatile("" : "+f"(state[40]) :: "memory");
                    asm volatile("" : "+f"(state[41]) :: "memory");
                    asm volatile("" : "+f"(state[42]) :: "memory");
                    asm volatile("" : "+f"(state[43]) :: "memory");
                    asm volatile("" : "+f"(state[44]) :: "memory");
                    asm volatile("" : "+f"(state[45]) :: "memory");
                    asm volatile("" : "+f"(state[46]) :: "memory");
                    asm volatile("" : "+f"(state[47]) :: "memory");
                    asm volatile("" : "+f"(state[48]) :: "memory");
                    asm volatile("" : "+f"(state[49]) :: "memory");
                    asm volatile("" : "+f"(state[50]) :: "memory");
                    asm volatile("" : "+f"(state[51]) :: "memory");
                    asm volatile("" : "+f"(state[52]) :: "memory");
                    asm volatile("" : "+f"(state[53]) :: "memory");
                    asm volatile("" : "+f"(state[54]) :: "memory");
                    asm volatile("" : "+f"(state[55]) :: "memory");
                    asm volatile("" : "+f"(state[56]) :: "memory");
                    asm volatile("" : "+f"(state[57]) :: "memory");
                    asm volatile("" : "+f"(state[58]) :: "memory");
                    asm volatile("" : "+f"(state[59]) :: "memory");
                    asm volatile("" : "+f"(state[60]) :: "memory");
                    asm volatile("" : "+f"(state[61]) :: "memory");
                    asm volatile("" : "+f"(state[62]) :: "memory");
                    asm volatile("" : "+f"(state[63]) :: "memory");
                    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
                    mbarrier_arrive(load_k_empty_addr + (k_cur) * 8);
                    k_cur += 1;
                    if (k_cur == 3) { k_cur = 0; _phase_load_k_full ^= 1; }
                    v_cur += 1;
                    if (v_cur == 2) { v_cur = 0; _phase_load_v_full ^= 1; }
                    g_cur += 1;
                    if (g_cur == 5) { g_cur = 0; _phase_alpha_full ^= 1; }
                    ab_cur += 1;
                    if (ab_cur == 3) { ab_cur = 0; _phase_ab_ready ^= 1; }
                    qk_cur += 1;
                    if (qk_cur == 3) { qk_cur = 0; _phase_qk_ready ^= 1; }
                    cg = cg + 1;
                }
                asm volatile("" : "+f"(q_state[0]) :: "memory");
                asm volatile("" : "+f"(q_state[1]) :: "memory");
                asm volatile("" : "+f"(q_state[2]) :: "memory");
                asm volatile("" : "+f"(q_state[3]) :: "memory");
                asm volatile("" : "+f"(q_state[4]) :: "memory");
                asm volatile("" : "+f"(q_state[5]) :: "memory");
                asm volatile("" : "+f"(q_state[6]) :: "memory");
                asm volatile("" : "+f"(q_state[7]) :: "memory");
                asm volatile("" : "+f"(q_state[8]) :: "memory");
                asm volatile("" : "+f"(q_state[9]) :: "memory");
                asm volatile("" : "+f"(q_state[10]) :: "memory");
                asm volatile("" : "+f"(q_state[11]) :: "memory");
                asm volatile("" : "+f"(q_state[12]) :: "memory");
                asm volatile("" : "+f"(q_state[13]) :: "memory");
                asm volatile("" : "+f"(q_state[14]) :: "memory");
                asm volatile("" : "+f"(q_state[15]) :: "memory");
                asm volatile("" : "+f"(q_state[16]) :: "memory");
                asm volatile("" : "+f"(q_state[17]) :: "memory");
                asm volatile("" : "+f"(q_state[18]) :: "memory");
                asm volatile("" : "+f"(q_state[19]) :: "memory");
                asm volatile("" : "+f"(q_state[20]) :: "memory");
                asm volatile("" : "+f"(q_state[21]) :: "memory");
                asm volatile("" : "+f"(q_state[22]) :: "memory");
                asm volatile("" : "+f"(q_state[23]) :: "memory");
                asm volatile("" : "+f"(q_state[24]) :: "memory");
                asm volatile("" : "+f"(q_state[25]) :: "memory");
                asm volatile("" : "+f"(q_state[26]) :: "memory");
                asm volatile("" : "+f"(q_state[27]) :: "memory");
                asm volatile("" : "+f"(q_state[28]) :: "memory");
                asm volatile("" : "+f"(q_state[29]) :: "memory");
                asm volatile("" : "+f"(q_state[30]) :: "memory");
                asm volatile("" : "+f"(q_state[31]) :: "memory");
                asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                if (num_chunks != 0) {
                    int obt_warp_1 = warp_id_in_role;
                    int obt_mtx_1 = lane / 8;
                    int obt_row_1 = lane & 7;
                    mbarrier_wait(o_empty_addr, _phase_o_empty);
                    unsigned int obt_img_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_state[_lp*2 + 0], q_state[_lp*2+1 + 0]));
                        obt_img_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int obt_base_1 = smem_o_addr + 0 * 8192;
                    #pragma unroll
                    for (int tg_1 = 0; tg_1 < 4; tg_1++) {
                        int obt_vb_1 = obt_warp_1 * 16 + (obt_mtx_1 & 1) * 8;
                        int obt_tok_1 = tg_1 * 16 + obt_mtx_1 / 2 * 8 + obt_row_1;
                        uint32_t _stmatrix_addr_17 = static_cast<uint32_t>((unsigned long long)(obt_base_1 + (unsigned int)(obt_tok_1 * 128 + obt_vb_1 * 2 ^ (obt_tok_1 * 128 + obt_vb_1 * 2 >> 7 & 7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&obt_img_1[tg_1 * 4])), "r"(*reinterpret_cast<const uint32_t*>(&obt_img_1[tg_1 * 4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&obt_img_1[tg_1 * 4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&obt_img_1[tg_1 * 4 + 3]))
                            : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(o_ready_addr);
                    _phase_o_empty ^= 1;
                    mbarrier_arrive(qk_empty_addr + (((qk_cur + 2 >= 3) ? qk_cur + 2 - 3 : qk_cur + 2)) * 8);
                }
                int row_top = warp_id_in_role * 16 + (lane >> 2);
                long long base_top = output_head_base + (long long)row_top * 128;
                long long base_bot = base_top + 1024;
                long long lane_pair = (long long)((lane & 3) * 2);
                {
                    float2 _v2 = make_float2(state[0 + 0], state[0 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 0)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[2 + 0], state[2 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 0)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[4 + 0], state[4 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 8)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[6 + 0], state[6 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 8)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[8 + 0], state[8 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 16)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[10 + 0], state[10 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 16)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[12 + 0], state[12 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 24)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[14 + 0], state[14 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 24)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[16 + 0], state[16 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 32)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[18 + 0], state[18 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 32)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[20 + 0], state[20 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 40)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[22 + 0], state[22 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 40)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[24 + 0], state[24 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 48)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[26 + 0], state[26 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 48)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[28 + 0], state[28 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 56)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[30 + 0], state[30 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 56)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[32 + 0], state[32 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 64)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[34 + 0], state[34 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 64)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[36 + 0], state[36 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 72)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[38 + 0], state[38 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 72)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[40 + 0], state[40 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 80)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[42 + 0], state[42 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 80)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[44 + 0], state[44 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 88)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[46 + 0], state[46 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 88)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[48 + 0], state[48 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 96)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[50 + 0], state[50 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 96)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[52 + 0], state[52 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 104)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[54 + 0], state[54 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 104)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[56 + 0], state[56 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 112)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[58 + 0], state[58 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 112)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[60 + 0], state[60 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_top + lane_pair + 120)) + 0) = _v2;
                }
                {
                    float2 _v2 = make_float2(state[62 + 0], state[62 + 1]);
                    *reinterpret_cast<float2*>((output_state + (base_bot + lane_pair + 120)) + 0) = _v2;
                }
            }
        }
    }
    // ---- Role: aux ----
    if (warp >= 4 && warp <= 7) {
        { // aux_main
            unsigned int ra_magic = (unsigned int)((1048576 + num_householder - 1) / num_householder);
            asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
            float a_scale = ((scale != 0.0f) ? scale : 0.088388346f);
            int warp_id_in_role_1 = warp - 4;
            int a_warp = warp_id_in_role_1;
            int a_quad = lane & 3;
            int a_lane_row = lane >> 2;
            int a_row_base = a_warp * 16;
            int a_thread = a_warp * 32 + lane;
            unsigned int k_cur_1 = 0;
            unsigned int k_nxt = 0;
            unsigned int q_cur_1 = 0;
            unsigned int g_cur_1 = 0;
            unsigned int g_prev = 0;
            unsigned int kk_cur = 0;
            unsigned int kk_prev = 0;
            unsigned int qk_cur_1 = 0;
            float acc_kk[32] = {0};
            float acc_qk[32] = {0};
            float kk_hold[32] = {0};
            float qk_hold[32] = {0};
            int ag = 0;
            uint32_t smid_a;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(smid_a));
            unsigned int _phase_load_k_full_1 = 0;
            unsigned int _phase_load_q_full_1 = 0;
            unsigned int _phase_alpha_full_1 = 0;
            unsigned int _phase_qk_empty = 1;
            unsigned int _phase_ab_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_i_1 = (int)tile_1;
                int batch_idx_1 = tile_i_1 / num_heads;
                int head_idx_1 = tile_i_1 - batch_idx_1 * num_heads;
                int raw_start_1 = cu_seqlens[batch_idx_1];
                int raw_end_1 = cu_seqlens[batch_idx_1 + 1];
                int start_floor_1 = ((raw_start_1 > 0) ? raw_start_1 : 0);
                int batch_start_1 = ((start_floor_1 < total_tokens) ? start_floor_1 : total_tokens);
                int end_floor_1 = ((raw_end_1 > batch_start_1) ? raw_end_1 : batch_start_1);
                int batch_end_1 = ((end_floor_1 < total_tokens) ? end_floor_1 : total_tokens);
                int exp_len_1 = (batch_end_1 - batch_start_1) * num_householder;
                int num_chunks_1 = (exp_len_1 + 63) / 64;
                if (num_chunks_1 > 0) {
                    int a_one = 1;
                    int a_last = num_chunks_1 - 1;
                    mbarrier_wait(load_k_full_addr + (k_nxt) * 8, _phase_load_k_full_1);
                    if (a_one != 0) {
                        mbarrier_wait(load_q_full_addr + (q_cur_1) * 8, _phase_load_q_full_1);
                    }
                    asm volatile("" : "+f"(acc_kk[0]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[1]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[2]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[3]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[4]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[5]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[6]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[7]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[8]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[9]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[10]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[11]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[12]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[13]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[14]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[15]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[16]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[17]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[18]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[19]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[20]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[21]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[22]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[23]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[24]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[25]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[26]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[27]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[28]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[29]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[30]) :: "memory");
                    asm volatile("" : "+f"(acc_kk[31]) :: "memory");
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_0 = (((uint64_t)(((smem_k_addr + k_nxt * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t pa_lo = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
                    uint64_t _wgmma_desc_1 = (((uint64_t)(((smem_k_addr + k_nxt * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t pb_lo = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
                    uint64_t _wgmma_desc_2 = (((uint64_t)(((smem_k_addr + k_nxt * 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t pa_hi = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
                    uint64_t _wgmma_desc_3 = (((uint64_t)(((smem_k_addr + k_nxt * 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t pb_hi = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_3 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_3);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                        : "l"(pa_lo), "l"(pb_lo)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                        : "l"(pa_lo + 2), "l"(pb_lo + 2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                        : "l"(pa_lo + 4), "l"(pb_lo + 4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                        : "l"(pa_lo + 6), "l"(pb_lo + 6)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                        : "l"(pa_hi), "l"(pb_hi)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                        : "l"(pa_hi + 2), "l"(pb_hi + 2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                        : "l"(pa_hi + 4), "l"(pb_hi + 4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                        : "l"(pa_hi + 6), "l"(pb_hi + 6)
                        : "memory");
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    asm volatile("" : "+f"(acc_qk[0]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[1]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[2]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[3]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[4]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[5]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[6]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[7]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[8]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[9]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[10]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[11]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[12]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[13]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[14]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[15]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[16]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[17]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[18]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[19]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[20]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[21]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[22]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[23]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[24]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[25]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[26]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[27]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[28]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[29]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[30]) :: "memory");
                    asm volatile("" : "+f"(acc_qk[31]) :: "memory");
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_4 = (((uint64_t)(((smem_q_addr + q_cur_1 * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t pa_lo_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_4 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_4);
                    uint64_t _wgmma_desc_5 = (((uint64_t)(((smem_k_addr + k_nxt * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t pb_lo_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_5 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_5);
                    uint64_t _wgmma_desc_6 = (((uint64_t)(((smem_q_addr + q_cur_1 * 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t pa_hi_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_6 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_6);
                    uint64_t _wgmma_desc_7 = (((uint64_t)(((smem_k_addr + k_nxt * 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t pb_hi_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_7 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_7);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                        : "l"(pa_lo_0), "l"(pb_lo_0)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                        : "l"(pa_lo_0 + 2), "l"(pb_lo_0 + 2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                        : "l"(pa_lo_0 + 4), "l"(pb_lo_0 + 4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                        : "l"(pa_lo_0 + 6), "l"(pb_lo_0 + 6)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                        : "l"(pa_hi_0), "l"(pb_hi_0)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                        : "l"(pa_hi_0 + 2), "l"(pb_hi_0 + 2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                        : "l"(pa_hi_0 + 4), "l"(pb_hi_0 + 4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                        : "l"(pa_hi_0 + 6), "l"(pb_hi_0 + 6)
                        : "memory");
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    k_nxt += 1;
                    if (k_nxt == 3) { k_nxt = 0; _phase_load_k_full_1 ^= 1; }
                    #pragma unroll 1
                    for (int c = 0; c < a_last; c++) {
                        int a_has_i = 1;
                        int a_gb = (int)g_cur_1 * 384;
                        int a_gbp = a_gb;
                        unsigned int a_tile_cur = smem_ab_addr + kk_cur * 8192;
                        unsigned int a_tile_inv = a_tile_cur;
                        int bp_blk_1 = (int)((unsigned int)c * ra_magic / 1048576);
                        int bp_i_in_1 = c - bp_blk_1 * num_householder;
                        int bp_roff_1 = bp_i_in_1 * 64;
                        int bp_blk_t0_1 = batch_start_1 + bp_blk_1 * 64;
                        int bp_rem_tok_1 = batch_end_1 - bp_blk_t0_1;
                        int bp_tib_1 = ((bp_rem_tok_1 < 64) ? bp_rem_tok_1 : 64);
                        int bp_is_head_1 = ((bp_i_in_1 == 0) ? 1 : 0);
                        int bp_is_tail_1 = ((bp_i_in_1 == num_householder - 1) ? 1 : ((c == num_chunks_1 - 1) ? 1 : 0));
                        int cp_blk_1 = (int)((unsigned int)c * ra_magic / 1048576);
                        int cp_i_in_1 = c - cp_blk_1 * num_householder;
                        int cp_sub_1 = (int)((unsigned int)(cp_i_in_1 * 64) * ra_magic / 1048576);
                        int cp_tok_off_1 = cp_blk_1 * 64 + cp_sub_1;
                        int cp_rem_1 = cp_i_in_1 * 64 - cp_sub_1 * num_householder;
                        int cp_t0_1 = batch_start_1 + cp_tok_off_1;
                        int cp_b_1 = num_householder - 1 - cp_rem_1;
                        int cp_cnt_1 = (int)((unsigned int)(63 - cp_b_1) * ra_magic / 1048576) + 1;
                        int a_is_tail = bp_is_tail_1;
                        asm volatile("" : "+f"(acc_kk[0]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[1]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[2]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[3]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[4]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[5]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[6]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[7]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[8]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[9]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[10]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[11]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[12]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[13]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[14]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[15]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[16]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[17]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[18]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[19]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[20]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[21]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[22]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[23]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[24]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[25]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[26]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[27]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[28]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[29]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[30]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[31]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[0]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[1]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[2]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[3]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[4]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[5]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[6]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[7]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[8]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[9]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[10]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[11]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[12]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[13]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[14]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[15]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[16]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[17]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[18]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[19]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[20]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[21]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[22]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[23]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[24]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[25]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[26]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[27]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[28]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[29]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[30]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[31]) :: "memory");
                        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                        mbarrier_arrive(load_k_empty_addr + (k_cur_1) * 8);
                        k_cur_1 += 1;
                        if (k_cur_1 == 3) { k_cur_1 = 0; }
                        if (a_is_tail != 0) {
                            mbarrier_arrive(q_empty_addr + (q_cur_1) * 8);
                            q_cur_1 += 1;
                            if (q_cur_1 == 2) { q_cur_1 = 0; _phase_load_q_full_1 ^= 1; }
                        }
                        mbarrier_wait(alpha_full_addr + (g_cur_1) * 8, _phase_alpha_full_1);
                        mbarrier_wait(beta_full_addr + (g_cur_1) * 8, _phase_alpha_full_1);
                        float cs16[16];
                        #pragma unroll
                        for (int i_3 = 0; i_3 < 8; i_3++) {
                            #pragma unroll
                            for (int e_3 = 0; e_3 < 2; e_3++) {
                                cs16[2 * i_3 + e_3] = smem_gate[a_gb + 0 + i_3 * 8 + a_quad * 2 + e_3];
                            }
                        }
                        int qrow0 = a_row_base + a_lane_row;
                        int qrow1 = qrow0 + 8;
                        float ctr0 = smem_gate[a_gb + 192 + qrow0];
                        float ctr1 = smem_gate[a_gb + 192 + qrow1];
                        int rr0 = qrow0 * num_householder + num_householder - 1 - bp_roff_1;
                        int rr1 = qrow1 * num_householder + num_householder - 1 - bp_roff_1;
                        float qmask[32];
                        #pragma unroll
                        for (int j_6 = 0; j_6 < 32; j_6++) {
                            int col = j_6 / 4 * 8 + a_quad * 2 + (j_6 & 1);
                            int ci = j_6 / 4 * 2 + (j_6 & 1);
                            float ctr = (((j_6 & 2) == 0) ? ctr0 : ctr1);
                            int rr = (((j_6 & 2) == 0) ? rr0 : rr1);
                            int qrow = (((j_6 & 2) == 0) ? qrow0 : qrow1);
                            float _exp2_8 = approx_exp2(ctr - cs16[ci]);
                            float fac = _exp2_8 * a_scale;
                            int keep = ((qrow < bp_tib_1) ? ((col <= rr) ? 1 : 0) : 0);
                            qmask[j_6] = ((keep != 0) ? fac : 0.0f);
                        }
                        mbarrier_wait(qk_empty_addr + (qk_cur_1) * 8, _phase_qk_empty);
                        int stsm_row = a_row_base + (lane & 7) + (lane & 8);
                        int stsm_col = (lane & 16) / 2;
                        #pragma unroll
                        for (int rep = 0; rep < 4; rep++) {
                            float scaled[8];
                            #pragma unroll
                            for (int q = 0; q < 8; q++) {
                                scaled[q] = acc_qk[rep * 8 + q] * qmask[rep * 8 + q];
                            }
                            unsigned int words[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                                words[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_9 = static_cast<uint32_t>((unsigned long long)(smem_qk_addr + qk_cur_1 * 8192 + (unsigned int)(stsm_row * 128 + (rep * 16 + stsm_col) * 2 ^ (stsm_row * 128 + (rep * 16 + stsm_col) * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&words[0])), "r"(*reinterpret_cast<const uint32_t*>(&words[1])), "r"(*reinterpret_cast<const uint32_t*>(&words[2])), "r"(*reinterpret_cast<const uint32_t*>(&words[3]))
                                : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(qk_ready_addr + (qk_cur_1) * 8);
                        qk_cur_1 += 1;
                        if (qk_cur_1 == 3) { qk_cur_1 = 0; _phase_qk_empty ^= 1; }
                        mbarrier_wait(ab_empty_addr + (kk_cur) * 8, _phase_ab_empty);
                        int srow0 = a_row_base + a_lane_row;
                        int srow1 = srow0 + 8;
                        float sb0 = smem_gate[a_gb + 320 + srow0];
                        float sb1 = smem_gate[a_gb + 320 + srow1];
                        float scs0 = smem_gate[a_gb + 0 + srow0];
                        float scs1 = smem_gate[a_gb + 0 + srow1];
                        int seed_row = lane % 16;
                        int seed_col = lane / 16 * 8;
                        #pragma unroll
                        for (int gn_5 = 0; gn_5 < 4; gn_5++) {
                            float seed[8];
                            #pragma unroll
                            for (int q_1 = 0; q_1 < 8; q_1++) {
                                int sj = gn_5 * 8 + q_1;
                                int scol = sj / 4 * 8 + a_quad * 2 + (sj & 1);
                                int sci = sj / 4 * 2 + (sj & 1);
                                int srow = (((sj & 2) == 0) ? srow0 : srow1);
                                float sbeta = (((sj & 2) == 0) ? sb0 : sb1);
                                float scs = (((sj & 2) == 0) ? scs0 : scs1);
                                float _exp2_10 = approx_exp2(scs - cs16[sci]);
                                float sfac = _exp2_10 * sbeta;
                                seed[q_1] = ((srow > scol) ? acc_kk[sj] * sfac : 0.0f);
                            }
                            unsigned int packed[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(seed[_lp*2 + 0], seed[_lp*2+1 + 0]));
                                packed[_lp] = *(uint32_t*)&_h2;
                            }
                            uint32_t _stmatrix_addr_11 = static_cast<uint32_t>((unsigned long long)(a_tile_cur + (unsigned int)((gn_5 * 16 + seed_col) / 8 * 1024 + (a_row_base + seed_row) * 16 + (gn_5 * 16 + seed_col) % 8 * 2)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[3]))
                                : "memory");
                        }
                        __syncwarp();
                        if (a_has_i != 0) {
                            int diag_block = (a_row_base + (lane & 15)) / 8;
                            int lane_in_diag = lane & 7;
                            int diag_col_base = diag_block * 8;
                            float inv_row[8];
                            unsigned int words_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                                : "r"(a_tile_inv + (unsigned int)(diag_col_base / 8 * 1024 + (a_row_base + (lane & 15)) * 16 + diag_col_base % 8 * 2)));
                            #pragma unroll
                            for (int word_3 = 0; word_3 < 4; word_3++) {
                                unsigned int hi_bits_3 = words_1[word_3] >> 16;
                                __half lo_f16_3 = 0.0f;
                                __half hi_f16_3 = 0.0f;
                                unsigned int _bit_cast_src_12 = words_1[word_3];
                                lo_f16_3 = reinterpret_cast<__half*>(&_bit_cast_src_12)[0];
                                hi_f16_3 = reinterpret_cast<__half*>(&hi_bits_3)[0];
                                inv_row[word_3 * 2] = (float)lo_f16_3;
                                inv_row[word_3 * 2 + 1] = (float)hi_f16_3;
                            }
                            #pragma unroll
                            for (int diag_col = 0; diag_col < 8; diag_col++) {
                                if (lane_in_diag == diag_col) {
                                    inv_row[diag_col] = 1.0f;
                                } else if (lane_in_diag < diag_col) {
                                    inv_row[diag_col] = 0.0f;
                                }
                            }
                            int subgroup_base = lane - lane_in_diag;
                            #pragma unroll
                            for (int src_row = 0; src_row < 7; src_row++) {
                                float row_scale = -inv_row[src_row];
                                #pragma unroll
                                for (int prev_col = 0; prev_col < src_row; prev_col++) {
                                    int pivot_lane = subgroup_base + src_row;
                                    float pivot = __shfl_sync(0xFFFFFFFF, inv_row[prev_col], pivot_lane);
                                    if (lane_in_diag > src_row) {
                                        inv_row[prev_col] = __fmaf_rn(row_scale, pivot, inv_row[prev_col]);
                                    }
                                }
                                if (lane_in_diag > src_row) {
                                    inv_row[src_row] = row_scale;
                                }
                            }
                            unsigned int words_0[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(inv_row[_lp*2 + 0], inv_row[_lp*2+1 + 0]));
                                words_0[_lp] = *(uint32_t*)&_h2;
                            }
                            __syncwarp();
                            if (lane < 16) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(a_tile_inv + (unsigned int)(diag_col_base / 8 * 1024 + (a_row_base + (lane & 15)) * 16 + diag_col_base % 8 * 2)), "r"(words_0[0]), "r"(words_0[1]), "r"(words_0[2]), "r"(words_0[3]) : "memory");
                            }
                        }
                        __syncwarp();
                        if (a_has_i != 0) {
                            int inv16_row = lane & 7;
                            unsigned int d_frag[2];
                            unsigned int c_frag[1];
                            float dc_acc[4];
                            unsigned int dc_f16[2];
                            unsigned int a_frag[1];
                            float out_acc[4];
                            unsigned int out_f16[2];
                            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                                : "=r"(d_frag[0])
                                : "r"(a_tile_inv + (unsigned int)((a_warp * 16 + 8) / 8 * 1024 + (a_warp * 16 + 8 + inv16_row) * 16 + (a_warp * 16 + 8) % 8 * 2))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                                : "=r"(d_frag[1])
                                : "r"(a_tile_inv + (unsigned int)((a_warp * 16 + 8) / 8 * 1024 + (a_warp * 16 + 8 + inv16_row) * 16 + (a_warp * 16 + 8) % 8 * 2))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                                : "=r"(c_frag[0])
                                : "r"(a_tile_inv + (unsigned int)(a_warp * 16 / 8 * 1024 + (a_warp * 16 + 8 + inv16_row) * 16 + a_warp * 16 % 8 * 2))
                                : "memory");
                            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                                : "=f"(dc_acc[0]), "=f"(dc_acc[1]), "=f"(dc_acc[2]), "=f"(dc_acc[3])
                                : "r"(d_frag[0]), "r"(d_frag[1]), "r"(c_frag[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++) {
                                dc_acc[_ls] = dc_acc[_ls] * -1.0f;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(dc_acc[_lp*2 + 0], dc_acc[_lp*2+1 + 0]));
                                dc_f16[_lp] = *(uint32_t*)&_h2;
                            }
                            asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                                : "=r"(a_frag[0])
                                : "r"(a_tile_inv + (unsigned int)(a_warp * 16 / 8 * 1024 + (a_warp * 16 + inv16_row) * 16 + a_warp * 16 % 8 * 2))
                                : "memory");
                            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                                : "=f"(out_acc[0]), "=f"(out_acc[1]), "=f"(out_acc[2]), "=f"(out_acc[3])
                                : "r"(dc_f16[0]), "r"(dc_f16[1]), "r"(a_frag[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(out_acc[_lp*2 + 0], out_acc[_lp*2+1 + 0]));
                                out_f16[_lp] = *(uint32_t*)&_h2;
                            }
                            __syncwarp();
                            uint32_t _stmatrix_addr_13 = static_cast<uint32_t>((unsigned long long)(a_tile_inv + (unsigned int)(a_warp * 16 / 8 * 1024 + (a_warp * 16 + 8 + inv16_row) * 16 + a_warp * 16 % 8 * 2)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
                                :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&out_f16[0]))
                                : "memory");
                        }
                        mbarrier_wait(load_k_full_addr + (k_nxt) * 8, _phase_load_k_full_1);
                        if (a_is_tail != 0) {
                            mbarrier_wait(load_q_full_addr + (q_cur_1) * 8, _phase_load_q_full_1);
                        }
                        asm volatile("" : "+f"(acc_kk[0]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[1]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[2]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[3]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[4]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[5]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[6]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[7]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[8]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[9]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[10]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[11]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[12]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[13]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[14]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[15]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[16]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[17]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[18]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[19]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[20]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[21]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[22]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[23]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[24]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[25]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[26]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[27]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[28]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[29]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[30]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[31]) :: "memory");
                        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                        uint64_t _wgmma_desc_14 = (((uint64_t)(((smem_k_addr + k_nxt * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t pa_lo_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_14 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_14);
                        uint64_t _wgmma_desc_15 = (((uint64_t)(((smem_k_addr + k_nxt * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t pb_lo_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_15 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_15);
                        uint64_t _wgmma_desc_16 = (((uint64_t)(((smem_k_addr + k_nxt * 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t pa_hi_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_16 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_16);
                        uint64_t _wgmma_desc_17 = (((uint64_t)(((smem_k_addr + k_nxt * 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t pb_hi_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_17 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_17);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                            : "l"(pa_lo_1), "l"(pb_lo_1)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                            : "l"(pa_lo_1 + 2), "l"(pb_lo_1 + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                            : "l"(pa_lo_1 + 4), "l"(pb_lo_1 + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                            : "l"(pa_lo_1 + 6), "l"(pb_lo_1 + 6)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                            : "l"(pa_hi_1), "l"(pb_hi_1)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                            : "l"(pa_hi_1 + 2), "l"(pb_hi_1 + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                            : "l"(pa_hi_1 + 4), "l"(pb_hi_1 + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                            : "l"(pa_hi_1 + 6), "l"(pb_hi_1 + 6)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        asm volatile("" : "+f"(acc_qk[0]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[1]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[2]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[3]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[4]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[5]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[6]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[7]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[8]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[9]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[10]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[11]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[12]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[13]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[14]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[15]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[16]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[17]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[18]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[19]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[20]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[21]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[22]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[23]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[24]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[25]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[26]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[27]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[28]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[29]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[30]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[31]) :: "memory");
                        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                        uint64_t _wgmma_desc_18 = (((uint64_t)(((smem_q_addr + q_cur_1 * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t pa_lo_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_18 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_18);
                        uint64_t _wgmma_desc_19 = (((uint64_t)(((smem_k_addr + k_nxt * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t pb_lo_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_19 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_19);
                        uint64_t _wgmma_desc_20 = (((uint64_t)(((smem_q_addr + q_cur_1 * 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t pa_hi_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_20 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_20);
                        uint64_t _wgmma_desc_21 = (((uint64_t)(((smem_k_addr + k_nxt * 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t pb_hi_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_21 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_21);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                            : "l"(pa_lo_2), "l"(pb_lo_2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                            : "l"(pa_lo_2 + 2), "l"(pb_lo_2 + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                            : "l"(pa_lo_2 + 4), "l"(pb_lo_2 + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                            : "l"(pa_lo_2 + 6), "l"(pb_lo_2 + 6)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                            : "l"(pa_hi_2), "l"(pb_hi_2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                            : "l"(pa_hi_2 + 2), "l"(pb_hi_2 + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                            : "l"(pa_hi_2 + 4), "l"(pb_hi_2 + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                            : "l"(pa_hi_2 + 6), "l"(pb_hi_2 + 6)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        k_nxt += 1;
                        if (k_nxt == 3) { k_nxt = 0; _phase_load_k_full_1 ^= 1; }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (a_has_i != 0) {
                            if (a_thread < 64) {
                                int lane_row_0 = lane % 16;
                                int blk = a_thread / 32 * 32;
                                unsigned int d_frag_1[4];
                                unsigned int c_frag_2[4];
                                float dc_acc_3[8];
                                unsigned int dc_f16_4[4];
                                unsigned int a_frag_5[4];
                                float out_acc_6[8];
                                unsigned int out_f16_7[4];
                                #pragma unroll
                                for (int rep_1 = 0; rep_1 < 2; rep_1++) {
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                        : "=r"(d_frag_1[rep_1 * 2]), "=r"(d_frag_1[rep_1 * 2 + 1])
                                        : "r"(a_tile_inv + (unsigned int)((blk + 16 + rep_1 * 8) / 8 * 1024 + (blk + 16 + lane_row_0) * 16 + (blk + 16 + rep_1 * 8) % 8 * 2))
                                        : "memory");
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                        : "=r"(c_frag_2[rep_1 * 2]), "=r"(c_frag_2[rep_1 * 2 + 1])
                                        : "r"(a_tile_inv + (unsigned int)((blk + rep_1 * 8) / 8 * 1024 + (blk + 16 + lane_row_0) * 16 + (blk + rep_1 * 8) % 8 * 2))
                                        : "memory");
                                }
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                    : "=f"(dc_acc_3[0]), "=f"(dc_acc_3[1]), "=f"(dc_acc_3[2]), "=f"(dc_acc_3[3])
                                    : "r"(d_frag_1[0]), "r"(d_frag_1[1]), "r"(d_frag_1[2]), "r"(d_frag_1[3]), "r"(c_frag_2[0]), "r"(c_frag_2[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                    : "=f"(dc_acc_3[4]), "=f"(dc_acc_3[(4) + 1]), "=f"(dc_acc_3[(4) + 2]), "=f"(dc_acc_3[(4) + 3])
                                    : "r"(d_frag_1[0]), "r"(d_frag_1[1]), "r"(d_frag_1[2]), "r"(d_frag_1[3]), "r"(c_frag_2[2]), "r"(c_frag_2[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++) {
                                    dc_acc_3[_ls] = dc_acc_3[_ls] * -1.0f;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(dc_acc_3[_lp*2 + 0], dc_acc_3[_lp*2+1 + 0]));
                                    dc_f16_4[_lp] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int rep_2 = 0; rep_2 < 2; rep_2++) {
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                        : "=r"(a_frag_5[rep_2 * 2]), "=r"(a_frag_5[rep_2 * 2 + 1])
                                        : "r"(a_tile_inv + (unsigned int)((blk + rep_2 * 8) / 8 * 1024 + (blk + lane_row_0) * 16 + (blk + rep_2 * 8) % 8 * 2))
                                        : "memory");
                                }
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                    : "=f"(out_acc_6[0]), "=f"(out_acc_6[1]), "=f"(out_acc_6[2]), "=f"(out_acc_6[3])
                                    : "r"(dc_f16_4[0]), "r"(dc_f16_4[1]), "r"(dc_f16_4[2]), "r"(dc_f16_4[3]), "r"(a_frag_5[0]), "r"(a_frag_5[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                    : "=f"(out_acc_6[4]), "=f"(out_acc_6[(4) + 1]), "=f"(out_acc_6[(4) + 2]), "=f"(out_acc_6[(4) + 3])
                                    : "r"(dc_f16_4[0]), "r"(dc_f16_4[1]), "r"(dc_f16_4[2]), "r"(dc_f16_4[3]), "r"(a_frag_5[2]), "r"(a_frag_5[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(out_acc_6[_lp*2 + 0], out_acc_6[_lp*2+1 + 0]));
                                    out_f16_7[_lp] = *(uint32_t*)&_h2;
                                }
                                __syncwarp();
                                #pragma unroll
                                for (int rep_3 = 0; rep_3 < 2; rep_3++) {
                                    uint32_t _stmatrix_addr_22 = static_cast<uint32_t>((unsigned long long)(a_tile_inv + (unsigned int)((blk + rep_3 * 8) / 8 * 1024 + (blk + 16 + lane_row_0) * 16 + (blk + rep_3 * 8) % 8 * 2)));
                                    asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                                        :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&out_f16_7[rep_3 * 2])), "r"(*reinterpret_cast<const uint32_t*>(&out_f16_7[rep_3 * 2 + 1]))
                                        : "memory");
                                }
                            }
                        }
                        if (a_has_i != 0) {
                            mbarrier_arrive(ab_ready_addr + (kk_cur) * 8);
                            mbarrier_arrive(gate_empty_addr + (g_cur_1) * 8);
                        }
                        kk_cur += 1;
                        if (kk_cur == 3) { kk_cur = 0; _phase_ab_empty ^= 1; }
                        g_cur_1 += 1;
                        if (g_cur_1 == 5) { g_cur_1 = 0; _phase_alpha_full_1 ^= 1; }
                        ag = ag + 1;
                    }
                    if (num_chunks_1 > 0) {
                        int a_has_i_1 = 1;
                        int a_gb_1 = (int)g_cur_1 * 384;
                        int a_gbp_1 = a_gb_1;
                        unsigned int a_tile_cur_1 = smem_ab_addr + kk_cur * 8192;
                        unsigned int a_tile_inv_1 = a_tile_cur_1;
                        int bp_blk_2 = (int)((unsigned int)a_last * ra_magic / 1048576);
                        int bp_i_in_2 = a_last - bp_blk_2 * num_householder;
                        int bp_roff_2 = bp_i_in_2 * 64;
                        int bp_blk_t0_2 = batch_start_1 + bp_blk_2 * 64;
                        int bp_rem_tok_2 = batch_end_1 - bp_blk_t0_2;
                        int bp_tib_2 = ((bp_rem_tok_2 < 64) ? bp_rem_tok_2 : 64);
                        int bp_is_head_2 = ((bp_i_in_2 == 0) ? 1 : 0);
                        int bp_is_tail_2 = ((bp_i_in_2 == num_householder - 1) ? 1 : ((a_last == num_chunks_1 - 1) ? 1 : 0));
                        int cp_blk_2 = (int)((unsigned int)a_last * ra_magic / 1048576);
                        int cp_i_in_2 = a_last - cp_blk_2 * num_householder;
                        int cp_sub_2 = (int)((unsigned int)(cp_i_in_2 * 64) * ra_magic / 1048576);
                        int cp_tok_off_2 = cp_blk_2 * 64 + cp_sub_2;
                        int cp_rem_2 = cp_i_in_2 * 64 - cp_sub_2 * num_householder;
                        int cp_t0_2 = batch_start_1 + cp_tok_off_2;
                        int cp_b_2 = num_householder - 1 - cp_rem_2;
                        int cp_cnt_2 = (int)((unsigned int)(63 - cp_b_2) * ra_magic / 1048576) + 1;
                        int a_is_tail_1 = bp_is_tail_2;
                        asm volatile("" : "+f"(acc_kk[0]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[1]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[2]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[3]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[4]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[5]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[6]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[7]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[8]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[9]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[10]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[11]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[12]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[13]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[14]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[15]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[16]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[17]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[18]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[19]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[20]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[21]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[22]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[23]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[24]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[25]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[26]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[27]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[28]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[29]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[30]) :: "memory");
                        asm volatile("" : "+f"(acc_kk[31]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[0]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[1]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[2]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[3]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[4]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[5]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[6]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[7]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[8]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[9]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[10]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[11]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[12]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[13]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[14]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[15]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[16]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[17]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[18]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[19]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[20]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[21]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[22]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[23]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[24]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[25]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[26]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[27]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[28]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[29]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[30]) :: "memory");
                        asm volatile("" : "+f"(acc_qk[31]) :: "memory");
                        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                        mbarrier_arrive(load_k_empty_addr + (k_cur_1) * 8);
                        k_cur_1 += 1;
                        if (k_cur_1 == 3) { k_cur_1 = 0; }
                        if (a_is_tail_1 != 0) {
                            mbarrier_arrive(q_empty_addr + (q_cur_1) * 8);
                            q_cur_1 += 1;
                            if (q_cur_1 == 2) { q_cur_1 = 0; _phase_load_q_full_1 ^= 1; }
                        }
                        mbarrier_wait(alpha_full_addr + (g_cur_1) * 8, _phase_alpha_full_1);
                        mbarrier_wait(beta_full_addr + (g_cur_1) * 8, _phase_alpha_full_1);
                        float cs16_1[16];
                        #pragma unroll
                        for (int i_4 = 0; i_4 < 8; i_4++) {
                            #pragma unroll
                            for (int e_4 = 0; e_4 < 2; e_4++) {
                                cs16_1[2 * i_4 + e_4] = smem_gate[a_gb_1 + 0 + i_4 * 8 + a_quad * 2 + e_4];
                            }
                        }
                        int qrow0_1 = a_row_base + a_lane_row;
                        int qrow1_1 = qrow0_1 + 8;
                        float ctr0_1 = smem_gate[a_gb_1 + 192 + qrow0_1];
                        float ctr1_1 = smem_gate[a_gb_1 + 192 + qrow1_1];
                        int rr0_1 = qrow0_1 * num_householder + num_householder - 1 - bp_roff_2;
                        int rr1_1 = qrow1_1 * num_householder + num_householder - 1 - bp_roff_2;
                        float qmask_1[32];
                        #pragma unroll
                        for (int j_7 = 0; j_7 < 32; j_7++) {
                            int col_1 = j_7 / 4 * 8 + a_quad * 2 + (j_7 & 1);
                            int ci_1 = j_7 / 4 * 2 + (j_7 & 1);
                            float ctr_1 = (((j_7 & 2) == 0) ? ctr0_1 : ctr1_1);
                            int rr_1 = (((j_7 & 2) == 0) ? rr0_1 : rr1_1);
                            int qrow_1 = (((j_7 & 2) == 0) ? qrow0_1 : qrow1_1);
                            float _exp2_23 = approx_exp2(ctr_1 - cs16_1[ci_1]);
                            float fac_1 = _exp2_23 * a_scale;
                            int keep_1 = ((qrow_1 < bp_tib_2) ? ((col_1 <= rr_1) ? 1 : 0) : 0);
                            qmask_1[j_7] = ((keep_1 != 0) ? fac_1 : 0.0f);
                        }
                        mbarrier_wait(qk_empty_addr + (qk_cur_1) * 8, _phase_qk_empty);
                        int stsm_row_1 = a_row_base + (lane & 7) + (lane & 8);
                        int stsm_col_1 = (lane & 16) / 2;
                        #pragma unroll
                        for (int rep_4 = 0; rep_4 < 4; rep_4++) {
                            float scaled_1[8];
                            #pragma unroll
                            for (int q_2 = 0; q_2 < 8; q_2++) {
                                scaled_1[q_2] = acc_qk[rep_4 * 8 + q_2] * qmask_1[rep_4 * 8 + q_2];
                            }
                            unsigned int words_2[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_1[_lp*2 + 0], scaled_1[_lp*2+1 + 0]));
                                words_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_24 = static_cast<uint32_t>((unsigned long long)(smem_qk_addr + qk_cur_1 * 8192 + (unsigned int)(stsm_row_1 * 128 + (rep_4 * 16 + stsm_col_1) * 2 ^ (stsm_row_1 * 128 + (rep_4 * 16 + stsm_col_1) * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_24), "r"(*reinterpret_cast<const uint32_t*>(&words_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&words_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&words_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&words_2[3]))
                                : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(qk_ready_addr + (qk_cur_1) * 8);
                        qk_cur_1 += 1;
                        if (qk_cur_1 == 3) { qk_cur_1 = 0; _phase_qk_empty ^= 1; }
                        mbarrier_wait(ab_empty_addr + (kk_cur) * 8, _phase_ab_empty);
                        int srow0_1 = a_row_base + a_lane_row;
                        int srow1_1 = srow0_1 + 8;
                        float sb0_1 = smem_gate[a_gb_1 + 320 + srow0_1];
                        float sb1_1 = smem_gate[a_gb_1 + 320 + srow1_1];
                        float scs0_1 = smem_gate[a_gb_1 + 0 + srow0_1];
                        float scs1_1 = smem_gate[a_gb_1 + 0 + srow1_1];
                        int seed_row_1 = lane % 16;
                        int seed_col_1 = lane / 16 * 8;
                        #pragma unroll
                        for (int gn_6 = 0; gn_6 < 4; gn_6++) {
                            float seed_1[8];
                            #pragma unroll
                            for (int q_3 = 0; q_3 < 8; q_3++) {
                                int sj_1 = gn_6 * 8 + q_3;
                                int scol_1 = sj_1 / 4 * 8 + a_quad * 2 + (sj_1 & 1);
                                int sci_1 = sj_1 / 4 * 2 + (sj_1 & 1);
                                int srow_1 = (((sj_1 & 2) == 0) ? srow0_1 : srow1_1);
                                float sbeta_1 = (((sj_1 & 2) == 0) ? sb0_1 : sb1_1);
                                float scs_1 = (((sj_1 & 2) == 0) ? scs0_1 : scs1_1);
                                float _exp2_25 = approx_exp2(scs_1 - cs16_1[sci_1]);
                                float sfac_1 = _exp2_25 * sbeta_1;
                                seed_1[q_3] = ((srow_1 > scol_1) ? acc_kk[sj_1] * sfac_1 : 0.0f);
                            }
                            unsigned int packed_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(seed_1[_lp*2 + 0], seed_1[_lp*2+1 + 0]));
                                packed_1[_lp] = *(uint32_t*)&_h2;
                            }
                            uint32_t _stmatrix_addr_26 = static_cast<uint32_t>((unsigned long long)(a_tile_cur_1 + (unsigned int)((gn_6 * 16 + seed_col_1) / 8 * 1024 + (a_row_base + seed_row_1) * 16 + (gn_6 * 16 + seed_col_1) % 8 * 2)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_26), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[3]))
                                : "memory");
                        }
                        __syncwarp();
                        if (a_has_i_1 != 0) {
                            int diag_block_1 = (a_row_base + (lane & 15)) / 8;
                            int lane_in_diag_1 = lane & 7;
                            int diag_col_base_1 = diag_block_1 * 8;
                            float inv_row_1[8];
                            unsigned int words_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 3]))
                                : "r"(a_tile_inv_1 + (unsigned int)(diag_col_base_1 / 8 * 1024 + (a_row_base + (lane & 15)) * 16 + diag_col_base_1 % 8 * 2)));
                            #pragma unroll
                            for (int word_4 = 0; word_4 < 4; word_4++) {
                                unsigned int hi_bits_4 = words_3[word_4] >> 16;
                                __half lo_f16_4 = 0.0f;
                                __half hi_f16_4 = 0.0f;
                                unsigned int _bit_cast_src_27 = words_3[word_4];
                                lo_f16_4 = reinterpret_cast<__half*>(&_bit_cast_src_27)[0];
                                hi_f16_4 = reinterpret_cast<__half*>(&hi_bits_4)[0];
                                inv_row_1[word_4 * 2] = (float)lo_f16_4;
                                inv_row_1[word_4 * 2 + 1] = (float)hi_f16_4;
                            }
                            #pragma unroll
                            for (int diag_col_1 = 0; diag_col_1 < 8; diag_col_1++) {
                                if (lane_in_diag_1 == diag_col_1) {
                                    inv_row_1[diag_col_1] = 1.0f;
                                } else if (lane_in_diag_1 < diag_col_1) {
                                    inv_row_1[diag_col_1] = 0.0f;
                                }
                            }
                            int subgroup_base_1 = lane - lane_in_diag_1;
                            #pragma unroll
                            for (int src_row_1 = 0; src_row_1 < 7; src_row_1++) {
                                float row_scale_1 = -inv_row_1[src_row_1];
                                #pragma unroll
                                for (int prev_col_1 = 0; prev_col_1 < src_row_1; prev_col_1++) {
                                    int pivot_lane_1 = subgroup_base_1 + src_row_1;
                                    float pivot_1 = __shfl_sync(0xFFFFFFFF, inv_row_1[prev_col_1], pivot_lane_1);
                                    if (lane_in_diag_1 > src_row_1) {
                                        inv_row_1[prev_col_1] = __fmaf_rn(row_scale_1, pivot_1, inv_row_1[prev_col_1]);
                                    }
                                }
                                if (lane_in_diag_1 > src_row_1) {
                                    inv_row_1[src_row_1] = row_scale_1;
                                }
                            }
                            unsigned int words_0_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(inv_row_1[_lp*2 + 0], inv_row_1[_lp*2+1 + 0]));
                                words_0_1[_lp] = *(uint32_t*)&_h2;
                            }
                            __syncwarp();
                            if (lane < 16) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(a_tile_inv_1 + (unsigned int)(diag_col_base_1 / 8 * 1024 + (a_row_base + (lane & 15)) * 16 + diag_col_base_1 % 8 * 2)), "r"(words_0_1[0]), "r"(words_0_1[1]), "r"(words_0_1[2]), "r"(words_0_1[3]) : "memory");
                            }
                        }
                        __syncwarp();
                        if (a_has_i_1 != 0) {
                            int inv16_row_1 = lane & 7;
                            unsigned int d_frag_2[2];
                            unsigned int c_frag_1[1];
                            float dc_acc_1[4];
                            unsigned int dc_f16_1[2];
                            unsigned int a_frag_1[1];
                            float out_acc_1[4];
                            unsigned int out_f16_1[2];
                            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                                : "=r"(d_frag_2[0])
                                : "r"(a_tile_inv_1 + (unsigned int)((a_warp * 16 + 8) / 8 * 1024 + (a_warp * 16 + 8 + inv16_row_1) * 16 + (a_warp * 16 + 8) % 8 * 2))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                                : "=r"(d_frag_2[1])
                                : "r"(a_tile_inv_1 + (unsigned int)((a_warp * 16 + 8) / 8 * 1024 + (a_warp * 16 + 8 + inv16_row_1) * 16 + (a_warp * 16 + 8) % 8 * 2))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                                : "=r"(c_frag_1[0])
                                : "r"(a_tile_inv_1 + (unsigned int)(a_warp * 16 / 8 * 1024 + (a_warp * 16 + 8 + inv16_row_1) * 16 + a_warp * 16 % 8 * 2))
                                : "memory");
                            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                                : "=f"(dc_acc_1[0]), "=f"(dc_acc_1[1]), "=f"(dc_acc_1[2]), "=f"(dc_acc_1[3])
                                : "r"(d_frag_2[0]), "r"(d_frag_2[1]), "r"(c_frag_1[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++) {
                                dc_acc_1[_ls] = dc_acc_1[_ls] * -1.0f;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(dc_acc_1[_lp*2 + 0], dc_acc_1[_lp*2+1 + 0]));
                                dc_f16_1[_lp] = *(uint32_t*)&_h2;
                            }
                            asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                                : "=r"(a_frag_1[0])
                                : "r"(a_tile_inv_1 + (unsigned int)(a_warp * 16 / 8 * 1024 + (a_warp * 16 + inv16_row_1) * 16 + a_warp * 16 % 8 * 2))
                                : "memory");
                            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                                : "=f"(out_acc_1[0]), "=f"(out_acc_1[1]), "=f"(out_acc_1[2]), "=f"(out_acc_1[3])
                                : "r"(dc_f16_1[0]), "r"(dc_f16_1[1]), "r"(a_frag_1[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(out_acc_1[_lp*2 + 0], out_acc_1[_lp*2+1 + 0]));
                                out_f16_1[_lp] = *(uint32_t*)&_h2;
                            }
                            __syncwarp();
                            uint32_t _stmatrix_addr_28 = static_cast<uint32_t>((unsigned long long)(a_tile_inv_1 + (unsigned int)(a_warp * 16 / 8 * 1024 + (a_warp * 16 + 8 + inv16_row_1) * 16 + a_warp * 16 % 8 * 2)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
                                :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&out_f16_1[0]))
                                : "memory");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (a_has_i_1 != 0) {
                            if (a_thread < 64) {
                                int lane_row_0_1 = lane % 16;
                                int blk_1 = a_thread / 32 * 32;
                                unsigned int d_frag_1_1[4];
                                unsigned int c_frag_2_1[4];
                                float dc_acc_3_1[8];
                                unsigned int dc_f16_4_1[4];
                                unsigned int a_frag_5_1[4];
                                float out_acc_6_1[8];
                                unsigned int out_f16_7_1[4];
                                #pragma unroll
                                for (int rep_5 = 0; rep_5 < 2; rep_5++) {
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                        : "=r"(d_frag_1_1[rep_5 * 2]), "=r"(d_frag_1_1[rep_5 * 2 + 1])
                                        : "r"(a_tile_inv_1 + (unsigned int)((blk_1 + 16 + rep_5 * 8) / 8 * 1024 + (blk_1 + 16 + lane_row_0_1) * 16 + (blk_1 + 16 + rep_5 * 8) % 8 * 2))
                                        : "memory");
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                        : "=r"(c_frag_2_1[rep_5 * 2]), "=r"(c_frag_2_1[rep_5 * 2 + 1])
                                        : "r"(a_tile_inv_1 + (unsigned int)((blk_1 + rep_5 * 8) / 8 * 1024 + (blk_1 + 16 + lane_row_0_1) * 16 + (blk_1 + rep_5 * 8) % 8 * 2))
                                        : "memory");
                                }
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                    : "=f"(dc_acc_3_1[0]), "=f"(dc_acc_3_1[1]), "=f"(dc_acc_3_1[2]), "=f"(dc_acc_3_1[3])
                                    : "r"(d_frag_1_1[0]), "r"(d_frag_1_1[1]), "r"(d_frag_1_1[2]), "r"(d_frag_1_1[3]), "r"(c_frag_2_1[0]), "r"(c_frag_2_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                    : "=f"(dc_acc_3_1[4]), "=f"(dc_acc_3_1[(4) + 1]), "=f"(dc_acc_3_1[(4) + 2]), "=f"(dc_acc_3_1[(4) + 3])
                                    : "r"(d_frag_1_1[0]), "r"(d_frag_1_1[1]), "r"(d_frag_1_1[2]), "r"(d_frag_1_1[3]), "r"(c_frag_2_1[2]), "r"(c_frag_2_1[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++) {
                                    dc_acc_3_1[_ls] = dc_acc_3_1[_ls] * -1.0f;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(dc_acc_3_1[_lp*2 + 0], dc_acc_3_1[_lp*2+1 + 0]));
                                    dc_f16_4_1[_lp] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int rep_6 = 0; rep_6 < 2; rep_6++) {
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                        : "=r"(a_frag_5_1[rep_6 * 2]), "=r"(a_frag_5_1[rep_6 * 2 + 1])
                                        : "r"(a_tile_inv_1 + (unsigned int)((blk_1 + rep_6 * 8) / 8 * 1024 + (blk_1 + lane_row_0_1) * 16 + (blk_1 + rep_6 * 8) % 8 * 2))
                                        : "memory");
                                }
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                    : "=f"(out_acc_6_1[0]), "=f"(out_acc_6_1[1]), "=f"(out_acc_6_1[2]), "=f"(out_acc_6_1[3])
                                    : "r"(dc_f16_4_1[0]), "r"(dc_f16_4_1[1]), "r"(dc_f16_4_1[2]), "r"(dc_f16_4_1[3]), "r"(a_frag_5_1[0]), "r"(a_frag_5_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                    : "=f"(out_acc_6_1[4]), "=f"(out_acc_6_1[(4) + 1]), "=f"(out_acc_6_1[(4) + 2]), "=f"(out_acc_6_1[(4) + 3])
                                    : "r"(dc_f16_4_1[0]), "r"(dc_f16_4_1[1]), "r"(dc_f16_4_1[2]), "r"(dc_f16_4_1[3]), "r"(a_frag_5_1[2]), "r"(a_frag_5_1[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(out_acc_6_1[_lp*2 + 0], out_acc_6_1[_lp*2+1 + 0]));
                                    out_f16_7_1[_lp] = *(uint32_t*)&_h2;
                                }
                                __syncwarp();
                                #pragma unroll
                                for (int rep_7 = 0; rep_7 < 2; rep_7++) {
                                    uint32_t _stmatrix_addr_29 = static_cast<uint32_t>((unsigned long long)(a_tile_inv_1 + (unsigned int)((blk_1 + rep_7 * 8) / 8 * 1024 + (blk_1 + 16 + lane_row_0_1) * 16 + (blk_1 + rep_7 * 8) % 8 * 2)));
                                    asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                                        :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&out_f16_7_1[rep_7 * 2])), "r"(*reinterpret_cast<const uint32_t*>(&out_f16_7_1[rep_7 * 2 + 1]))
                                        : "memory");
                                }
                            }
                        }
                        if (a_has_i_1 != 0) {
                            mbarrier_arrive(ab_ready_addr + (kk_cur) * 8);
                            mbarrier_arrive(gate_empty_addr + (g_cur_1) * 8);
                        }
                        kk_cur += 1;
                        if (kk_cur == 3) { kk_cur = 0; _phase_ab_empty ^= 1; }
                        g_cur_1 += 1;
                        if (g_cur_1 == 5) { g_cur_1 = 0; _phase_alpha_full_1 ^= 1; }
                        ag = ag + 1;
                    }
                }
            }
        }
    }
    // ---- Role: tma_w ----
    if (warp == 8) {
        { // tma_w_main
            unsigned int rt_magic = (unsigned int)((1048576 + num_householder - 1) / num_householder);
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");
            unsigned int k_cur_2 = 0;
            unsigned int v_cur_1 = 0;
            unsigned int q_cur_2 = 0;
            unsigned int _phase_load_k_empty = 1;
            unsigned int _phase_v_empty = 1;
            unsigned int _phase_q_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_2 = bid; tile_2 < total_tiles; tile_2 += num_bids) {
                int tile_i_2 = (int)tile_2;
                int batch_idx_2 = tile_i_2 / num_heads;
                int head_idx_2 = tile_i_2 - batch_idx_2 * num_heads;
                int raw_start_2 = cu_seqlens[batch_idx_2];
                int raw_end_2 = cu_seqlens[batch_idx_2 + 1];
                int start_floor_2 = ((raw_start_2 > 0) ? raw_start_2 : 0);
                int batch_start_2 = ((start_floor_2 < total_tokens) ? start_floor_2 : total_tokens);
                int end_floor_2 = ((raw_end_2 > batch_start_2) ? raw_end_2 : batch_start_2);
                int batch_end_2 = ((end_floor_2 < total_tokens) ? end_floor_2 : total_tokens);
                int exp_len_2 = (batch_end_2 - batch_start_2) * num_householder;
                int num_chunks_2 = (exp_len_2 + 63) / 64;
                int t_exp_start = batch_start_2 * num_householder;
                #pragma unroll 1
                for (int chunk_idx_1 = 0; chunk_idx_1 < num_chunks_2; chunk_idx_1++) {
                    int cp_blk_3 = (int)((unsigned int)chunk_idx_1 * rt_magic / 1048576);
                    int cp_i_in_3 = chunk_idx_1 - cp_blk_3 * num_householder;
                    int cp_sub_3 = (int)((unsigned int)(cp_i_in_3 * 64) * rt_magic / 1048576);
                    int cp_tok_off_3 = cp_blk_3 * 64 + cp_sub_3;
                    int cp_rem_3 = cp_i_in_3 * 64 - cp_sub_3 * num_householder;
                    int cp_t0_3 = batch_start_2 + cp_tok_off_3;
                    int cp_b_3 = num_householder - 1 - cp_rem_3;
                    int cp_cnt_3 = (int)((unsigned int)(63 - cp_b_3) * rt_magic / 1048576) + 1;
                    int bp_blk_3 = (int)((unsigned int)chunk_idx_1 * rt_magic / 1048576);
                    int bp_i_in_3 = chunk_idx_1 - bp_blk_3 * num_householder;
                    int bp_roff_3 = bp_i_in_3 * 64;
                    int bp_blk_t0_3 = batch_start_2 + bp_blk_3 * 64;
                    int bp_rem_tok_3 = batch_end_2 - bp_blk_t0_3;
                    int bp_tib_3 = ((bp_rem_tok_3 < 64) ? bp_rem_tok_3 : 64);
                    int bp_is_head_3 = ((bp_i_in_3 == 0) ? 1 : 0);
                    int bp_is_tail_3 = ((bp_i_in_3 == num_householder - 1) ? 1 : ((chunk_idx_1 == num_chunks_2 - 1) ? 1 : 0));
                    int t_exp_offset = t_exp_start + chunk_idx_1 * 64;
                    int t_load_q = bp_is_head_3;
                    int t_q_t0 = bp_blk_t0_3;
                    if (elect_sync()) {
                        mbarrier_wait(load_k_empty_addr + (k_cur_2) * 8, _phase_load_k_empty);
                        mbarrier_arrive_expect_tx(load_k_full_addr + (k_cur_2) * 8, 16384);
                        #pragma unroll
                        for (int dh = 0; dh < 2; dh++) {
                            tma_3d_gmem2smem(smem_k_addr + k_cur_2 * 16384 + (unsigned int)(dh * 8192), (&K), dh * 64, t_exp_offset, head_idx_2, load_k_full_addr + (k_cur_2) * 8);
                        }
                        mbarrier_wait(v_empty_addr + (v_cur_1) * 8, _phase_v_empty);
                        mbarrier_arrive_expect_tx(load_v_full_addr + (v_cur_1) * 8, 8192);
                        tma_3d_gmem2smem(smem_v_addr + v_cur_1 * 8192, (&V), 0, t_exp_offset, head_idx_2, load_v_full_addr + (v_cur_1) * 8);
                        if (t_load_q != 0) {
                            mbarrier_wait(q_empty_addr + (q_cur_2) * 8, _phase_q_empty);
                            mbarrier_arrive_expect_tx(load_q_full_addr + (q_cur_2) * 8, 16384);
                            #pragma unroll
                            for (int dh_1 = 0; dh_1 < 2; dh_1++) {
                                tma_3d_gmem2smem(smem_q_addr + q_cur_2 * 16384 + (unsigned int)(dh_1 * 8192), (&Q), dh_1 * 64, t_q_t0, head_idx_2, load_q_full_addr + (q_cur_2) * 8);
                            }
                        }
                    }
                    if (t_load_q != 0) {
                        q_cur_2 += 1;
                        if (q_cur_2 == 2) { q_cur_2 = 0; _phase_q_empty ^= 1; }
                    }
                    k_cur_2 += 1;
                    if (k_cur_2 == 3) { k_cur_2 = 0; _phase_load_k_empty ^= 1; }
                    v_cur_1 += 1;
                    if (v_cur_1 == 2) { v_cur_1 = 0; _phase_v_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: alpha_w ----
    if (warp == 9) {
        { // alpha_w_main
            unsigned int rl_magic = (unsigned int)((1048576 + num_householder - 1) / num_householder);
            unsigned int g_cur_2 = 0;
            unsigned int _phase_gate_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_3 = bid; tile_3 < total_tiles; tile_3 += num_bids) {
                int tile_i_3 = (int)tile_3;
                int batch_idx_3 = tile_i_3 / num_heads;
                int head_idx_3 = tile_i_3 - batch_idx_3 * num_heads;
                int raw_start_3 = cu_seqlens[batch_idx_3];
                int raw_end_3 = cu_seqlens[batch_idx_3 + 1];
                int start_floor_3 = ((raw_start_3 > 0) ? raw_start_3 : 0);
                int batch_start_3 = ((start_floor_3 < total_tokens) ? start_floor_3 : total_tokens);
                int end_floor_3 = ((raw_end_3 > batch_start_3) ? raw_end_3 : batch_start_3);
                int batch_end_3 = ((end_floor_3 < total_tokens) ? end_floor_3 : total_tokens);
                int exp_len_3 = (batch_end_3 - batch_start_3) * num_householder;
                int num_chunks_3 = (exp_len_3 + 63) / 64;
                #pragma unroll 1
                for (int chunk_idx_2 = 0; chunk_idx_2 < num_chunks_3; chunk_idx_2++) {
                    int bp_blk_4 = (int)((unsigned int)chunk_idx_2 * rl_magic / 1048576);
                    int bp_i_in_4 = chunk_idx_2 - bp_blk_4 * num_householder;
                    int bp_roff_4 = bp_i_in_4 * 64;
                    int bp_blk_t0_4 = batch_start_3 + bp_blk_4 * 64;
                    int bp_rem_tok_4 = batch_end_3 - bp_blk_t0_4;
                    int bp_tib_4 = ((bp_rem_tok_4 < 64) ? bp_rem_tok_4 : 64);
                    int bp_is_head_4 = ((bp_i_in_4 == 0) ? 1 : 0);
                    int bp_is_tail_4 = ((bp_i_in_4 == num_householder - 1) ? 1 : ((chunk_idx_2 == num_chunks_3 - 1) ? 1 : 0));
                    if (bp_is_head_4 != 0) {
                        int al_lane = lane;
                        __syncwarp();
                        int al_tok_a = bp_blk_t0_4 + al_lane;
                        float al_ga = 1.0f;
                        if (al_tok_a < batch_end_3) {
                            al_ga = gate[al_tok_a * num_heads + head_idx_3];
                        }
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(al_ga + 1e-10f));
                        float al_la = ((al_tok_a < batch_end_3) ? _log2_0 : 0.0f);
                        float al_up_a_0 = __shfl_up_sync(0xFFFFFFFF, al_la, 1, 32);
                        if (al_lane >= 1) {
                            al_la = al_la + al_up_a_0;
                        }
                        float al_up_a_1 = __shfl_up_sync(0xFFFFFFFF, al_la, 2, 32);
                        if (al_lane >= 2) {
                            al_la = al_la + al_up_a_1;
                        }
                        float al_up_a_2 = __shfl_up_sync(0xFFFFFFFF, al_la, 4, 32);
                        if (al_lane >= 4) {
                            al_la = al_la + al_up_a_2;
                        }
                        float al_up_a_3 = __shfl_up_sync(0xFFFFFFFF, al_la, 8, 32);
                        if (al_lane >= 8) {
                            al_la = al_la + al_up_a_3;
                        }
                        float al_up_a_4 = __shfl_up_sync(0xFFFFFFFF, al_la, 16, 32);
                        if (al_lane >= 16) {
                            al_la = al_la + al_up_a_4;
                        }
                        float al_carry;
                        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(al_carry) : "f"(al_la), "r"(31));
                        smem_ct[al_lane] = al_la;
                        int al_tok_b = al_tok_a + 32;
                        float al_gb = 1.0f;
                        if (al_tok_b < batch_end_3) {
                            al_gb = gate[al_tok_b * num_heads + head_idx_3];
                        }
                        float _log2_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(al_gb + 1e-10f));
                        float al_lb = ((al_tok_b < batch_end_3) ? _log2_1 : 0.0f);
                        float al_up_b_0 = __shfl_up_sync(0xFFFFFFFF, al_lb, 1, 32);
                        if (al_lane >= 1) {
                            al_lb = al_lb + al_up_b_0;
                        }
                        float al_up_b_1 = __shfl_up_sync(0xFFFFFFFF, al_lb, 2, 32);
                        if (al_lane >= 2) {
                            al_lb = al_lb + al_up_b_1;
                        }
                        float al_up_b_2 = __shfl_up_sync(0xFFFFFFFF, al_lb, 4, 32);
                        if (al_lane >= 4) {
                            al_lb = al_lb + al_up_b_2;
                        }
                        float al_up_b_3 = __shfl_up_sync(0xFFFFFFFF, al_lb, 8, 32);
                        if (al_lane >= 8) {
                            al_lb = al_lb + al_up_b_3;
                        }
                        float al_up_b_4 = __shfl_up_sync(0xFFFFFFFF, al_lb, 16, 32);
                        if (al_lane >= 16) {
                            al_lb = al_lb + al_up_b_4;
                        }
                        al_lb = al_lb + al_carry;
                        smem_ct[al_lane + 32] = al_lb;
                        __syncwarp();
                    }
                    mbarrier_wait(gate_empty_addr + (g_cur_2) * 8, _phase_gate_empty);
                    int ac_lane = lane;
                    int ac_gb = (int)g_cur_2 * 384;
                    float ac_scale = ((scale != 0.0f) ? scale : 0.088388346f);
                    int ac_tkx = ((bp_roff_4 > 0) ? bp_roff_4 - 1 : 0);
                    int ac_tkc = (int)((unsigned int)ac_tkx * rl_magic / 1048576);
                    float ac_carry = ((bp_roff_4 > 0) ? smem_ct[ac_tkc] : 0.0f);
                    int ac_tk63 = (int)((unsigned int)(bp_roff_4 + 63) * rl_magic / 1048576);
                    float ac_cs63 = smem_ct[ac_tk63] - ac_carry;
                    float ac_cta = smem_ct[ac_lane];
                    float ac_ctb = smem_ct[ac_lane + 32];
                    smem_gate[ac_gb + 192 + ac_lane] = ac_cta - ac_carry;
                    float _exp2_2 = approx_exp2(ac_cta);
                    smem_gate[ac_gb + 256 + ac_lane] = _exp2_2 * ac_scale;
                    smem_gate[ac_gb + 192 + ac_lane + 32] = ac_ctb - ac_carry;
                    float _exp2_3 = approx_exp2(ac_ctb);
                    smem_gate[ac_gb + 256 + ac_lane + 32] = _exp2_3 * ac_scale;
                    #pragma unroll 1
                    for (int ac_h = 0; ac_h < 2; ac_h++) {
                        int ac_m = ac_lane + ac_h * 32;
                        int ac_tk = (int)((unsigned int)(bp_roff_4 + ac_m) * rl_magic / 1048576);
                        float ac_l = smem_ct[ac_tk] - ac_carry;
                        smem_gate[ac_gb + 0 + ac_m] = ac_l;
                        float _exp2_4 = approx_exp2(ac_l);
                        smem_gate[ac_gb + 64 + ac_m] = _exp2_4;
                        float _exp2_5 = approx_exp2(ac_cs63 - ac_l);
                        smem_gate[ac_gb + 128 + ac_m] = _exp2_5;
                    }
                    mbarrier_arrive(alpha_full_addr + (g_cur_2) * 8);
                    g_cur_2 += 1;
                    if (g_cur_2 == 5) { g_cur_2 = 0; _phase_gate_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: beta_w ----
    if (warp == 10) {
        { // beta_w_main
            unsigned int g_cur_3 = 0;
            unsigned int _phase_gate_empty_1 = 1;
            #pragma unroll 1
            for (unsigned int tile_4 = bid; tile_4 < total_tiles; tile_4 += num_bids) {
                int tile_i_4 = (int)tile_4;
                int batch_idx_4 = tile_i_4 / num_heads;
                int head_idx_4 = tile_i_4 - batch_idx_4 * num_heads;
                int raw_start_4 = cu_seqlens[batch_idx_4];
                int raw_end_4 = cu_seqlens[batch_idx_4 + 1];
                int start_floor_4 = ((raw_start_4 > 0) ? raw_start_4 : 0);
                int batch_start_4 = ((start_floor_4 < total_tokens) ? start_floor_4 : total_tokens);
                int end_floor_4 = ((raw_end_4 > batch_start_4) ? raw_end_4 : batch_start_4);
                int batch_end_4 = ((end_floor_4 < total_tokens) ? end_floor_4 : total_tokens);
                int exp_len_4 = (batch_end_4 - batch_start_4) * num_householder;
                int num_chunks_4 = (exp_len_4 + 63) / 64;
                int be_exp_start = batch_start_4 * num_householder;
                int be_exp_end = be_exp_start + exp_len_4;
                #pragma unroll 1
                for (int chunk_idx_3 = 0; chunk_idx_3 < num_chunks_4; chunk_idx_3++) {
                    int be_exp_offset = be_exp_start + chunk_idx_3 * 64;
                    mbarrier_wait(gate_empty_addr + (g_cur_3) * 8, _phase_gate_empty_1);
                    int be_lane = lane;
                    int be_gb = (int)g_cur_3 * 384;
                    int be_m0 = be_exp_offset + be_lane;
                    int be_m1 = be_m0 + 32;
                    float be_b0 = 0.0f;
                    float be_b1 = 0.0f;
                    if (be_m0 < be_exp_end) {
                        be_b0 = beta[be_m0 * num_heads + head_idx_4];
                    }
                    if (be_m1 < be_exp_end) {
                        be_b1 = beta[be_m1 * num_heads + head_idx_4];
                    }
                    smem_gate[be_gb + 320 + be_lane] = be_b0;
                    smem_gate[be_gb + 320 + be_lane + 32] = be_b1;
                    mbarrier_arrive(beta_full_addr + (g_cur_3) * 8);
                    g_cur_3 += 1;
                    if (g_cur_3 == 5) { g_cur_3 = 0; _phase_gate_empty_1 ^= 1; }
                }
            }
        }
    }
    // ---- Role: ostore_w ----
    if (warp == 11) {
        { // ostore_w_main
            unsigned int ro_magic = (unsigned int)((1048576 + num_householder - 1) / num_householder);
            int os_slot_base = bid * 512;
            if (elect_sync()) {
                {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>((&O));
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + os_slot_base));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
                asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + os_slot_base)) : "memory");
            }
            unsigned int _phase_o_ready = 0;
            #pragma unroll 1
            for (unsigned int tile_5 = bid; tile_5 < total_tiles; tile_5 += num_bids) {
                int tile_i_5 = (int)tile_5;
                int batch_idx_5 = tile_i_5 / num_heads;
                int head_idx_5 = tile_i_5 - batch_idx_5 * num_heads;
                int raw_start_5 = cu_seqlens[batch_idx_5];
                int raw_end_5 = cu_seqlens[batch_idx_5 + 1];
                int start_floor_5 = ((raw_start_5 > 0) ? raw_start_5 : 0);
                int batch_start_5 = ((start_floor_5 < total_tokens) ? start_floor_5 : total_tokens);
                int end_floor_5 = ((raw_end_5 > batch_start_5) ? raw_end_5 : batch_start_5);
                int batch_end_5 = ((end_floor_5 < total_tokens) ? end_floor_5 : total_tokens);
                int exp_len_5 = (batch_end_5 - batch_start_5) * num_householder;
                int num_chunks_5 = (exp_len_5 + 63) / 64;
                if (num_chunks_5 > 0) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                        asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + os_slot_base)), "r"((uint32_t)(batch_end_5)) : "memory");
                        asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
                        asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + os_slot_base)) : "memory");
                    }
                }
                #pragma unroll 1
                for (int chunk_idx_4 = 0; chunk_idx_4 < num_chunks_5; chunk_idx_4++) {
                    int cp_blk_4 = (int)((unsigned int)chunk_idx_4 * ro_magic / 1048576);
                    int cp_i_in_4 = chunk_idx_4 - cp_blk_4 * num_householder;
                    int cp_sub_4 = (int)((unsigned int)(cp_i_in_4 * 64) * ro_magic / 1048576);
                    int cp_tok_off_4 = cp_blk_4 * 64 + cp_sub_4;
                    int cp_rem_4 = cp_i_in_4 * 64 - cp_sub_4 * num_householder;
                    int cp_t0_4 = batch_start_5 + cp_tok_off_4;
                    int cp_b_4 = num_householder - 1 - cp_rem_4;
                    int cp_cnt_4 = (int)((unsigned int)(63 - cp_b_4) * ro_magic / 1048576) + 1;
                    int bp_blk_5 = (int)((unsigned int)chunk_idx_4 * ro_magic / 1048576);
                    int bp_i_in_5 = chunk_idx_4 - bp_blk_5 * num_householder;
                    int bp_roff_5 = bp_i_in_5 * 64;
                    int bp_blk_t0_5 = batch_start_5 + bp_blk_5 * 64;
                    int bp_rem_tok_5 = batch_end_5 - bp_blk_t0_5;
                    int bp_tib_5 = ((bp_rem_tok_5 < 64) ? bp_rem_tok_5 : 64);
                    int bp_is_head_5 = ((bp_i_in_5 == 0) ? 1 : 0);
                    int bp_is_tail_5 = ((bp_i_in_5 == num_householder - 1) ? 1 : ((chunk_idx_4 == num_chunks_5 - 1) ? 1 : 0));
                    if (bp_is_tail_5 != 0) {
                        mbarrier_wait(o_ready_addr, _phase_o_ready);
                        unsigned int os_base = smem_o_addr + 0 * 8192;
                        if (elect_sync()) {
                            tma_store_3d(tensormap_workspace + os_slot_base, 0, bp_blk_t0_5, head_idx_5, os_base);
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        __syncwarp();
                        mbarrier_arrive(o_empty_addr);
                        _phase_o_ready ^= 1;
                    }
                }
            }
            if (elect_sync()) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }

    // Cleanup
}

} // extern "C"
