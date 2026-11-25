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
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xFFFFFFFF;"
        : "=r"(result) : "r"(val));
    return result;
}


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

__device__ __forceinline__ unsigned int __as_u32(__half2 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}

__device__ __forceinline__ float mul_f32_rz_noftz(float a, float b) {
    float r;
    asm("mul.rz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdp_90c40c9a853e217d9464(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap O, float* __restrict__ gate, float* __restrict__ beta, int* __restrict__ cu_seqlens, int* __restrict__ state_indices, int* __restrict__ output_slot_indices, __half* __restrict__ initial_state, __half* __restrict__ output_state, uint8_t* __restrict__ tensormap_workspace, long long initial_state_stride_slot, long long output_state_stride_slot, float scale, int num_seqs, int total_tokens, int num_heads, int num_householder, int total_tiles, int metadata_i64_mask, uint8_t* __restrict__ ainv_workspace, long long ainv_chunk_stride, int* __restrict__ output_state_indices, long long* __restrict__ rand_seed, int philox_rounds, long long initial_state_size_p0, long long output_state_size_p0)
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

    // Prologue
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

    // Mbarrier init (43 barriers at smem_raw[0..344))

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
                int rd_raw = state_indices[batch_idx];
                int rd_valid = ((rd_raw >= 0 && (long long)rd_raw < initial_state_size_p0) ? 1 : 0);
                int rd_slot = ((rd_valid != 0) ? rd_raw : 0);
                long long output_head_base = (long long)rd_slot * output_state_stride_slot + (long long)head_idx * 8192ll;
                long long initial_head_base = (long long)rd_slot * initial_state_stride_slot + (long long)head_idx * 8192ll;
                int sd16_row_top = warp_id_in_role * 16 + (lane >> 2);
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
                if (rd_valid != 0) {
                    long long sd16_base_top = initial_head_base + (long long)sd16_row_top * 128ll;
                    int sd16_col_pair = (lane & 3) * 2;
                    unsigned int sd16_w[32];
                    {
                        sd16_w[0] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)sd16_col_pair);
                    }
                    {
                        sd16_w[1] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair));
                    }
                    {
                        sd16_w[2] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(sd16_col_pair + 8));
                    }
                    {
                        sd16_w[3] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair + 8));
                    }
                    {
                        sd16_w[4] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(sd16_col_pair + 16));
                    }
                    {
                        sd16_w[5] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair + 16));
                    }
                    {
                        sd16_w[6] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(sd16_col_pair + 24));
                    }
                    {
                        sd16_w[7] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair + 24));
                    }
                    {
                        sd16_w[8] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(sd16_col_pair + 32));
                    }
                    {
                        sd16_w[9] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair + 32));
                    }
                    {
                        sd16_w[10] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(sd16_col_pair + 40));
                    }
                    {
                        sd16_w[11] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair + 40));
                    }
                    {
                        sd16_w[12] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(sd16_col_pair + 48));
                    }
                    {
                        sd16_w[13] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair + 48));
                    }
                    {
                        sd16_w[14] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(sd16_col_pair + 56));
                    }
                    {
                        sd16_w[15] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair + 56));
                    }
                    {
                        sd16_w[16] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(sd16_col_pair + 64));
                    }
                    {
                        sd16_w[17] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair + 64));
                    }
                    {
                        sd16_w[18] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(sd16_col_pair + 72));
                    }
                    {
                        sd16_w[19] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair + 72));
                    }
                    {
                        sd16_w[20] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(sd16_col_pair + 80));
                    }
                    {
                        sd16_w[21] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair + 80));
                    }
                    {
                        sd16_w[22] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(sd16_col_pair + 88));
                    }
                    {
                        sd16_w[23] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair + 88));
                    }
                    {
                        sd16_w[24] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(sd16_col_pair + 96));
                    }
                    {
                        sd16_w[25] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair + 96));
                    }
                    {
                        sd16_w[26] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(sd16_col_pair + 104));
                    }
                    {
                        sd16_w[27] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair + 104));
                    }
                    {
                        sd16_w[28] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(sd16_col_pair + 112));
                    }
                    {
                        sd16_w[29] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair + 112));
                    }
                    {
                        sd16_w[30] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(sd16_col_pair + 120));
                    }
                    {
                        sd16_w[31] = *reinterpret_cast<const uint32_t*>(initial_state + sd16_base_top + (long long)(1024 + sd16_col_pair + 120));
                    }
                    float _cvt_f32_f16_32;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_32) : "h"((uint16_t)(sd16_w[0])));
                    state[0] = _cvt_f32_f16_32;
                    float _cvt_f32_f16_33;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_33) : "h"((uint16_t)(sd16_w[0] >> 16)));
                    state[1] = _cvt_f32_f16_33;
                    float _cvt_f32_f16_34;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_34) : "h"((uint16_t)(sd16_w[1])));
                    state[2] = _cvt_f32_f16_34;
                    float _cvt_f32_f16_35;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_35) : "h"((uint16_t)(sd16_w[1] >> 16)));
                    state[3] = _cvt_f32_f16_35;
                    float _cvt_f32_f16_36;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_36) : "h"((uint16_t)(sd16_w[2])));
                    state[4] = _cvt_f32_f16_36;
                    float _cvt_f32_f16_37;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_37) : "h"((uint16_t)(sd16_w[2] >> 16)));
                    state[5] = _cvt_f32_f16_37;
                    float _cvt_f32_f16_38;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_38) : "h"((uint16_t)(sd16_w[3])));
                    state[6] = _cvt_f32_f16_38;
                    float _cvt_f32_f16_39;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_39) : "h"((uint16_t)(sd16_w[3] >> 16)));
                    state[7] = _cvt_f32_f16_39;
                    float _cvt_f32_f16_40;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_40) : "h"((uint16_t)(sd16_w[4])));
                    state[8] = _cvt_f32_f16_40;
                    float _cvt_f32_f16_41;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_41) : "h"((uint16_t)(sd16_w[4] >> 16)));
                    state[9] = _cvt_f32_f16_41;
                    float _cvt_f32_f16_42;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_42) : "h"((uint16_t)(sd16_w[5])));
                    state[10] = _cvt_f32_f16_42;
                    float _cvt_f32_f16_43;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_43) : "h"((uint16_t)(sd16_w[5] >> 16)));
                    state[11] = _cvt_f32_f16_43;
                    float _cvt_f32_f16_44;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_44) : "h"((uint16_t)(sd16_w[6])));
                    state[12] = _cvt_f32_f16_44;
                    float _cvt_f32_f16_45;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_45) : "h"((uint16_t)(sd16_w[6] >> 16)));
                    state[13] = _cvt_f32_f16_45;
                    float _cvt_f32_f16_46;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_46) : "h"((uint16_t)(sd16_w[7])));
                    state[14] = _cvt_f32_f16_46;
                    float _cvt_f32_f16_47;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_47) : "h"((uint16_t)(sd16_w[7] >> 16)));
                    state[15] = _cvt_f32_f16_47;
                    float _cvt_f32_f16_48;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_48) : "h"((uint16_t)(sd16_w[8])));
                    state[16] = _cvt_f32_f16_48;
                    float _cvt_f32_f16_49;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_49) : "h"((uint16_t)(sd16_w[8] >> 16)));
                    state[17] = _cvt_f32_f16_49;
                    float _cvt_f32_f16_50;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_50) : "h"((uint16_t)(sd16_w[9])));
                    state[18] = _cvt_f32_f16_50;
                    float _cvt_f32_f16_51;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_51) : "h"((uint16_t)(sd16_w[9] >> 16)));
                    state[19] = _cvt_f32_f16_51;
                    float _cvt_f32_f16_52;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_52) : "h"((uint16_t)(sd16_w[10])));
                    state[20] = _cvt_f32_f16_52;
                    float _cvt_f32_f16_53;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_53) : "h"((uint16_t)(sd16_w[10] >> 16)));
                    state[21] = _cvt_f32_f16_53;
                    float _cvt_f32_f16_54;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_54) : "h"((uint16_t)(sd16_w[11])));
                    state[22] = _cvt_f32_f16_54;
                    float _cvt_f32_f16_55;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_55) : "h"((uint16_t)(sd16_w[11] >> 16)));
                    state[23] = _cvt_f32_f16_55;
                    float _cvt_f32_f16_56;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_56) : "h"((uint16_t)(sd16_w[12])));
                    state[24] = _cvt_f32_f16_56;
                    float _cvt_f32_f16_57;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_57) : "h"((uint16_t)(sd16_w[12] >> 16)));
                    state[25] = _cvt_f32_f16_57;
                    float _cvt_f32_f16_58;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_58) : "h"((uint16_t)(sd16_w[13])));
                    state[26] = _cvt_f32_f16_58;
                    float _cvt_f32_f16_59;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_59) : "h"((uint16_t)(sd16_w[13] >> 16)));
                    state[27] = _cvt_f32_f16_59;
                    float _cvt_f32_f16_60;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_60) : "h"((uint16_t)(sd16_w[14])));
                    state[28] = _cvt_f32_f16_60;
                    float _cvt_f32_f16_61;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_61) : "h"((uint16_t)(sd16_w[14] >> 16)));
                    state[29] = _cvt_f32_f16_61;
                    float _cvt_f32_f16_62;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_62) : "h"((uint16_t)(sd16_w[15])));
                    state[30] = _cvt_f32_f16_62;
                    float _cvt_f32_f16_63;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_63) : "h"((uint16_t)(sd16_w[15] >> 16)));
                    state[31] = _cvt_f32_f16_63;
                    float _cvt_f32_f16_64;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_64) : "h"((uint16_t)(sd16_w[16])));
                    state[32] = _cvt_f32_f16_64;
                    float _cvt_f32_f16_65;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_65) : "h"((uint16_t)(sd16_w[16] >> 16)));
                    state[33] = _cvt_f32_f16_65;
                    float _cvt_f32_f16_66;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_66) : "h"((uint16_t)(sd16_w[17])));
                    state[34] = _cvt_f32_f16_66;
                    float _cvt_f32_f16_67;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_67) : "h"((uint16_t)(sd16_w[17] >> 16)));
                    state[35] = _cvt_f32_f16_67;
                    float _cvt_f32_f16_68;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_68) : "h"((uint16_t)(sd16_w[18])));
                    state[36] = _cvt_f32_f16_68;
                    float _cvt_f32_f16_69;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_69) : "h"((uint16_t)(sd16_w[18] >> 16)));
                    state[37] = _cvt_f32_f16_69;
                    float _cvt_f32_f16_70;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_70) : "h"((uint16_t)(sd16_w[19])));
                    state[38] = _cvt_f32_f16_70;
                    float _cvt_f32_f16_71;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_71) : "h"((uint16_t)(sd16_w[19] >> 16)));
                    state[39] = _cvt_f32_f16_71;
                    float _cvt_f32_f16_72;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_72) : "h"((uint16_t)(sd16_w[20])));
                    state[40] = _cvt_f32_f16_72;
                    float _cvt_f32_f16_73;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_73) : "h"((uint16_t)(sd16_w[20] >> 16)));
                    state[41] = _cvt_f32_f16_73;
                    float _cvt_f32_f16_74;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_74) : "h"((uint16_t)(sd16_w[21])));
                    state[42] = _cvt_f32_f16_74;
                    float _cvt_f32_f16_75;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_75) : "h"((uint16_t)(sd16_w[21] >> 16)));
                    state[43] = _cvt_f32_f16_75;
                    float _cvt_f32_f16_76;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_76) : "h"((uint16_t)(sd16_w[22])));
                    state[44] = _cvt_f32_f16_76;
                    float _cvt_f32_f16_77;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_77) : "h"((uint16_t)(sd16_w[22] >> 16)));
                    state[45] = _cvt_f32_f16_77;
                    float _cvt_f32_f16_78;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_78) : "h"((uint16_t)(sd16_w[23])));
                    state[46] = _cvt_f32_f16_78;
                    float _cvt_f32_f16_79;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_79) : "h"((uint16_t)(sd16_w[23] >> 16)));
                    state[47] = _cvt_f32_f16_79;
                    float _cvt_f32_f16_80;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_80) : "h"((uint16_t)(sd16_w[24])));
                    state[48] = _cvt_f32_f16_80;
                    float _cvt_f32_f16_81;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_81) : "h"((uint16_t)(sd16_w[24] >> 16)));
                    state[49] = _cvt_f32_f16_81;
                    float _cvt_f32_f16_82;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_82) : "h"((uint16_t)(sd16_w[25])));
                    state[50] = _cvt_f32_f16_82;
                    float _cvt_f32_f16_83;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_83) : "h"((uint16_t)(sd16_w[25] >> 16)));
                    state[51] = _cvt_f32_f16_83;
                    float _cvt_f32_f16_84;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_84) : "h"((uint16_t)(sd16_w[26])));
                    state[52] = _cvt_f32_f16_84;
                    float _cvt_f32_f16_85;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_85) : "h"((uint16_t)(sd16_w[26] >> 16)));
                    state[53] = _cvt_f32_f16_85;
                    float _cvt_f32_f16_86;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_86) : "h"((uint16_t)(sd16_w[27])));
                    state[54] = _cvt_f32_f16_86;
                    float _cvt_f32_f16_87;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_87) : "h"((uint16_t)(sd16_w[27] >> 16)));
                    state[55] = _cvt_f32_f16_87;
                    float _cvt_f32_f16_88;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_88) : "h"((uint16_t)(sd16_w[28])));
                    state[56] = _cvt_f32_f16_88;
                    float _cvt_f32_f16_89;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_89) : "h"((uint16_t)(sd16_w[28] >> 16)));
                    state[57] = _cvt_f32_f16_89;
                    float _cvt_f32_f16_90;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_90) : "h"((uint16_t)(sd16_w[29])));
                    state[58] = _cvt_f32_f16_90;
                    float _cvt_f32_f16_91;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_91) : "h"((uint16_t)(sd16_w[29] >> 16)));
                    state[59] = _cvt_f32_f16_91;
                    float _cvt_f32_f16_92;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_92) : "h"((uint16_t)(sd16_w[30])));
                    state[60] = _cvt_f32_f16_92;
                    float _cvt_f32_f16_93;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_93) : "h"((uint16_t)(sd16_w[30] >> 16)));
                    state[61] = _cvt_f32_f16_93;
                    float _cvt_f32_f16_94;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_94) : "h"((uint16_t)(sd16_w[31])));
                    state[62] = _cvt_f32_f16_94;
                    float _cvt_f32_f16_95;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_95) : "h"((uint16_t)(sd16_w[31] >> 16)));
                    state[63] = _cvt_f32_f16_95;
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
                uint64_t _wgmma_desc_96 = (((uint64_t)(((smem_prime_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t prime_desc = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_96 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_96);
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n}\n"
                    : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                    : "r"(prime_a[0]), "r"(prime_a[1]), "r"(prime_a[2]), "r"(prime_a[3]), "l"(prime_desc)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                    : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                    : "r"(prime_a[4]), "r"(prime_a[(4) + 1]), "r"(prime_a[(4) + 2]), "r"(prime_a[(4) + 3]), "l"(prime_desc + 2ull)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                    : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                    : "r"(prime_a[8]), "r"(prime_a[(8) + 1]), "r"(prime_a[(8) + 2]), "r"(prime_a[(8) + 3]), "l"(prime_desc + 4ull)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                    : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                    : "r"(prime_a[12]), "r"(prime_a[(12) + 1]), "r"(prime_a[(12) + 2]), "r"(prime_a[(12) + 3]), "l"(prime_desc + 6ull)
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
                                uint32_t _stmatrix_addr_97 = static_cast<uint32_t>((unsigned long long)(obt_base + (unsigned int)(obt_tok * 128 + obt_vb * 2 ^ (obt_tok * 128 + obt_vb * 2 >> 7 & 7) << 4)));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_97), "r"(*reinterpret_cast<const uint32_t*>(&obt_img[tg * 4])), "r"(*reinterpret_cast<const uint32_t*>(&obt_img[tg * 4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&obt_img[tg * 4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&obt_img[tg * 4 + 3]))
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
                        uint64_t _wgmma_desc_98 = (((uint64_t)(((smem_q_addr + q_cur * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t q_lo = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_98 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_98);
                        uint64_t _wgmma_desc_99 = (((uint64_t)(((smem_q_addr + q_cur * 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t q_hi = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_99 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_99);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[0]), "r"(s_img[1]), "r"(s_img[2]), "r"(s_img[3]), "l"(q_lo)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[4]), "r"(s_img[(4) + 1]), "r"(s_img[(4) + 2]), "r"(s_img[(4) + 3]), "l"(q_lo + 2ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[8]), "r"(s_img[(8) + 1]), "r"(s_img[(8) + 2]), "r"(s_img[(8) + 3]), "l"(q_lo + 4ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[12]), "r"(s_img[(12) + 1]), "r"(s_img[(12) + 2]), "r"(s_img[(12) + 3]), "l"(q_lo + 6ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[16]), "r"(s_img[(16) + 1]), "r"(s_img[(16) + 2]), "r"(s_img[(16) + 3]), "l"(q_hi)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[20]), "r"(s_img[(20) + 1]), "r"(s_img[(20) + 2]), "r"(s_img[(20) + 3]), "l"(q_hi + 2ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[24]), "r"(s_img[(24) + 1]), "r"(s_img[(24) + 2]), "r"(s_img[(24) + 3]), "l"(q_hi + 4ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                            : "r"(s_img[28]), "r"(s_img[(28) + 1]), "r"(s_img[(28) + 2]), "r"(s_img[(28) + 3]), "l"(q_hi + 6ull)
                            : "memory");
                        uint64_t _wgmma_desc_100 = (((uint64_t)(((smem_k_addr + k_cur * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t kh_lo = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_100 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_100);
                        uint64_t _wgmma_desc_101 = (((uint64_t)(((smem_k_addr + k_cur * 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t kh_hi = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_101 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_101);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[0]), "r"(s_img[1]), "r"(s_img[2]), "r"(s_img[3]), "l"(kh_lo)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[4]), "r"(s_img[(4) + 1]), "r"(s_img[(4) + 2]), "r"(s_img[(4) + 3]), "l"(kh_lo + 2ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[8]), "r"(s_img[(8) + 1]), "r"(s_img[(8) + 2]), "r"(s_img[(8) + 3]), "l"(kh_lo + 4ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[12]), "r"(s_img[(12) + 1]), "r"(s_img[(12) + 2]), "r"(s_img[(12) + 3]), "l"(kh_lo + 6ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[16]), "r"(s_img[(16) + 1]), "r"(s_img[(16) + 2]), "r"(s_img[(16) + 3]), "l"(kh_hi)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[20]), "r"(s_img[(20) + 1]), "r"(s_img[(20) + 2]), "r"(s_img[(20) + 3]), "l"(kh_hi + 2ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[24]), "r"(s_img[(24) + 1]), "r"(s_img[(24) + 2]), "r"(s_img[(24) + 3]), "l"(kh_hi + 4ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[28]), "r"(s_img[(28) + 1]), "r"(s_img[(28) + 2]), "r"(s_img[(28) + 3]), "l"(kh_hi + 6ull)
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
                        uint64_t _wgmma_desc_102 = (((uint64_t)(((smem_k_addr + k_cur * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t k_lo = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_102 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_102);
                        uint64_t _wgmma_desc_103 = (((uint64_t)(((smem_k_addr + k_cur * 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t k_hi = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_103 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_103);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[0]), "r"(s_img[1]), "r"(s_img[2]), "r"(s_img[3]), "l"(k_lo)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[4]), "r"(s_img[(4) + 1]), "r"(s_img[(4) + 2]), "r"(s_img[(4) + 3]), "l"(k_lo + 2ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[8]), "r"(s_img[(8) + 1]), "r"(s_img[(8) + 2]), "r"(s_img[(8) + 3]), "l"(k_lo + 4ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[12]), "r"(s_img[(12) + 1]), "r"(s_img[(12) + 2]), "r"(s_img[(12) + 3]), "l"(k_lo + 6ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[16]), "r"(s_img[(16) + 1]), "r"(s_img[(16) + 2]), "r"(s_img[(16) + 3]), "l"(k_hi)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[20]), "r"(s_img[(20) + 1]), "r"(s_img[(20) + 2]), "r"(s_img[(20) + 3]), "l"(k_hi + 2ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[24]), "r"(s_img[(24) + 1]), "r"(s_img[(24) + 2]), "r"(s_img[(24) + 3]), "l"(k_hi + 4ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                            : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                            : "r"(s_img[28]), "r"(s_img[(28) + 1]), "r"(s_img[(28) + 2]), "r"(s_img[(28) + 3]), "l"(k_hi + 6ull)
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
                                unsigned int _bit_cast_src_104 = pq_blk[word];
                                lo_f16 = reinterpret_cast<__half*>(&_bit_cast_src_104)[0];
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
                            uint32_t _stmatrix_addr_105 = static_cast<uint32_t>((unsigned long long)(s_abase + (unsigned int)(pq_srow * 128 + (gn * 16 + pq_scol) * 2 ^ (pq_srow * 128 + (gn * 16 + pq_scol) * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_105), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed[3]))
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
                                unsigned int _bit_cast_src_106 = pq_blk_1[word_1];
                                lo_f16_1 = reinterpret_cast<__half*>(&_bit_cast_src_106)[0];
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
                            uint32_t _stmatrix_addr_107 = static_cast<uint32_t>((unsigned long long)(s_abase + (unsigned int)(pq_srow_1 * 128 + (gn_2 * 16 + pq_scol_1) * 2 ^ (pq_srow_1 * 128 + (gn_2 * 16 + pq_scol_1) * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_107), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_1[3]))
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
                                unsigned int _bit_cast_src_108 = pq_blk_2[word_2];
                                lo_f16_2 = reinterpret_cast<__half*>(&_bit_cast_src_108)[0];
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
                            uint32_t _stmatrix_addr_109 = static_cast<uint32_t>((unsigned long long)(s_abase + (unsigned int)(pq_srow_0 * 128 + (gn_4 * 16 + pq_scol_0) * 2 ^ (pq_srow_0 * 128 + (gn_4 * 16 + pq_scol_0) * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_109), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&pq_packed_2[3]))
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
                    uint64_t _wgmma_desc_110 = (((uint64_t)(((smem_abo_addr + ab_cur * 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t ab_desc = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_110 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_110);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n}\n"
                        : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                        : "r"(y_img[0]), "r"(y_img[1]), "r"(y_img[2]), "r"(y_img[3]), "l"(ab_desc)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                        : "r"(y_img[4]), "r"(y_img[(4) + 1]), "r"(y_img[(4) + 2]), "r"(y_img[(4) + 3]), "l"(ab_desc + 2ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                        : "r"(y_img[8]), "r"(y_img[(8) + 1]), "r"(y_img[(8) + 2]), "r"(y_img[(8) + 3]), "l"(ab_desc + 4ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(ks[0]), "+f"(ks[1]), "+f"(ks[2]), "+f"(ks[3]), "+f"(ks[4]), "+f"(ks[5]), "+f"(ks[6]), "+f"(ks[7]), "+f"(ks[8]), "+f"(ks[9]), "+f"(ks[10]), "+f"(ks[11]), "+f"(ks[12]), "+f"(ks[13]), "+f"(ks[14]), "+f"(ks[15]), "+f"(ks[16]), "+f"(ks[17]), "+f"(ks[18]), "+f"(ks[19]), "+f"(ks[20]), "+f"(ks[21]), "+f"(ks[22]), "+f"(ks[23]), "+f"(ks[24]), "+f"(ks[25]), "+f"(ks[26]), "+f"(ks[27]), "+f"(ks[28]), "+f"(ks[29]), "+f"(ks[30]), "+f"(ks[31])
                        : "r"(y_img[12]), "r"(y_img[(12) + 1]), "r"(y_img[(12) + 2]), "r"(y_img[(12) + 3]), "l"(ab_desc + 6ull)
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
                    uint64_t _wgmma_desc_111 = (((uint64_t)(((smem_k_mn_addr + k_cur * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t kmn_desc = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_111 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_111);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_img[0]), "r"(du_img[1]), "r"(du_img[2]), "r"(du_img[3]), "l"(kmn_desc)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_img[4]), "r"(du_img[(4) + 1]), "r"(du_img[(4) + 2]), "r"(du_img[(4) + 3]), "l"(kmn_desc + 128ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_img[8]), "r"(du_img[(8) + 1]), "r"(du_img[(8) + 2]), "r"(du_img[(8) + 3]), "l"(kmn_desc + 256ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_img[12]), "r"(du_img[(12) + 1]), "r"(du_img[(12) + 2]), "r"(du_img[(12) + 3]), "l"(kmn_desc + 384ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_lo[0]), "r"(du_lo[1]), "r"(du_lo[2]), "r"(du_lo[3]), "l"(kmn_desc)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_lo[4]), "r"(du_lo[(4) + 1]), "r"(du_lo[(4) + 2]), "r"(du_lo[(4) + 3]), "l"(kmn_desc + 128ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_lo[8]), "r"(du_lo[(8) + 1]), "r"(du_lo[(8) + 2]), "r"(du_lo[(8) + 3]), "l"(kmn_desc + 256ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(state[0]), "+f"(state[1]), "+f"(state[2]), "+f"(state[3]), "+f"(state[4]), "+f"(state[5]), "+f"(state[6]), "+f"(state[7]), "+f"(state[8]), "+f"(state[9]), "+f"(state[10]), "+f"(state[11]), "+f"(state[12]), "+f"(state[13]), "+f"(state[14]), "+f"(state[15]), "+f"(state[16]), "+f"(state[17]), "+f"(state[18]), "+f"(state[19]), "+f"(state[20]), "+f"(state[21]), "+f"(state[22]), "+f"(state[23]), "+f"(state[24]), "+f"(state[25]), "+f"(state[26]), "+f"(state[27]), "+f"(state[28]), "+f"(state[29]), "+f"(state[30]), "+f"(state[31]), "+f"(state[32]), "+f"(state[33]), "+f"(state[34]), "+f"(state[35]), "+f"(state[36]), "+f"(state[37]), "+f"(state[38]), "+f"(state[39]), "+f"(state[40]), "+f"(state[41]), "+f"(state[42]), "+f"(state[43]), "+f"(state[44]), "+f"(state[45]), "+f"(state[46]), "+f"(state[47]), "+f"(state[48]), "+f"(state[49]), "+f"(state[50]), "+f"(state[51]), "+f"(state[52]), "+f"(state[53]), "+f"(state[54]), "+f"(state[55]), "+f"(state[56]), "+f"(state[57]), "+f"(state[58]), "+f"(state[59]), "+f"(state[60]), "+f"(state[61]), "+f"(state[62]), "+f"(state[63])
                        : "r"(du_lo[12]), "r"(du_lo[(12) + 1]), "r"(du_lo[(12) + 2]), "r"(du_lo[(12) + 3]), "l"(kmn_desc + 384ull)
                        : "memory");
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_112 = (((uint64_t)(((smem_qk_addr + qk_cur * 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t qk_desc = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_112 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_112);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                        : "r"(nv_img[0]), "r"(nv_img[1]), "r"(nv_img[2]), "r"(nv_img[3]), "l"(qk_desc)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                        : "r"(nv_img[4]), "r"(nv_img[(4) + 1]), "r"(nv_img[(4) + 2]), "r"(nv_img[(4) + 3]), "l"(qk_desc + 2ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                        : "r"(nv_img[8]), "r"(nv_img[(8) + 1]), "r"(nv_img[(8) + 2]), "r"(nv_img[(8) + 3]), "l"(qk_desc + 4ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(q_state[0]), "+f"(q_state[1]), "+f"(q_state[2]), "+f"(q_state[3]), "+f"(q_state[4]), "+f"(q_state[5]), "+f"(q_state[6]), "+f"(q_state[7]), "+f"(q_state[8]), "+f"(q_state[9]), "+f"(q_state[10]), "+f"(q_state[11]), "+f"(q_state[12]), "+f"(q_state[13]), "+f"(q_state[14]), "+f"(q_state[15]), "+f"(q_state[16]), "+f"(q_state[17]), "+f"(q_state[18]), "+f"(q_state[19]), "+f"(q_state[20]), "+f"(q_state[21]), "+f"(q_state[22]), "+f"(q_state[23]), "+f"(q_state[24]), "+f"(q_state[25]), "+f"(q_state[26]), "+f"(q_state[27]), "+f"(q_state[28]), "+f"(q_state[29]), "+f"(q_state[30]), "+f"(q_state[31])
                        : "r"(nv_img[12]), "r"(nv_img[(12) + 1]), "r"(nv_img[(12) + 2]), "r"(nv_img[(12) + 3]), "l"(qk_desc + 6ull)
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
                        uint32_t _stmatrix_addr_113 = static_cast<uint32_t>((unsigned long long)(obt_base_1 + (unsigned int)(obt_tok_1 * 128 + obt_vb_1 * 2 ^ (obt_tok_1 * 128 + obt_vb_1 * 2 >> 7 & 7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_113), "r"(*reinterpret_cast<const uint32_t*>(&obt_img_1[tg_1 * 4])), "r"(*reinterpret_cast<const uint32_t*>(&obt_img_1[tg_1 * 4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&obt_img_1[tg_1 * 4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&obt_img_1[tg_1 * 4 + 3]))
                            : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(o_ready_addr);
                    _phase_o_empty ^= 1;
                    mbarrier_arrive(qk_empty_addr + (((qk_cur + 2 >= 3) ? qk_cur + 2 - 3 : qk_cur + 2)) * 8);
                }
                int ps_w = output_state_indices[batch_idx];
                int ps_ok = ((ps_w >= 0 && (long long)ps_w < output_state_size_p0) ? 1 : 0);
                if (ps_ok != 0) {
                    long long ps_whead = (long long)ps_w * output_state_stride_slot + (long long)head_idx * 8192ll;
                    int ps_r = state_indices[batch_idx];
                    long long ps_rhead = (long long)ps_r * initial_state_stride_slot + (long long)head_idx * 8192ll;
                    long long ps_ctr = ((ps_r >= 0 && (long long)ps_r < initial_state_size_p0) ? ps_rhead : ps_whead);
                    int st16_quad = lane & 3;
                    int st16_row_top = warp_id_in_role * 16 + (lane >> 2);
                    long long st16_top = ps_whead + (long long)st16_row_top * 128ll + (long long)(st16_quad * 2);
                    unsigned int st16_pk[1];
                    if (philox_rounds == 0) {
                        uint32_t _cvt_rn_f16x2_f32_114;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_114) : "f"(state[1]), "f"(state[0]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_114;
                        reinterpret_cast<unsigned*>(output_state + st16_top)[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_115;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_115) : "f"(state[3]), "f"(state[2]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_115;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1024ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_116;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_116) : "f"(state[5]), "f"(state[4]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_116;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 8ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_117;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_117) : "f"(state[7]), "f"(state[6]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_117;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1032ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_118;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_118) : "f"(state[9]), "f"(state[8]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_118;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 16ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_119;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_119) : "f"(state[11]), "f"(state[10]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_119;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1040ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_120;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_120) : "f"(state[13]), "f"(state[12]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_120;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 24ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_121;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_121) : "f"(state[15]), "f"(state[14]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_121;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1048ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_122;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_122) : "f"(state[17]), "f"(state[16]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_122;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 32ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_123;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_123) : "f"(state[19]), "f"(state[18]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_123;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1056ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_124;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_124) : "f"(state[21]), "f"(state[20]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_124;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 40ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_125;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_125) : "f"(state[23]), "f"(state[22]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_125;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1064ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_126;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_126) : "f"(state[25]), "f"(state[24]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_126;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 48ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_127;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_127) : "f"(state[27]), "f"(state[26]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_127;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1072ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_128;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_128) : "f"(state[29]), "f"(state[28]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_128;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 56ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_129;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_129) : "f"(state[31]), "f"(state[30]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_129;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1080ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_130;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_130) : "f"(state[33]), "f"(state[32]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_130;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 64ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_131;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_131) : "f"(state[35]), "f"(state[34]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_131;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1088ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_132;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_132) : "f"(state[37]), "f"(state[36]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_132;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 72ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_133;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_133) : "f"(state[39]), "f"(state[38]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_133;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1096ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_134;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_134) : "f"(state[41]), "f"(state[40]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_134;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 80ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_135;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_135) : "f"(state[43]), "f"(state[42]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_135;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1104ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_136;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_136) : "f"(state[45]), "f"(state[44]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_136;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 88ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_137;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_137) : "f"(state[47]), "f"(state[46]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_137;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1112ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_138;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_138) : "f"(state[49]), "f"(state[48]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_138;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 96ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_139;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_139) : "f"(state[51]), "f"(state[50]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_139;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1120ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_140;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_140) : "f"(state[53]), "f"(state[52]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_140;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 104ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_141;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_141) : "f"(state[55]), "f"(state[54]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_141;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1128ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_142;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_142) : "f"(state[57]), "f"(state[56]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_142;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 112ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_143;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_143) : "f"(state[59]), "f"(state[58]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_143;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1136ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_144;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_144) : "f"(state[61]), "f"(state[60]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_144;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 120ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        uint32_t _cvt_rn_f16x2_f32_145;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_145) : "f"(state[63]), "f"(state[62]));
                        st16_pk[0] = _cvt_rn_f16x2_f32_145;
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1144ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                    }
                    if (philox_rounds != 0) {
                        long long st16_seed = *rand_seed;
                        unsigned int st16_key_lo = (unsigned int)st16_seed;
                        unsigned int st16_key_hi = (unsigned int)(st16_seed >> 32ll);
                        int st16_par = st16_quad & 1;
                        long long st16_base = ps_ctr + (long long)(st16_row_top + 8 * st16_par) * 128ll + (long long)((st16_quad >> 1) * 4);
                        unsigned int st16_w[4];
                        unsigned int st16_rb[4];
                        unsigned int key0 = st16_key_lo;
                        unsigned int key1 = st16_key_hi;
                        long long o_0 = st16_base;
                        unsigned long long p0_0 = (unsigned long long)(unsigned int)o_0 * 3528531795ull;
                        st16_w[0] = (unsigned int)(o_0 >> 32ll) ^ key0;
                        st16_w[2] = (unsigned int)(p0_0 >> 32ull) ^ key1;
                        st16_w[3] = (unsigned int)p0_0;
                        key0 = key0 + 2654435769u;
                        key1 = key1 + 3144134277u;
                        unsigned long long q0_0 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long q1_0 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(q1_0 >> 32ull) ^ key0;
                        st16_w[1] = (unsigned int)q1_0;
                        st16_w[2] = (unsigned int)(q0_0 >> 32ull) ^ st16_w[3] ^ key1;
                        st16_w[3] = (unsigned int)q0_0;
                        key0 = key0 + 2654435769u;
                        key1 = key1 + 3144134277u;
                        unsigned long long s0_0 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_0 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_0 >> 32ull) ^ st16_w[1] ^ key0;
                        st16_w[1] = (unsigned int)s1_0;
                        st16_w[2] = (unsigned int)(s0_0 >> 32ull) ^ st16_w[3] ^ key1;
                        st16_w[3] = (unsigned int)s0_0;
                        key0 = key0 + 2654435769u;
                        key1 = key1 + 3144134277u;
                        unsigned long long s0_1 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_1 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_1 >> 32ull) ^ st16_w[1] ^ key0;
                        st16_w[1] = (unsigned int)s1_1;
                        st16_w[2] = (unsigned int)(s0_1 >> 32ull) ^ st16_w[3] ^ key1;
                        st16_w[3] = (unsigned int)s0_1;
                        key0 = key0 + 2654435769u;
                        key1 = key1 + 3144134277u;
                        unsigned long long s0_2 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_2 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_2 >> 32ull) ^ st16_w[1] ^ key0;
                        st16_w[1] = (unsigned int)s1_2;
                        st16_w[2] = (unsigned int)(s0_2 >> 32ull) ^ st16_w[3] ^ key1;
                        st16_w[3] = (unsigned int)s0_2;
                        key0 = key0 + 2654435769u;
                        key1 = key1 + 3144134277u;
                        #pragma unroll 1
                        for (int _round = 5; _round < philox_rounds; _round++) {
                            unsigned long long t0_0 = (unsigned long long)st16_w[0] * 3528531795ull;
                            unsigned long long t1_0 = (unsigned long long)st16_w[2] * 3449720151ull;
                            st16_w[0] = (unsigned int)(t1_0 >> 32ull) ^ st16_w[1] ^ key0;
                            st16_w[1] = (unsigned int)t1_0;
                            st16_w[2] = (unsigned int)(t0_0 >> 32ull) ^ st16_w[3] ^ key1;
                            st16_w[3] = (unsigned int)t0_0;
                            key0 = key0 + 2654435769u;
                            key1 = key1 + 3144134277u;
                        }
                        unsigned int st16_give0_0 = ((st16_par != 0) ? st16_w[0] : st16_w[1]);
                        unsigned int st16_give1_0 = ((st16_par != 0) ? st16_w[2] : st16_w[3]);
                        unsigned int st16_own0_0 = ((st16_par != 0) ? st16_w[1] : st16_w[0]);
                        unsigned int st16_own1_0 = ((st16_par != 0) ? st16_w[3] : st16_w[2]);
                        unsigned int st16_got0_0 = __shfl_xor_sync(0xFFFFFFFF, st16_give0_0, 1);
                        unsigned int st16_got1_0 = __shfl_xor_sync(0xFFFFFFFF, st16_give1_0, 1);
                        st16_rb[0] = ((st16_par != 0) ? st16_got0_0 : st16_own0_0);
                        st16_rb[1] = ((st16_par != 0) ? st16_own0_0 : st16_got0_0);
                        st16_rb[2] = ((st16_par != 0) ? st16_got1_0 : st16_own1_0);
                        st16_rb[3] = ((st16_par != 0) ? st16_own1_0 : st16_got1_0);
                        float st16_lo_0 = state[0];
                        float st16_hi_0 = state[1];
                        unsigned int st16_da_0 = (__as_u32(st16_lo_0) & 2147483647) - 947912704;
                        unsigned int st16_db_0 = (__as_u32(st16_hi_0) & 2147483647) - 947912704;
                        float flo = __uint_as_float(__as_u32(st16_lo_0) + (st16_rb[0] & 8191) & 4294959104u);
                        float fhi = __uint_as_float(__as_u32(st16_hi_0) + (st16_rb[0] >> 16 & 8191) & 4294959104u);
                        uint32_t word_3;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_3) : "f"(fhi), "f"(flo));
                        unsigned int st16_fw_0 = word_3;
                        float _abs_146;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_146) : "f"(st16_lo_0));
                        unsigned int s = __as_u32(mul_f32_rz_noftz(_abs_146, __uint_as_float(125829120)));
                        unsigned int _min_147 = ((s) < (260046848) ? (s) : (260046848));
                        unsigned int h = _min_147 + (st16_rb[0] & 8191) >> 13;
                        unsigned int mag = ((s <= 2139095040) ? h : 32256);
                        float _abs_148;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_148) : "f"(st16_hi_0));
                        unsigned int s_0 = __as_u32(mul_f32_rz_noftz(_abs_148, __uint_as_float(125829120)));
                        unsigned int _min_149 = ((s_0) < (260046848) ? (s_0) : (260046848));
                        unsigned int h_0 = _min_149 + ((st16_rb[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_0 = ((s_0 <= 2139095040) ? h_0 : 32256);
                        uint32_t mags;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags) : "r"(mag), "r"(mag_0));
                        uint32_t signs;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs) : "r"(__as_u32(st16_lo_0)), "r"(__as_u32(st16_hi_0)));
                        unsigned int word_0 = signs & 2147516416u | mags;
                        unsigned int st16_sw_0 = word_0;
                        unsigned int _max_150 = ((st16_da_0) > (st16_db_0) ? (st16_da_0) : (st16_db_0));
                        st16_pk[0] = ((_max_150 <= 1191182336) ? st16_fw_0 : st16_sw_0);
                        reinterpret_cast<unsigned*>(output_state + st16_top)[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_1 = state[2];
                        float st16_hi_1 = state[3];
                        unsigned int st16_da_1 = (__as_u32(st16_lo_1) & 2147483647) - 947912704;
                        unsigned int st16_db_1 = (__as_u32(st16_hi_1) & 2147483647) - 947912704;
                        float flo_0 = __uint_as_float(__as_u32(st16_lo_1) + (st16_rb[1] & 8191) & 4294959104u);
                        float fhi_0 = __uint_as_float(__as_u32(st16_hi_1) + (st16_rb[1] >> 16 & 8191) & 4294959104u);
                        uint32_t word_1_1;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_1_1) : "f"(fhi_0), "f"(flo_0));
                        unsigned int st16_fw_1 = word_1_1;
                        float _abs_151;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_151) : "f"(st16_lo_1));
                        unsigned int s_1 = __as_u32(mul_f32_rz_noftz(_abs_151, __uint_as_float(125829120)));
                        unsigned int _min_152 = ((s_1) < (260046848) ? (s_1) : (260046848));
                        unsigned int h_1 = _min_152 + (st16_rb[1] & 8191) >> 13;
                        unsigned int mag_1 = ((s_1 <= 2139095040) ? h_1 : 32256);
                        float _abs_153;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_153) : "f"(st16_hi_1));
                        unsigned int s_2 = __as_u32(mul_f32_rz_noftz(_abs_153, __uint_as_float(125829120)));
                        unsigned int _min_154 = ((s_2) < (260046848) ? (s_2) : (260046848));
                        unsigned int h_2 = _min_154 + ((st16_rb[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_2 = ((s_2 <= 2139095040) ? h_2 : 32256);
                        uint32_t mags_0;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_0) : "r"(mag_1), "r"(mag_2));
                        uint32_t signs_0;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_0) : "r"(__as_u32(st16_lo_1)), "r"(__as_u32(st16_hi_1)));
                        unsigned int word_2_1 = signs_0 & 2147516416u | mags_0;
                        unsigned int st16_sw_1 = word_2_1;
                        unsigned int _max_155 = ((st16_da_1) > (st16_db_1) ? (st16_da_1) : (st16_db_1));
                        st16_pk[0] = ((_max_155 <= 1191182336) ? st16_fw_1 : st16_sw_1);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1024ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_2 = state[32];
                        float st16_hi_2 = state[33];
                        unsigned int st16_da_2 = (__as_u32(st16_lo_2) & 2147483647) - 947912704;
                        unsigned int st16_db_2 = (__as_u32(st16_hi_2) & 2147483647) - 947912704;
                        float flo_1 = __uint_as_float(__as_u32(st16_lo_2) + (st16_rb[2] & 8191) & 4294959104u);
                        float fhi_1 = __uint_as_float(__as_u32(st16_hi_2) + (st16_rb[2] >> 16 & 8191) & 4294959104u);
                        uint32_t word_3_1;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_3_1) : "f"(fhi_1), "f"(flo_1));
                        unsigned int st16_fw_2 = word_3_1;
                        float _abs_156;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_156) : "f"(st16_lo_2));
                        unsigned int s_3 = __as_u32(mul_f32_rz_noftz(_abs_156, __uint_as_float(125829120)));
                        unsigned int _min_157 = ((s_3) < (260046848) ? (s_3) : (260046848));
                        unsigned int h_3 = _min_157 + (st16_rb[2] & 8191) >> 13;
                        unsigned int mag_3 = ((s_3 <= 2139095040) ? h_3 : 32256);
                        float _abs_158;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_158) : "f"(st16_hi_2));
                        unsigned int s_4 = __as_u32(mul_f32_rz_noftz(_abs_158, __uint_as_float(125829120)));
                        unsigned int _min_159 = ((s_4) < (260046848) ? (s_4) : (260046848));
                        unsigned int h_4 = _min_159 + ((st16_rb[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_4 = ((s_4 <= 2139095040) ? h_4 : 32256);
                        uint32_t mags_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_1) : "r"(mag_3), "r"(mag_4));
                        uint32_t signs_1;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_1) : "r"(__as_u32(st16_lo_2)), "r"(__as_u32(st16_hi_2)));
                        unsigned int word_4 = signs_1 & 2147516416u | mags_1;
                        unsigned int st16_sw_2 = word_4;
                        unsigned int _max_160 = ((st16_da_2) > (st16_db_2) ? (st16_da_2) : (st16_db_2));
                        st16_pk[0] = ((_max_160 <= 1191182336) ? st16_fw_2 : st16_sw_2);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 64ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_3 = state[34];
                        float st16_hi_3 = state[35];
                        unsigned int st16_da_3 = (__as_u32(st16_lo_3) & 2147483647) - 947912704;
                        unsigned int st16_db_3 = (__as_u32(st16_hi_3) & 2147483647) - 947912704;
                        float flo_2 = __uint_as_float(__as_u32(st16_lo_3) + (st16_rb[3] & 8191) & 4294959104u);
                        float fhi_2 = __uint_as_float(__as_u32(st16_hi_3) + (st16_rb[3] >> 16 & 8191) & 4294959104u);
                        uint32_t word_5;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_5) : "f"(fhi_2), "f"(flo_2));
                        unsigned int st16_fw_3 = word_5;
                        float _abs_161;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_161) : "f"(st16_lo_3));
                        unsigned int s_5 = __as_u32(mul_f32_rz_noftz(_abs_161, __uint_as_float(125829120)));
                        unsigned int _min_162 = ((s_5) < (260046848) ? (s_5) : (260046848));
                        unsigned int h_5 = _min_162 + (st16_rb[3] & 8191) >> 13;
                        unsigned int mag_5 = ((s_5 <= 2139095040) ? h_5 : 32256);
                        float _abs_163;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_163) : "f"(st16_hi_3));
                        unsigned int s_6 = __as_u32(mul_f32_rz_noftz(_abs_163, __uint_as_float(125829120)));
                        unsigned int _min_164 = ((s_6) < (260046848) ? (s_6) : (260046848));
                        unsigned int h_6 = _min_164 + ((st16_rb[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_6 = ((s_6 <= 2139095040) ? h_6 : 32256);
                        uint32_t mags_2;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_2) : "r"(mag_5), "r"(mag_6));
                        uint32_t signs_2;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_2) : "r"(__as_u32(st16_lo_3)), "r"(__as_u32(st16_hi_3)));
                        unsigned int word_6 = signs_2 & 2147516416u | mags_2;
                        unsigned int st16_sw_3 = word_6;
                        unsigned int _max_165 = ((st16_da_3) > (st16_db_3) ? (st16_da_3) : (st16_db_3));
                        st16_pk[0] = ((_max_165 <= 1191182336) ? st16_fw_3 : st16_sw_3);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1088ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        unsigned int key0_0 = st16_key_lo;
                        unsigned int key1_0 = st16_key_hi;
                        long long o_1 = st16_base + 8ll;
                        unsigned long long p0_1 = (unsigned long long)(unsigned int)o_1 * 3528531795ull;
                        st16_w[0] = (unsigned int)(o_1 >> 32ll) ^ key0_0;
                        st16_w[2] = (unsigned int)(p0_1 >> 32ull) ^ key1_0;
                        st16_w[3] = (unsigned int)p0_1;
                        key0_0 = key0_0 + 2654435769u;
                        key1_0 = key1_0 + 3144134277u;
                        unsigned long long q0_1 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long q1_1 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(q1_1 >> 32ull) ^ key0_0;
                        st16_w[1] = (unsigned int)q1_1;
                        st16_w[2] = (unsigned int)(q0_1 >> 32ull) ^ st16_w[3] ^ key1_0;
                        st16_w[3] = (unsigned int)q0_1;
                        key0_0 = key0_0 + 2654435769u;
                        key1_0 = key1_0 + 3144134277u;
                        unsigned long long s0_3 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_3 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_3 >> 32ull) ^ st16_w[1] ^ key0_0;
                        st16_w[1] = (unsigned int)s1_3;
                        st16_w[2] = (unsigned int)(s0_3 >> 32ull) ^ st16_w[3] ^ key1_0;
                        st16_w[3] = (unsigned int)s0_3;
                        key0_0 = key0_0 + 2654435769u;
                        key1_0 = key1_0 + 3144134277u;
                        unsigned long long s0_4 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_4 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_4 >> 32ull) ^ st16_w[1] ^ key0_0;
                        st16_w[1] = (unsigned int)s1_4;
                        st16_w[2] = (unsigned int)(s0_4 >> 32ull) ^ st16_w[3] ^ key1_0;
                        st16_w[3] = (unsigned int)s0_4;
                        key0_0 = key0_0 + 2654435769u;
                        key1_0 = key1_0 + 3144134277u;
                        unsigned long long s0_5 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_5 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_5 >> 32ull) ^ st16_w[1] ^ key0_0;
                        st16_w[1] = (unsigned int)s1_5;
                        st16_w[2] = (unsigned int)(s0_5 >> 32ull) ^ st16_w[3] ^ key1_0;
                        st16_w[3] = (unsigned int)s0_5;
                        key0_0 = key0_0 + 2654435769u;
                        key1_0 = key1_0 + 3144134277u;
                        #pragma unroll 1
                        for (int _round_1 = 5; _round_1 < philox_rounds; _round_1++) {
                            unsigned long long t0_1 = (unsigned long long)st16_w[0] * 3528531795ull;
                            unsigned long long t1_1 = (unsigned long long)st16_w[2] * 3449720151ull;
                            st16_w[0] = (unsigned int)(t1_1 >> 32ull) ^ st16_w[1] ^ key0_0;
                            st16_w[1] = (unsigned int)t1_1;
                            st16_w[2] = (unsigned int)(t0_1 >> 32ull) ^ st16_w[3] ^ key1_0;
                            st16_w[3] = (unsigned int)t0_1;
                            key0_0 = key0_0 + 2654435769u;
                            key1_0 = key1_0 + 3144134277u;
                        }
                        unsigned int st16_give0_1 = ((st16_par != 0) ? st16_w[0] : st16_w[1]);
                        unsigned int st16_give1_1 = ((st16_par != 0) ? st16_w[2] : st16_w[3]);
                        unsigned int st16_own0_1 = ((st16_par != 0) ? st16_w[1] : st16_w[0]);
                        unsigned int st16_own1_1 = ((st16_par != 0) ? st16_w[3] : st16_w[2]);
                        unsigned int st16_got0_1 = __shfl_xor_sync(0xFFFFFFFF, st16_give0_1, 1);
                        unsigned int st16_got1_1 = __shfl_xor_sync(0xFFFFFFFF, st16_give1_1, 1);
                        st16_rb[0] = ((st16_par != 0) ? st16_got0_1 : st16_own0_1);
                        st16_rb[1] = ((st16_par != 0) ? st16_own0_1 : st16_got0_1);
                        st16_rb[2] = ((st16_par != 0) ? st16_got1_1 : st16_own1_1);
                        st16_rb[3] = ((st16_par != 0) ? st16_own1_1 : st16_got1_1);
                        float st16_lo_4 = state[4];
                        float st16_hi_4 = state[5];
                        unsigned int st16_da_4 = (__as_u32(st16_lo_4) & 2147483647) - 947912704;
                        unsigned int st16_db_4 = (__as_u32(st16_hi_4) & 2147483647) - 947912704;
                        float flo_3 = __uint_as_float(__as_u32(st16_lo_4) + (st16_rb[0] & 8191) & 4294959104u);
                        float fhi_3 = __uint_as_float(__as_u32(st16_hi_4) + (st16_rb[0] >> 16 & 8191) & 4294959104u);
                        uint32_t word_7;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_7) : "f"(fhi_3), "f"(flo_3));
                        unsigned int st16_fw_4 = word_7;
                        float _abs_166;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_166) : "f"(st16_lo_4));
                        unsigned int s_7 = __as_u32(mul_f32_rz_noftz(_abs_166, __uint_as_float(125829120)));
                        unsigned int _min_167 = ((s_7) < (260046848) ? (s_7) : (260046848));
                        unsigned int h_7 = _min_167 + (st16_rb[0] & 8191) >> 13;
                        unsigned int mag_7 = ((s_7 <= 2139095040) ? h_7 : 32256);
                        float _abs_168;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_168) : "f"(st16_hi_4));
                        unsigned int s_8 = __as_u32(mul_f32_rz_noftz(_abs_168, __uint_as_float(125829120)));
                        unsigned int _min_169 = ((s_8) < (260046848) ? (s_8) : (260046848));
                        unsigned int h_8 = _min_169 + ((st16_rb[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_8 = ((s_8 <= 2139095040) ? h_8 : 32256);
                        uint32_t mags_3;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_3) : "r"(mag_7), "r"(mag_8));
                        uint32_t signs_3;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_3) : "r"(__as_u32(st16_lo_4)), "r"(__as_u32(st16_hi_4)));
                        unsigned int word_8 = signs_3 & 2147516416u | mags_3;
                        unsigned int st16_sw_4 = word_8;
                        unsigned int _max_170 = ((st16_da_4) > (st16_db_4) ? (st16_da_4) : (st16_db_4));
                        st16_pk[0] = ((_max_170 <= 1191182336) ? st16_fw_4 : st16_sw_4);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 8ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_5 = state[6];
                        float st16_hi_5 = state[7];
                        unsigned int st16_da_5 = (__as_u32(st16_lo_5) & 2147483647) - 947912704;
                        unsigned int st16_db_5 = (__as_u32(st16_hi_5) & 2147483647) - 947912704;
                        float flo_4 = __uint_as_float(__as_u32(st16_lo_5) + (st16_rb[1] & 8191) & 4294959104u);
                        float fhi_4 = __uint_as_float(__as_u32(st16_hi_5) + (st16_rb[1] >> 16 & 8191) & 4294959104u);
                        uint32_t word_9;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_9) : "f"(fhi_4), "f"(flo_4));
                        unsigned int st16_fw_5 = word_9;
                        float _abs_171;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_171) : "f"(st16_lo_5));
                        unsigned int s_9 = __as_u32(mul_f32_rz_noftz(_abs_171, __uint_as_float(125829120)));
                        unsigned int _min_172 = ((s_9) < (260046848) ? (s_9) : (260046848));
                        unsigned int h_9 = _min_172 + (st16_rb[1] & 8191) >> 13;
                        unsigned int mag_9 = ((s_9 <= 2139095040) ? h_9 : 32256);
                        float _abs_173;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_173) : "f"(st16_hi_5));
                        unsigned int s_10 = __as_u32(mul_f32_rz_noftz(_abs_173, __uint_as_float(125829120)));
                        unsigned int _min_174 = ((s_10) < (260046848) ? (s_10) : (260046848));
                        unsigned int h_10 = _min_174 + ((st16_rb[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_10 = ((s_10 <= 2139095040) ? h_10 : 32256);
                        uint32_t mags_4;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_4) : "r"(mag_9), "r"(mag_10));
                        uint32_t signs_4;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_4) : "r"(__as_u32(st16_lo_5)), "r"(__as_u32(st16_hi_5)));
                        unsigned int word_10 = signs_4 & 2147516416u | mags_4;
                        unsigned int st16_sw_5 = word_10;
                        unsigned int _max_175 = ((st16_da_5) > (st16_db_5) ? (st16_da_5) : (st16_db_5));
                        st16_pk[0] = ((_max_175 <= 1191182336) ? st16_fw_5 : st16_sw_5);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1032ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_6 = state[36];
                        float st16_hi_6 = state[37];
                        unsigned int st16_da_6 = (__as_u32(st16_lo_6) & 2147483647) - 947912704;
                        unsigned int st16_db_6 = (__as_u32(st16_hi_6) & 2147483647) - 947912704;
                        float flo_5 = __uint_as_float(__as_u32(st16_lo_6) + (st16_rb[2] & 8191) & 4294959104u);
                        float fhi_5 = __uint_as_float(__as_u32(st16_hi_6) + (st16_rb[2] >> 16 & 8191) & 4294959104u);
                        uint32_t word_11;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_11) : "f"(fhi_5), "f"(flo_5));
                        unsigned int st16_fw_6 = word_11;
                        float _abs_176;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_176) : "f"(st16_lo_6));
                        unsigned int s_11 = __as_u32(mul_f32_rz_noftz(_abs_176, __uint_as_float(125829120)));
                        unsigned int _min_177 = ((s_11) < (260046848) ? (s_11) : (260046848));
                        unsigned int h_11 = _min_177 + (st16_rb[2] & 8191) >> 13;
                        unsigned int mag_11 = ((s_11 <= 2139095040) ? h_11 : 32256);
                        float _abs_178;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_178) : "f"(st16_hi_6));
                        unsigned int s_12 = __as_u32(mul_f32_rz_noftz(_abs_178, __uint_as_float(125829120)));
                        unsigned int _min_179 = ((s_12) < (260046848) ? (s_12) : (260046848));
                        unsigned int h_12 = _min_179 + ((st16_rb[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_12 = ((s_12 <= 2139095040) ? h_12 : 32256);
                        uint32_t mags_5;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_5) : "r"(mag_11), "r"(mag_12));
                        uint32_t signs_5;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_5) : "r"(__as_u32(st16_lo_6)), "r"(__as_u32(st16_hi_6)));
                        unsigned int word_12 = signs_5 & 2147516416u | mags_5;
                        unsigned int st16_sw_6 = word_12;
                        unsigned int _max_180 = ((st16_da_6) > (st16_db_6) ? (st16_da_6) : (st16_db_6));
                        st16_pk[0] = ((_max_180 <= 1191182336) ? st16_fw_6 : st16_sw_6);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 72ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_7 = state[38];
                        float st16_hi_7 = state[39];
                        unsigned int st16_da_7 = (__as_u32(st16_lo_7) & 2147483647) - 947912704;
                        unsigned int st16_db_7 = (__as_u32(st16_hi_7) & 2147483647) - 947912704;
                        float flo_6 = __uint_as_float(__as_u32(st16_lo_7) + (st16_rb[3] & 8191) & 4294959104u);
                        float fhi_6 = __uint_as_float(__as_u32(st16_hi_7) + (st16_rb[3] >> 16 & 8191) & 4294959104u);
                        uint32_t word_13;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_13) : "f"(fhi_6), "f"(flo_6));
                        unsigned int st16_fw_7 = word_13;
                        float _abs_181;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_181) : "f"(st16_lo_7));
                        unsigned int s_13 = __as_u32(mul_f32_rz_noftz(_abs_181, __uint_as_float(125829120)));
                        unsigned int _min_182 = ((s_13) < (260046848) ? (s_13) : (260046848));
                        unsigned int h_13 = _min_182 + (st16_rb[3] & 8191) >> 13;
                        unsigned int mag_13 = ((s_13 <= 2139095040) ? h_13 : 32256);
                        float _abs_183;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_183) : "f"(st16_hi_7));
                        unsigned int s_14 = __as_u32(mul_f32_rz_noftz(_abs_183, __uint_as_float(125829120)));
                        unsigned int _min_184 = ((s_14) < (260046848) ? (s_14) : (260046848));
                        unsigned int h_14 = _min_184 + ((st16_rb[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_14 = ((s_14 <= 2139095040) ? h_14 : 32256);
                        uint32_t mags_6;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_6) : "r"(mag_13), "r"(mag_14));
                        uint32_t signs_6;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_6) : "r"(__as_u32(st16_lo_7)), "r"(__as_u32(st16_hi_7)));
                        unsigned int word_14 = signs_6 & 2147516416u | mags_6;
                        unsigned int st16_sw_7 = word_14;
                        unsigned int _max_185 = ((st16_da_7) > (st16_db_7) ? (st16_da_7) : (st16_db_7));
                        st16_pk[0] = ((_max_185 <= 1191182336) ? st16_fw_7 : st16_sw_7);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1096ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        unsigned int key0_1 = st16_key_lo;
                        unsigned int key1_1 = st16_key_hi;
                        long long o_2 = st16_base + 16ll;
                        unsigned long long p0_2 = (unsigned long long)(unsigned int)o_2 * 3528531795ull;
                        st16_w[0] = (unsigned int)(o_2 >> 32ll) ^ key0_1;
                        st16_w[2] = (unsigned int)(p0_2 >> 32ull) ^ key1_1;
                        st16_w[3] = (unsigned int)p0_2;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long q0_2 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long q1_2 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(q1_2 >> 32ull) ^ key0_1;
                        st16_w[1] = (unsigned int)q1_2;
                        st16_w[2] = (unsigned int)(q0_2 >> 32ull) ^ st16_w[3] ^ key1_1;
                        st16_w[3] = (unsigned int)q0_2;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_6 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_6 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_6 >> 32ull) ^ st16_w[1] ^ key0_1;
                        st16_w[1] = (unsigned int)s1_6;
                        st16_w[2] = (unsigned int)(s0_6 >> 32ull) ^ st16_w[3] ^ key1_1;
                        st16_w[3] = (unsigned int)s0_6;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_7 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_7 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_7 >> 32ull) ^ st16_w[1] ^ key0_1;
                        st16_w[1] = (unsigned int)s1_7;
                        st16_w[2] = (unsigned int)(s0_7 >> 32ull) ^ st16_w[3] ^ key1_1;
                        st16_w[3] = (unsigned int)s0_7;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_8 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_8 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_8 >> 32ull) ^ st16_w[1] ^ key0_1;
                        st16_w[1] = (unsigned int)s1_8;
                        st16_w[2] = (unsigned int)(s0_8 >> 32ull) ^ st16_w[3] ^ key1_1;
                        st16_w[3] = (unsigned int)s0_8;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        #pragma unroll 1
                        for (int _round_2 = 5; _round_2 < philox_rounds; _round_2++) {
                            unsigned long long t0_2 = (unsigned long long)st16_w[0] * 3528531795ull;
                            unsigned long long t1_2 = (unsigned long long)st16_w[2] * 3449720151ull;
                            st16_w[0] = (unsigned int)(t1_2 >> 32ull) ^ st16_w[1] ^ key0_1;
                            st16_w[1] = (unsigned int)t1_2;
                            st16_w[2] = (unsigned int)(t0_2 >> 32ull) ^ st16_w[3] ^ key1_1;
                            st16_w[3] = (unsigned int)t0_2;
                            key0_1 = key0_1 + 2654435769u;
                            key1_1 = key1_1 + 3144134277u;
                        }
                        unsigned int st16_give0_2 = ((st16_par != 0) ? st16_w[0] : st16_w[1]);
                        unsigned int st16_give1_2 = ((st16_par != 0) ? st16_w[2] : st16_w[3]);
                        unsigned int st16_own0_2 = ((st16_par != 0) ? st16_w[1] : st16_w[0]);
                        unsigned int st16_own1_2 = ((st16_par != 0) ? st16_w[3] : st16_w[2]);
                        unsigned int st16_got0_2 = __shfl_xor_sync(0xFFFFFFFF, st16_give0_2, 1);
                        unsigned int st16_got1_2 = __shfl_xor_sync(0xFFFFFFFF, st16_give1_2, 1);
                        st16_rb[0] = ((st16_par != 0) ? st16_got0_2 : st16_own0_2);
                        st16_rb[1] = ((st16_par != 0) ? st16_own0_2 : st16_got0_2);
                        st16_rb[2] = ((st16_par != 0) ? st16_got1_2 : st16_own1_2);
                        st16_rb[3] = ((st16_par != 0) ? st16_own1_2 : st16_got1_2);
                        float st16_lo_8 = state[8];
                        float st16_hi_8 = state[9];
                        unsigned int st16_da_8 = (__as_u32(st16_lo_8) & 2147483647) - 947912704;
                        unsigned int st16_db_8 = (__as_u32(st16_hi_8) & 2147483647) - 947912704;
                        float flo_7 = __uint_as_float(__as_u32(st16_lo_8) + (st16_rb[0] & 8191) & 4294959104u);
                        float fhi_7 = __uint_as_float(__as_u32(st16_hi_8) + (st16_rb[0] >> 16 & 8191) & 4294959104u);
                        uint32_t word_15;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_15) : "f"(fhi_7), "f"(flo_7));
                        unsigned int st16_fw_8 = word_15;
                        float _abs_186;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_186) : "f"(st16_lo_8));
                        unsigned int s_15 = __as_u32(mul_f32_rz_noftz(_abs_186, __uint_as_float(125829120)));
                        unsigned int _min_187 = ((s_15) < (260046848) ? (s_15) : (260046848));
                        unsigned int h_15 = _min_187 + (st16_rb[0] & 8191) >> 13;
                        unsigned int mag_15 = ((s_15 <= 2139095040) ? h_15 : 32256);
                        float _abs_188;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_188) : "f"(st16_hi_8));
                        unsigned int s_16 = __as_u32(mul_f32_rz_noftz(_abs_188, __uint_as_float(125829120)));
                        unsigned int _min_189 = ((s_16) < (260046848) ? (s_16) : (260046848));
                        unsigned int h_16 = _min_189 + ((st16_rb[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_16 = ((s_16 <= 2139095040) ? h_16 : 32256);
                        uint32_t mags_7;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_7) : "r"(mag_15), "r"(mag_16));
                        uint32_t signs_7;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_7) : "r"(__as_u32(st16_lo_8)), "r"(__as_u32(st16_hi_8)));
                        unsigned int word_16 = signs_7 & 2147516416u | mags_7;
                        unsigned int st16_sw_8 = word_16;
                        unsigned int _max_190 = ((st16_da_8) > (st16_db_8) ? (st16_da_8) : (st16_db_8));
                        st16_pk[0] = ((_max_190 <= 1191182336) ? st16_fw_8 : st16_sw_8);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 16ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_9 = state[10];
                        float st16_hi_9 = state[11];
                        unsigned int st16_da_9 = (__as_u32(st16_lo_9) & 2147483647) - 947912704;
                        unsigned int st16_db_9 = (__as_u32(st16_hi_9) & 2147483647) - 947912704;
                        float flo_8 = __uint_as_float(__as_u32(st16_lo_9) + (st16_rb[1] & 8191) & 4294959104u);
                        float fhi_8 = __uint_as_float(__as_u32(st16_hi_9) + (st16_rb[1] >> 16 & 8191) & 4294959104u);
                        uint32_t word_17;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_17) : "f"(fhi_8), "f"(flo_8));
                        unsigned int st16_fw_9 = word_17;
                        float _abs_191;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_191) : "f"(st16_lo_9));
                        unsigned int s_17 = __as_u32(mul_f32_rz_noftz(_abs_191, __uint_as_float(125829120)));
                        unsigned int _min_192 = ((s_17) < (260046848) ? (s_17) : (260046848));
                        unsigned int h_17 = _min_192 + (st16_rb[1] & 8191) >> 13;
                        unsigned int mag_17 = ((s_17 <= 2139095040) ? h_17 : 32256);
                        float _abs_193;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_193) : "f"(st16_hi_9));
                        unsigned int s_18 = __as_u32(mul_f32_rz_noftz(_abs_193, __uint_as_float(125829120)));
                        unsigned int _min_194 = ((s_18) < (260046848) ? (s_18) : (260046848));
                        unsigned int h_18 = _min_194 + ((st16_rb[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_18 = ((s_18 <= 2139095040) ? h_18 : 32256);
                        uint32_t mags_8;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_8) : "r"(mag_17), "r"(mag_18));
                        uint32_t signs_8;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_8) : "r"(__as_u32(st16_lo_9)), "r"(__as_u32(st16_hi_9)));
                        unsigned int word_18 = signs_8 & 2147516416u | mags_8;
                        unsigned int st16_sw_9 = word_18;
                        unsigned int _max_195 = ((st16_da_9) > (st16_db_9) ? (st16_da_9) : (st16_db_9));
                        st16_pk[0] = ((_max_195 <= 1191182336) ? st16_fw_9 : st16_sw_9);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1040ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_10 = state[40];
                        float st16_hi_10 = state[41];
                        unsigned int st16_da_10 = (__as_u32(st16_lo_10) & 2147483647) - 947912704;
                        unsigned int st16_db_10 = (__as_u32(st16_hi_10) & 2147483647) - 947912704;
                        float flo_9 = __uint_as_float(__as_u32(st16_lo_10) + (st16_rb[2] & 8191) & 4294959104u);
                        float fhi_9 = __uint_as_float(__as_u32(st16_hi_10) + (st16_rb[2] >> 16 & 8191) & 4294959104u);
                        uint32_t word_19;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_19) : "f"(fhi_9), "f"(flo_9));
                        unsigned int st16_fw_10 = word_19;
                        float _abs_196;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_196) : "f"(st16_lo_10));
                        unsigned int s_19 = __as_u32(mul_f32_rz_noftz(_abs_196, __uint_as_float(125829120)));
                        unsigned int _min_197 = ((s_19) < (260046848) ? (s_19) : (260046848));
                        unsigned int h_19 = _min_197 + (st16_rb[2] & 8191) >> 13;
                        unsigned int mag_19 = ((s_19 <= 2139095040) ? h_19 : 32256);
                        float _abs_198;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_198) : "f"(st16_hi_10));
                        unsigned int s_20 = __as_u32(mul_f32_rz_noftz(_abs_198, __uint_as_float(125829120)));
                        unsigned int _min_199 = ((s_20) < (260046848) ? (s_20) : (260046848));
                        unsigned int h_20 = _min_199 + ((st16_rb[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_20 = ((s_20 <= 2139095040) ? h_20 : 32256);
                        uint32_t mags_9;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_9) : "r"(mag_19), "r"(mag_20));
                        uint32_t signs_9;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_9) : "r"(__as_u32(st16_lo_10)), "r"(__as_u32(st16_hi_10)));
                        unsigned int word_20 = signs_9 & 2147516416u | mags_9;
                        unsigned int st16_sw_10 = word_20;
                        unsigned int _max_200 = ((st16_da_10) > (st16_db_10) ? (st16_da_10) : (st16_db_10));
                        st16_pk[0] = ((_max_200 <= 1191182336) ? st16_fw_10 : st16_sw_10);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 80ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_11 = state[42];
                        float st16_hi_11 = state[43];
                        unsigned int st16_da_11 = (__as_u32(st16_lo_11) & 2147483647) - 947912704;
                        unsigned int st16_db_11 = (__as_u32(st16_hi_11) & 2147483647) - 947912704;
                        float flo_10 = __uint_as_float(__as_u32(st16_lo_11) + (st16_rb[3] & 8191) & 4294959104u);
                        float fhi_10 = __uint_as_float(__as_u32(st16_hi_11) + (st16_rb[3] >> 16 & 8191) & 4294959104u);
                        uint32_t word_21;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_21) : "f"(fhi_10), "f"(flo_10));
                        unsigned int st16_fw_11 = word_21;
                        float _abs_201;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_201) : "f"(st16_lo_11));
                        unsigned int s_21 = __as_u32(mul_f32_rz_noftz(_abs_201, __uint_as_float(125829120)));
                        unsigned int _min_202 = ((s_21) < (260046848) ? (s_21) : (260046848));
                        unsigned int h_21 = _min_202 + (st16_rb[3] & 8191) >> 13;
                        unsigned int mag_21 = ((s_21 <= 2139095040) ? h_21 : 32256);
                        float _abs_203;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_203) : "f"(st16_hi_11));
                        unsigned int s_22 = __as_u32(mul_f32_rz_noftz(_abs_203, __uint_as_float(125829120)));
                        unsigned int _min_204 = ((s_22) < (260046848) ? (s_22) : (260046848));
                        unsigned int h_22 = _min_204 + ((st16_rb[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_22 = ((s_22 <= 2139095040) ? h_22 : 32256);
                        uint32_t mags_10;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_10) : "r"(mag_21), "r"(mag_22));
                        uint32_t signs_10;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_10) : "r"(__as_u32(st16_lo_11)), "r"(__as_u32(st16_hi_11)));
                        unsigned int word_22 = signs_10 & 2147516416u | mags_10;
                        unsigned int st16_sw_11 = word_22;
                        unsigned int _max_205 = ((st16_da_11) > (st16_db_11) ? (st16_da_11) : (st16_db_11));
                        st16_pk[0] = ((_max_205 <= 1191182336) ? st16_fw_11 : st16_sw_11);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1104ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        unsigned int key0_2 = st16_key_lo;
                        unsigned int key1_2 = st16_key_hi;
                        long long o_3 = st16_base + 24ll;
                        unsigned long long p0_3 = (unsigned long long)(unsigned int)o_3 * 3528531795ull;
                        st16_w[0] = (unsigned int)(o_3 >> 32ll) ^ key0_2;
                        st16_w[2] = (unsigned int)(p0_3 >> 32ull) ^ key1_2;
                        st16_w[3] = (unsigned int)p0_3;
                        key0_2 = key0_2 + 2654435769u;
                        key1_2 = key1_2 + 3144134277u;
                        unsigned long long q0_3 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long q1_3 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(q1_3 >> 32ull) ^ key0_2;
                        st16_w[1] = (unsigned int)q1_3;
                        st16_w[2] = (unsigned int)(q0_3 >> 32ull) ^ st16_w[3] ^ key1_2;
                        st16_w[3] = (unsigned int)q0_3;
                        key0_2 = key0_2 + 2654435769u;
                        key1_2 = key1_2 + 3144134277u;
                        unsigned long long s0_9 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_9 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_9 >> 32ull) ^ st16_w[1] ^ key0_2;
                        st16_w[1] = (unsigned int)s1_9;
                        st16_w[2] = (unsigned int)(s0_9 >> 32ull) ^ st16_w[3] ^ key1_2;
                        st16_w[3] = (unsigned int)s0_9;
                        key0_2 = key0_2 + 2654435769u;
                        key1_2 = key1_2 + 3144134277u;
                        unsigned long long s0_10 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_10 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_10 >> 32ull) ^ st16_w[1] ^ key0_2;
                        st16_w[1] = (unsigned int)s1_10;
                        st16_w[2] = (unsigned int)(s0_10 >> 32ull) ^ st16_w[3] ^ key1_2;
                        st16_w[3] = (unsigned int)s0_10;
                        key0_2 = key0_2 + 2654435769u;
                        key1_2 = key1_2 + 3144134277u;
                        unsigned long long s0_11 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_11 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_11 >> 32ull) ^ st16_w[1] ^ key0_2;
                        st16_w[1] = (unsigned int)s1_11;
                        st16_w[2] = (unsigned int)(s0_11 >> 32ull) ^ st16_w[3] ^ key1_2;
                        st16_w[3] = (unsigned int)s0_11;
                        key0_2 = key0_2 + 2654435769u;
                        key1_2 = key1_2 + 3144134277u;
                        #pragma unroll 1
                        for (int _round_3 = 5; _round_3 < philox_rounds; _round_3++) {
                            unsigned long long t0_3 = (unsigned long long)st16_w[0] * 3528531795ull;
                            unsigned long long t1_3 = (unsigned long long)st16_w[2] * 3449720151ull;
                            st16_w[0] = (unsigned int)(t1_3 >> 32ull) ^ st16_w[1] ^ key0_2;
                            st16_w[1] = (unsigned int)t1_3;
                            st16_w[2] = (unsigned int)(t0_3 >> 32ull) ^ st16_w[3] ^ key1_2;
                            st16_w[3] = (unsigned int)t0_3;
                            key0_2 = key0_2 + 2654435769u;
                            key1_2 = key1_2 + 3144134277u;
                        }
                        unsigned int st16_give0_3 = ((st16_par != 0) ? st16_w[0] : st16_w[1]);
                        unsigned int st16_give1_3 = ((st16_par != 0) ? st16_w[2] : st16_w[3]);
                        unsigned int st16_own0_3 = ((st16_par != 0) ? st16_w[1] : st16_w[0]);
                        unsigned int st16_own1_3 = ((st16_par != 0) ? st16_w[3] : st16_w[2]);
                        unsigned int st16_got0_3 = __shfl_xor_sync(0xFFFFFFFF, st16_give0_3, 1);
                        unsigned int st16_got1_3 = __shfl_xor_sync(0xFFFFFFFF, st16_give1_3, 1);
                        st16_rb[0] = ((st16_par != 0) ? st16_got0_3 : st16_own0_3);
                        st16_rb[1] = ((st16_par != 0) ? st16_own0_3 : st16_got0_3);
                        st16_rb[2] = ((st16_par != 0) ? st16_got1_3 : st16_own1_3);
                        st16_rb[3] = ((st16_par != 0) ? st16_own1_3 : st16_got1_3);
                        float st16_lo_12 = state[12];
                        float st16_hi_12 = state[13];
                        unsigned int st16_da_12 = (__as_u32(st16_lo_12) & 2147483647) - 947912704;
                        unsigned int st16_db_12 = (__as_u32(st16_hi_12) & 2147483647) - 947912704;
                        float flo_11 = __uint_as_float(__as_u32(st16_lo_12) + (st16_rb[0] & 8191) & 4294959104u);
                        float fhi_11 = __uint_as_float(__as_u32(st16_hi_12) + (st16_rb[0] >> 16 & 8191) & 4294959104u);
                        uint32_t word_23;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_23) : "f"(fhi_11), "f"(flo_11));
                        unsigned int st16_fw_12 = word_23;
                        float _abs_206;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_206) : "f"(st16_lo_12));
                        unsigned int s_23 = __as_u32(mul_f32_rz_noftz(_abs_206, __uint_as_float(125829120)));
                        unsigned int _min_207 = ((s_23) < (260046848) ? (s_23) : (260046848));
                        unsigned int h_23 = _min_207 + (st16_rb[0] & 8191) >> 13;
                        unsigned int mag_23 = ((s_23 <= 2139095040) ? h_23 : 32256);
                        float _abs_208;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_208) : "f"(st16_hi_12));
                        unsigned int s_24 = __as_u32(mul_f32_rz_noftz(_abs_208, __uint_as_float(125829120)));
                        unsigned int _min_209 = ((s_24) < (260046848) ? (s_24) : (260046848));
                        unsigned int h_24 = _min_209 + ((st16_rb[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_24 = ((s_24 <= 2139095040) ? h_24 : 32256);
                        uint32_t mags_11;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_11) : "r"(mag_23), "r"(mag_24));
                        uint32_t signs_11;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_11) : "r"(__as_u32(st16_lo_12)), "r"(__as_u32(st16_hi_12)));
                        unsigned int word_24 = signs_11 & 2147516416u | mags_11;
                        unsigned int st16_sw_12 = word_24;
                        unsigned int _max_210 = ((st16_da_12) > (st16_db_12) ? (st16_da_12) : (st16_db_12));
                        st16_pk[0] = ((_max_210 <= 1191182336) ? st16_fw_12 : st16_sw_12);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 24ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_13 = state[14];
                        float st16_hi_13 = state[15];
                        unsigned int st16_da_13 = (__as_u32(st16_lo_13) & 2147483647) - 947912704;
                        unsigned int st16_db_13 = (__as_u32(st16_hi_13) & 2147483647) - 947912704;
                        float flo_12 = __uint_as_float(__as_u32(st16_lo_13) + (st16_rb[1] & 8191) & 4294959104u);
                        float fhi_12 = __uint_as_float(__as_u32(st16_hi_13) + (st16_rb[1] >> 16 & 8191) & 4294959104u);
                        uint32_t word_25;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_25) : "f"(fhi_12), "f"(flo_12));
                        unsigned int st16_fw_13 = word_25;
                        float _abs_211;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_211) : "f"(st16_lo_13));
                        unsigned int s_25 = __as_u32(mul_f32_rz_noftz(_abs_211, __uint_as_float(125829120)));
                        unsigned int _min_212 = ((s_25) < (260046848) ? (s_25) : (260046848));
                        unsigned int h_25 = _min_212 + (st16_rb[1] & 8191) >> 13;
                        unsigned int mag_25 = ((s_25 <= 2139095040) ? h_25 : 32256);
                        float _abs_213;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_213) : "f"(st16_hi_13));
                        unsigned int s_26 = __as_u32(mul_f32_rz_noftz(_abs_213, __uint_as_float(125829120)));
                        unsigned int _min_214 = ((s_26) < (260046848) ? (s_26) : (260046848));
                        unsigned int h_26 = _min_214 + ((st16_rb[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_26 = ((s_26 <= 2139095040) ? h_26 : 32256);
                        uint32_t mags_12;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_12) : "r"(mag_25), "r"(mag_26));
                        uint32_t signs_12;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_12) : "r"(__as_u32(st16_lo_13)), "r"(__as_u32(st16_hi_13)));
                        unsigned int word_26 = signs_12 & 2147516416u | mags_12;
                        unsigned int st16_sw_13 = word_26;
                        unsigned int _max_215 = ((st16_da_13) > (st16_db_13) ? (st16_da_13) : (st16_db_13));
                        st16_pk[0] = ((_max_215 <= 1191182336) ? st16_fw_13 : st16_sw_13);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1048ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_14 = state[44];
                        float st16_hi_14 = state[45];
                        unsigned int st16_da_14 = (__as_u32(st16_lo_14) & 2147483647) - 947912704;
                        unsigned int st16_db_14 = (__as_u32(st16_hi_14) & 2147483647) - 947912704;
                        float flo_13 = __uint_as_float(__as_u32(st16_lo_14) + (st16_rb[2] & 8191) & 4294959104u);
                        float fhi_13 = __uint_as_float(__as_u32(st16_hi_14) + (st16_rb[2] >> 16 & 8191) & 4294959104u);
                        uint32_t word_27;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_27) : "f"(fhi_13), "f"(flo_13));
                        unsigned int st16_fw_14 = word_27;
                        float _abs_216;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_216) : "f"(st16_lo_14));
                        unsigned int s_27 = __as_u32(mul_f32_rz_noftz(_abs_216, __uint_as_float(125829120)));
                        unsigned int _min_217 = ((s_27) < (260046848) ? (s_27) : (260046848));
                        unsigned int h_27 = _min_217 + (st16_rb[2] & 8191) >> 13;
                        unsigned int mag_27 = ((s_27 <= 2139095040) ? h_27 : 32256);
                        float _abs_218;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_218) : "f"(st16_hi_14));
                        unsigned int s_28 = __as_u32(mul_f32_rz_noftz(_abs_218, __uint_as_float(125829120)));
                        unsigned int _min_219 = ((s_28) < (260046848) ? (s_28) : (260046848));
                        unsigned int h_28 = _min_219 + ((st16_rb[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_28 = ((s_28 <= 2139095040) ? h_28 : 32256);
                        uint32_t mags_13;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_13) : "r"(mag_27), "r"(mag_28));
                        uint32_t signs_13;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_13) : "r"(__as_u32(st16_lo_14)), "r"(__as_u32(st16_hi_14)));
                        unsigned int word_28 = signs_13 & 2147516416u | mags_13;
                        unsigned int st16_sw_14 = word_28;
                        unsigned int _max_220 = ((st16_da_14) > (st16_db_14) ? (st16_da_14) : (st16_db_14));
                        st16_pk[0] = ((_max_220 <= 1191182336) ? st16_fw_14 : st16_sw_14);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 88ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_15 = state[46];
                        float st16_hi_15 = state[47];
                        unsigned int st16_da_15 = (__as_u32(st16_lo_15) & 2147483647) - 947912704;
                        unsigned int st16_db_15 = (__as_u32(st16_hi_15) & 2147483647) - 947912704;
                        float flo_14 = __uint_as_float(__as_u32(st16_lo_15) + (st16_rb[3] & 8191) & 4294959104u);
                        float fhi_14 = __uint_as_float(__as_u32(st16_hi_15) + (st16_rb[3] >> 16 & 8191) & 4294959104u);
                        uint32_t word_29;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_29) : "f"(fhi_14), "f"(flo_14));
                        unsigned int st16_fw_15 = word_29;
                        float _abs_221;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_221) : "f"(st16_lo_15));
                        unsigned int s_29 = __as_u32(mul_f32_rz_noftz(_abs_221, __uint_as_float(125829120)));
                        unsigned int _min_222 = ((s_29) < (260046848) ? (s_29) : (260046848));
                        unsigned int h_29 = _min_222 + (st16_rb[3] & 8191) >> 13;
                        unsigned int mag_29 = ((s_29 <= 2139095040) ? h_29 : 32256);
                        float _abs_223;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_223) : "f"(st16_hi_15));
                        unsigned int s_30 = __as_u32(mul_f32_rz_noftz(_abs_223, __uint_as_float(125829120)));
                        unsigned int _min_224 = ((s_30) < (260046848) ? (s_30) : (260046848));
                        unsigned int h_30 = _min_224 + ((st16_rb[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_30 = ((s_30 <= 2139095040) ? h_30 : 32256);
                        uint32_t mags_14;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_14) : "r"(mag_29), "r"(mag_30));
                        uint32_t signs_14;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_14) : "r"(__as_u32(st16_lo_15)), "r"(__as_u32(st16_hi_15)));
                        unsigned int word_30 = signs_14 & 2147516416u | mags_14;
                        unsigned int st16_sw_15 = word_30;
                        unsigned int _max_225 = ((st16_da_15) > (st16_db_15) ? (st16_da_15) : (st16_db_15));
                        st16_pk[0] = ((_max_225 <= 1191182336) ? st16_fw_15 : st16_sw_15);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1112ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        unsigned int key0_3 = st16_key_lo;
                        unsigned int key1_3 = st16_key_hi;
                        long long o_4 = st16_base + 32ll;
                        unsigned long long p0_4 = (unsigned long long)(unsigned int)o_4 * 3528531795ull;
                        st16_w[0] = (unsigned int)(o_4 >> 32ll) ^ key0_3;
                        st16_w[2] = (unsigned int)(p0_4 >> 32ull) ^ key1_3;
                        st16_w[3] = (unsigned int)p0_4;
                        key0_3 = key0_3 + 2654435769u;
                        key1_3 = key1_3 + 3144134277u;
                        unsigned long long q0_4 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long q1_4 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(q1_4 >> 32ull) ^ key0_3;
                        st16_w[1] = (unsigned int)q1_4;
                        st16_w[2] = (unsigned int)(q0_4 >> 32ull) ^ st16_w[3] ^ key1_3;
                        st16_w[3] = (unsigned int)q0_4;
                        key0_3 = key0_3 + 2654435769u;
                        key1_3 = key1_3 + 3144134277u;
                        unsigned long long s0_12 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_12 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_12 >> 32ull) ^ st16_w[1] ^ key0_3;
                        st16_w[1] = (unsigned int)s1_12;
                        st16_w[2] = (unsigned int)(s0_12 >> 32ull) ^ st16_w[3] ^ key1_3;
                        st16_w[3] = (unsigned int)s0_12;
                        key0_3 = key0_3 + 2654435769u;
                        key1_3 = key1_3 + 3144134277u;
                        unsigned long long s0_13 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_13 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_13 >> 32ull) ^ st16_w[1] ^ key0_3;
                        st16_w[1] = (unsigned int)s1_13;
                        st16_w[2] = (unsigned int)(s0_13 >> 32ull) ^ st16_w[3] ^ key1_3;
                        st16_w[3] = (unsigned int)s0_13;
                        key0_3 = key0_3 + 2654435769u;
                        key1_3 = key1_3 + 3144134277u;
                        unsigned long long s0_14 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_14 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_14 >> 32ull) ^ st16_w[1] ^ key0_3;
                        st16_w[1] = (unsigned int)s1_14;
                        st16_w[2] = (unsigned int)(s0_14 >> 32ull) ^ st16_w[3] ^ key1_3;
                        st16_w[3] = (unsigned int)s0_14;
                        key0_3 = key0_3 + 2654435769u;
                        key1_3 = key1_3 + 3144134277u;
                        #pragma unroll 1
                        for (int _round_4 = 5; _round_4 < philox_rounds; _round_4++) {
                            unsigned long long t0_4 = (unsigned long long)st16_w[0] * 3528531795ull;
                            unsigned long long t1_4 = (unsigned long long)st16_w[2] * 3449720151ull;
                            st16_w[0] = (unsigned int)(t1_4 >> 32ull) ^ st16_w[1] ^ key0_3;
                            st16_w[1] = (unsigned int)t1_4;
                            st16_w[2] = (unsigned int)(t0_4 >> 32ull) ^ st16_w[3] ^ key1_3;
                            st16_w[3] = (unsigned int)t0_4;
                            key0_3 = key0_3 + 2654435769u;
                            key1_3 = key1_3 + 3144134277u;
                        }
                        unsigned int st16_give0_4 = ((st16_par != 0) ? st16_w[0] : st16_w[1]);
                        unsigned int st16_give1_4 = ((st16_par != 0) ? st16_w[2] : st16_w[3]);
                        unsigned int st16_own0_4 = ((st16_par != 0) ? st16_w[1] : st16_w[0]);
                        unsigned int st16_own1_4 = ((st16_par != 0) ? st16_w[3] : st16_w[2]);
                        unsigned int st16_got0_4 = __shfl_xor_sync(0xFFFFFFFF, st16_give0_4, 1);
                        unsigned int st16_got1_4 = __shfl_xor_sync(0xFFFFFFFF, st16_give1_4, 1);
                        st16_rb[0] = ((st16_par != 0) ? st16_got0_4 : st16_own0_4);
                        st16_rb[1] = ((st16_par != 0) ? st16_own0_4 : st16_got0_4);
                        st16_rb[2] = ((st16_par != 0) ? st16_got1_4 : st16_own1_4);
                        st16_rb[3] = ((st16_par != 0) ? st16_own1_4 : st16_got1_4);
                        float st16_lo_16 = state[16];
                        float st16_hi_16 = state[17];
                        unsigned int st16_da_16 = (__as_u32(st16_lo_16) & 2147483647) - 947912704;
                        unsigned int st16_db_16 = (__as_u32(st16_hi_16) & 2147483647) - 947912704;
                        float flo_15 = __uint_as_float(__as_u32(st16_lo_16) + (st16_rb[0] & 8191) & 4294959104u);
                        float fhi_15 = __uint_as_float(__as_u32(st16_hi_16) + (st16_rb[0] >> 16 & 8191) & 4294959104u);
                        uint32_t word_31;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_31) : "f"(fhi_15), "f"(flo_15));
                        unsigned int st16_fw_16 = word_31;
                        float _abs_226;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_226) : "f"(st16_lo_16));
                        unsigned int s_31 = __as_u32(mul_f32_rz_noftz(_abs_226, __uint_as_float(125829120)));
                        unsigned int _min_227 = ((s_31) < (260046848) ? (s_31) : (260046848));
                        unsigned int h_31 = _min_227 + (st16_rb[0] & 8191) >> 13;
                        unsigned int mag_31 = ((s_31 <= 2139095040) ? h_31 : 32256);
                        float _abs_228;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_228) : "f"(st16_hi_16));
                        unsigned int s_32 = __as_u32(mul_f32_rz_noftz(_abs_228, __uint_as_float(125829120)));
                        unsigned int _min_229 = ((s_32) < (260046848) ? (s_32) : (260046848));
                        unsigned int h_32 = _min_229 + ((st16_rb[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_32 = ((s_32 <= 2139095040) ? h_32 : 32256);
                        uint32_t mags_15;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_15) : "r"(mag_31), "r"(mag_32));
                        uint32_t signs_15;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_15) : "r"(__as_u32(st16_lo_16)), "r"(__as_u32(st16_hi_16)));
                        unsigned int word_32 = signs_15 & 2147516416u | mags_15;
                        unsigned int st16_sw_16 = word_32;
                        unsigned int _max_230 = ((st16_da_16) > (st16_db_16) ? (st16_da_16) : (st16_db_16));
                        st16_pk[0] = ((_max_230 <= 1191182336) ? st16_fw_16 : st16_sw_16);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 32ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_17 = state[18];
                        float st16_hi_17 = state[19];
                        unsigned int st16_da_17 = (__as_u32(st16_lo_17) & 2147483647) - 947912704;
                        unsigned int st16_db_17 = (__as_u32(st16_hi_17) & 2147483647) - 947912704;
                        float flo_16 = __uint_as_float(__as_u32(st16_lo_17) + (st16_rb[1] & 8191) & 4294959104u);
                        float fhi_16 = __uint_as_float(__as_u32(st16_hi_17) + (st16_rb[1] >> 16 & 8191) & 4294959104u);
                        uint32_t word_33;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_33) : "f"(fhi_16), "f"(flo_16));
                        unsigned int st16_fw_17 = word_33;
                        float _abs_231;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_231) : "f"(st16_lo_17));
                        unsigned int s_33 = __as_u32(mul_f32_rz_noftz(_abs_231, __uint_as_float(125829120)));
                        unsigned int _min_232 = ((s_33) < (260046848) ? (s_33) : (260046848));
                        unsigned int h_33 = _min_232 + (st16_rb[1] & 8191) >> 13;
                        unsigned int mag_33 = ((s_33 <= 2139095040) ? h_33 : 32256);
                        float _abs_233;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_233) : "f"(st16_hi_17));
                        unsigned int s_34 = __as_u32(mul_f32_rz_noftz(_abs_233, __uint_as_float(125829120)));
                        unsigned int _min_234 = ((s_34) < (260046848) ? (s_34) : (260046848));
                        unsigned int h_34 = _min_234 + ((st16_rb[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_34 = ((s_34 <= 2139095040) ? h_34 : 32256);
                        uint32_t mags_16;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_16) : "r"(mag_33), "r"(mag_34));
                        uint32_t signs_16;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_16) : "r"(__as_u32(st16_lo_17)), "r"(__as_u32(st16_hi_17)));
                        unsigned int word_34 = signs_16 & 2147516416u | mags_16;
                        unsigned int st16_sw_17 = word_34;
                        unsigned int _max_235 = ((st16_da_17) > (st16_db_17) ? (st16_da_17) : (st16_db_17));
                        st16_pk[0] = ((_max_235 <= 1191182336) ? st16_fw_17 : st16_sw_17);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1056ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_18 = state[48];
                        float st16_hi_18 = state[49];
                        unsigned int st16_da_18 = (__as_u32(st16_lo_18) & 2147483647) - 947912704;
                        unsigned int st16_db_18 = (__as_u32(st16_hi_18) & 2147483647) - 947912704;
                        float flo_17 = __uint_as_float(__as_u32(st16_lo_18) + (st16_rb[2] & 8191) & 4294959104u);
                        float fhi_17 = __uint_as_float(__as_u32(st16_hi_18) + (st16_rb[2] >> 16 & 8191) & 4294959104u);
                        uint32_t word_35;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_35) : "f"(fhi_17), "f"(flo_17));
                        unsigned int st16_fw_18 = word_35;
                        float _abs_236;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_236) : "f"(st16_lo_18));
                        unsigned int s_35 = __as_u32(mul_f32_rz_noftz(_abs_236, __uint_as_float(125829120)));
                        unsigned int _min_237 = ((s_35) < (260046848) ? (s_35) : (260046848));
                        unsigned int h_35 = _min_237 + (st16_rb[2] & 8191) >> 13;
                        unsigned int mag_35 = ((s_35 <= 2139095040) ? h_35 : 32256);
                        float _abs_238;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_238) : "f"(st16_hi_18));
                        unsigned int s_36 = __as_u32(mul_f32_rz_noftz(_abs_238, __uint_as_float(125829120)));
                        unsigned int _min_239 = ((s_36) < (260046848) ? (s_36) : (260046848));
                        unsigned int h_36 = _min_239 + ((st16_rb[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_36 = ((s_36 <= 2139095040) ? h_36 : 32256);
                        uint32_t mags_17;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_17) : "r"(mag_35), "r"(mag_36));
                        uint32_t signs_17;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_17) : "r"(__as_u32(st16_lo_18)), "r"(__as_u32(st16_hi_18)));
                        unsigned int word_36 = signs_17 & 2147516416u | mags_17;
                        unsigned int st16_sw_18 = word_36;
                        unsigned int _max_240 = ((st16_da_18) > (st16_db_18) ? (st16_da_18) : (st16_db_18));
                        st16_pk[0] = ((_max_240 <= 1191182336) ? st16_fw_18 : st16_sw_18);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 96ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_19 = state[50];
                        float st16_hi_19 = state[51];
                        unsigned int st16_da_19 = (__as_u32(st16_lo_19) & 2147483647) - 947912704;
                        unsigned int st16_db_19 = (__as_u32(st16_hi_19) & 2147483647) - 947912704;
                        float flo_18 = __uint_as_float(__as_u32(st16_lo_19) + (st16_rb[3] & 8191) & 4294959104u);
                        float fhi_18 = __uint_as_float(__as_u32(st16_hi_19) + (st16_rb[3] >> 16 & 8191) & 4294959104u);
                        uint32_t word_37;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_37) : "f"(fhi_18), "f"(flo_18));
                        unsigned int st16_fw_19 = word_37;
                        float _abs_241;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_241) : "f"(st16_lo_19));
                        unsigned int s_37 = __as_u32(mul_f32_rz_noftz(_abs_241, __uint_as_float(125829120)));
                        unsigned int _min_242 = ((s_37) < (260046848) ? (s_37) : (260046848));
                        unsigned int h_37 = _min_242 + (st16_rb[3] & 8191) >> 13;
                        unsigned int mag_37 = ((s_37 <= 2139095040) ? h_37 : 32256);
                        float _abs_243;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_243) : "f"(st16_hi_19));
                        unsigned int s_38 = __as_u32(mul_f32_rz_noftz(_abs_243, __uint_as_float(125829120)));
                        unsigned int _min_244 = ((s_38) < (260046848) ? (s_38) : (260046848));
                        unsigned int h_38 = _min_244 + ((st16_rb[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_38 = ((s_38 <= 2139095040) ? h_38 : 32256);
                        uint32_t mags_18;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_18) : "r"(mag_37), "r"(mag_38));
                        uint32_t signs_18;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_18) : "r"(__as_u32(st16_lo_19)), "r"(__as_u32(st16_hi_19)));
                        unsigned int word_38 = signs_18 & 2147516416u | mags_18;
                        unsigned int st16_sw_19 = word_38;
                        unsigned int _max_245 = ((st16_da_19) > (st16_db_19) ? (st16_da_19) : (st16_db_19));
                        st16_pk[0] = ((_max_245 <= 1191182336) ? st16_fw_19 : st16_sw_19);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1120ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        unsigned int key0_4 = st16_key_lo;
                        unsigned int key1_4 = st16_key_hi;
                        long long o_5 = st16_base + 40ll;
                        unsigned long long p0_5 = (unsigned long long)(unsigned int)o_5 * 3528531795ull;
                        st16_w[0] = (unsigned int)(o_5 >> 32ll) ^ key0_4;
                        st16_w[2] = (unsigned int)(p0_5 >> 32ull) ^ key1_4;
                        st16_w[3] = (unsigned int)p0_5;
                        key0_4 = key0_4 + 2654435769u;
                        key1_4 = key1_4 + 3144134277u;
                        unsigned long long q0_5 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long q1_5 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(q1_5 >> 32ull) ^ key0_4;
                        st16_w[1] = (unsigned int)q1_5;
                        st16_w[2] = (unsigned int)(q0_5 >> 32ull) ^ st16_w[3] ^ key1_4;
                        st16_w[3] = (unsigned int)q0_5;
                        key0_4 = key0_4 + 2654435769u;
                        key1_4 = key1_4 + 3144134277u;
                        unsigned long long s0_15 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_15 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_15 >> 32ull) ^ st16_w[1] ^ key0_4;
                        st16_w[1] = (unsigned int)s1_15;
                        st16_w[2] = (unsigned int)(s0_15 >> 32ull) ^ st16_w[3] ^ key1_4;
                        st16_w[3] = (unsigned int)s0_15;
                        key0_4 = key0_4 + 2654435769u;
                        key1_4 = key1_4 + 3144134277u;
                        unsigned long long s0_16 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_16 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_16 >> 32ull) ^ st16_w[1] ^ key0_4;
                        st16_w[1] = (unsigned int)s1_16;
                        st16_w[2] = (unsigned int)(s0_16 >> 32ull) ^ st16_w[3] ^ key1_4;
                        st16_w[3] = (unsigned int)s0_16;
                        key0_4 = key0_4 + 2654435769u;
                        key1_4 = key1_4 + 3144134277u;
                        unsigned long long s0_17 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_17 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_17 >> 32ull) ^ st16_w[1] ^ key0_4;
                        st16_w[1] = (unsigned int)s1_17;
                        st16_w[2] = (unsigned int)(s0_17 >> 32ull) ^ st16_w[3] ^ key1_4;
                        st16_w[3] = (unsigned int)s0_17;
                        key0_4 = key0_4 + 2654435769u;
                        key1_4 = key1_4 + 3144134277u;
                        #pragma unroll 1
                        for (int _round_5 = 5; _round_5 < philox_rounds; _round_5++) {
                            unsigned long long t0_5 = (unsigned long long)st16_w[0] * 3528531795ull;
                            unsigned long long t1_5 = (unsigned long long)st16_w[2] * 3449720151ull;
                            st16_w[0] = (unsigned int)(t1_5 >> 32ull) ^ st16_w[1] ^ key0_4;
                            st16_w[1] = (unsigned int)t1_5;
                            st16_w[2] = (unsigned int)(t0_5 >> 32ull) ^ st16_w[3] ^ key1_4;
                            st16_w[3] = (unsigned int)t0_5;
                            key0_4 = key0_4 + 2654435769u;
                            key1_4 = key1_4 + 3144134277u;
                        }
                        unsigned int st16_give0_5 = ((st16_par != 0) ? st16_w[0] : st16_w[1]);
                        unsigned int st16_give1_5 = ((st16_par != 0) ? st16_w[2] : st16_w[3]);
                        unsigned int st16_own0_5 = ((st16_par != 0) ? st16_w[1] : st16_w[0]);
                        unsigned int st16_own1_5 = ((st16_par != 0) ? st16_w[3] : st16_w[2]);
                        unsigned int st16_got0_5 = __shfl_xor_sync(0xFFFFFFFF, st16_give0_5, 1);
                        unsigned int st16_got1_5 = __shfl_xor_sync(0xFFFFFFFF, st16_give1_5, 1);
                        st16_rb[0] = ((st16_par != 0) ? st16_got0_5 : st16_own0_5);
                        st16_rb[1] = ((st16_par != 0) ? st16_own0_5 : st16_got0_5);
                        st16_rb[2] = ((st16_par != 0) ? st16_got1_5 : st16_own1_5);
                        st16_rb[3] = ((st16_par != 0) ? st16_own1_5 : st16_got1_5);
                        float st16_lo_20 = state[20];
                        float st16_hi_20 = state[21];
                        unsigned int st16_da_20 = (__as_u32(st16_lo_20) & 2147483647) - 947912704;
                        unsigned int st16_db_20 = (__as_u32(st16_hi_20) & 2147483647) - 947912704;
                        float flo_19 = __uint_as_float(__as_u32(st16_lo_20) + (st16_rb[0] & 8191) & 4294959104u);
                        float fhi_19 = __uint_as_float(__as_u32(st16_hi_20) + (st16_rb[0] >> 16 & 8191) & 4294959104u);
                        uint32_t word_39;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_39) : "f"(fhi_19), "f"(flo_19));
                        unsigned int st16_fw_20 = word_39;
                        float _abs_246;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_246) : "f"(st16_lo_20));
                        unsigned int s_39 = __as_u32(mul_f32_rz_noftz(_abs_246, __uint_as_float(125829120)));
                        unsigned int _min_247 = ((s_39) < (260046848) ? (s_39) : (260046848));
                        unsigned int h_39 = _min_247 + (st16_rb[0] & 8191) >> 13;
                        unsigned int mag_39 = ((s_39 <= 2139095040) ? h_39 : 32256);
                        float _abs_248;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_248) : "f"(st16_hi_20));
                        unsigned int s_40 = __as_u32(mul_f32_rz_noftz(_abs_248, __uint_as_float(125829120)));
                        unsigned int _min_249 = ((s_40) < (260046848) ? (s_40) : (260046848));
                        unsigned int h_40 = _min_249 + ((st16_rb[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_40 = ((s_40 <= 2139095040) ? h_40 : 32256);
                        uint32_t mags_19;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_19) : "r"(mag_39), "r"(mag_40));
                        uint32_t signs_19;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_19) : "r"(__as_u32(st16_lo_20)), "r"(__as_u32(st16_hi_20)));
                        unsigned int word_40 = signs_19 & 2147516416u | mags_19;
                        unsigned int st16_sw_20 = word_40;
                        unsigned int _max_250 = ((st16_da_20) > (st16_db_20) ? (st16_da_20) : (st16_db_20));
                        st16_pk[0] = ((_max_250 <= 1191182336) ? st16_fw_20 : st16_sw_20);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 40ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_21 = state[22];
                        float st16_hi_21 = state[23];
                        unsigned int st16_da_21 = (__as_u32(st16_lo_21) & 2147483647) - 947912704;
                        unsigned int st16_db_21 = (__as_u32(st16_hi_21) & 2147483647) - 947912704;
                        float flo_20 = __uint_as_float(__as_u32(st16_lo_21) + (st16_rb[1] & 8191) & 4294959104u);
                        float fhi_20 = __uint_as_float(__as_u32(st16_hi_21) + (st16_rb[1] >> 16 & 8191) & 4294959104u);
                        uint32_t word_41;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_41) : "f"(fhi_20), "f"(flo_20));
                        unsigned int st16_fw_21 = word_41;
                        float _abs_251;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_251) : "f"(st16_lo_21));
                        unsigned int s_41 = __as_u32(mul_f32_rz_noftz(_abs_251, __uint_as_float(125829120)));
                        unsigned int _min_252 = ((s_41) < (260046848) ? (s_41) : (260046848));
                        unsigned int h_41 = _min_252 + (st16_rb[1] & 8191) >> 13;
                        unsigned int mag_41 = ((s_41 <= 2139095040) ? h_41 : 32256);
                        float _abs_253;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_253) : "f"(st16_hi_21));
                        unsigned int s_42 = __as_u32(mul_f32_rz_noftz(_abs_253, __uint_as_float(125829120)));
                        unsigned int _min_254 = ((s_42) < (260046848) ? (s_42) : (260046848));
                        unsigned int h_42 = _min_254 + ((st16_rb[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_42 = ((s_42 <= 2139095040) ? h_42 : 32256);
                        uint32_t mags_20;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_20) : "r"(mag_41), "r"(mag_42));
                        uint32_t signs_20;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_20) : "r"(__as_u32(st16_lo_21)), "r"(__as_u32(st16_hi_21)));
                        unsigned int word_42 = signs_20 & 2147516416u | mags_20;
                        unsigned int st16_sw_21 = word_42;
                        unsigned int _max_255 = ((st16_da_21) > (st16_db_21) ? (st16_da_21) : (st16_db_21));
                        st16_pk[0] = ((_max_255 <= 1191182336) ? st16_fw_21 : st16_sw_21);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1064ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_22 = state[52];
                        float st16_hi_22 = state[53];
                        unsigned int st16_da_22 = (__as_u32(st16_lo_22) & 2147483647) - 947912704;
                        unsigned int st16_db_22 = (__as_u32(st16_hi_22) & 2147483647) - 947912704;
                        float flo_21 = __uint_as_float(__as_u32(st16_lo_22) + (st16_rb[2] & 8191) & 4294959104u);
                        float fhi_21 = __uint_as_float(__as_u32(st16_hi_22) + (st16_rb[2] >> 16 & 8191) & 4294959104u);
                        uint32_t word_43;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_43) : "f"(fhi_21), "f"(flo_21));
                        unsigned int st16_fw_22 = word_43;
                        float _abs_256;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_256) : "f"(st16_lo_22));
                        unsigned int s_43 = __as_u32(mul_f32_rz_noftz(_abs_256, __uint_as_float(125829120)));
                        unsigned int _min_257 = ((s_43) < (260046848) ? (s_43) : (260046848));
                        unsigned int h_43 = _min_257 + (st16_rb[2] & 8191) >> 13;
                        unsigned int mag_43 = ((s_43 <= 2139095040) ? h_43 : 32256);
                        float _abs_258;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_258) : "f"(st16_hi_22));
                        unsigned int s_44 = __as_u32(mul_f32_rz_noftz(_abs_258, __uint_as_float(125829120)));
                        unsigned int _min_259 = ((s_44) < (260046848) ? (s_44) : (260046848));
                        unsigned int h_44 = _min_259 + ((st16_rb[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_44 = ((s_44 <= 2139095040) ? h_44 : 32256);
                        uint32_t mags_21;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_21) : "r"(mag_43), "r"(mag_44));
                        uint32_t signs_21;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_21) : "r"(__as_u32(st16_lo_22)), "r"(__as_u32(st16_hi_22)));
                        unsigned int word_44 = signs_21 & 2147516416u | mags_21;
                        unsigned int st16_sw_22 = word_44;
                        unsigned int _max_260 = ((st16_da_22) > (st16_db_22) ? (st16_da_22) : (st16_db_22));
                        st16_pk[0] = ((_max_260 <= 1191182336) ? st16_fw_22 : st16_sw_22);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 104ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_23 = state[54];
                        float st16_hi_23 = state[55];
                        unsigned int st16_da_23 = (__as_u32(st16_lo_23) & 2147483647) - 947912704;
                        unsigned int st16_db_23 = (__as_u32(st16_hi_23) & 2147483647) - 947912704;
                        float flo_22 = __uint_as_float(__as_u32(st16_lo_23) + (st16_rb[3] & 8191) & 4294959104u);
                        float fhi_22 = __uint_as_float(__as_u32(st16_hi_23) + (st16_rb[3] >> 16 & 8191) & 4294959104u);
                        uint32_t word_45;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_45) : "f"(fhi_22), "f"(flo_22));
                        unsigned int st16_fw_23 = word_45;
                        float _abs_261;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_261) : "f"(st16_lo_23));
                        unsigned int s_45 = __as_u32(mul_f32_rz_noftz(_abs_261, __uint_as_float(125829120)));
                        unsigned int _min_262 = ((s_45) < (260046848) ? (s_45) : (260046848));
                        unsigned int h_45 = _min_262 + (st16_rb[3] & 8191) >> 13;
                        unsigned int mag_45 = ((s_45 <= 2139095040) ? h_45 : 32256);
                        float _abs_263;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_263) : "f"(st16_hi_23));
                        unsigned int s_46 = __as_u32(mul_f32_rz_noftz(_abs_263, __uint_as_float(125829120)));
                        unsigned int _min_264 = ((s_46) < (260046848) ? (s_46) : (260046848));
                        unsigned int h_46 = _min_264 + ((st16_rb[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_46 = ((s_46 <= 2139095040) ? h_46 : 32256);
                        uint32_t mags_22;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_22) : "r"(mag_45), "r"(mag_46));
                        uint32_t signs_22;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_22) : "r"(__as_u32(st16_lo_23)), "r"(__as_u32(st16_hi_23)));
                        unsigned int word_46 = signs_22 & 2147516416u | mags_22;
                        unsigned int st16_sw_23 = word_46;
                        unsigned int _max_265 = ((st16_da_23) > (st16_db_23) ? (st16_da_23) : (st16_db_23));
                        st16_pk[0] = ((_max_265 <= 1191182336) ? st16_fw_23 : st16_sw_23);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1128ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        unsigned int key0_5 = st16_key_lo;
                        unsigned int key1_5 = st16_key_hi;
                        long long o_6 = st16_base + 48ll;
                        unsigned long long p0_6 = (unsigned long long)(unsigned int)o_6 * 3528531795ull;
                        st16_w[0] = (unsigned int)(o_6 >> 32ll) ^ key0_5;
                        st16_w[2] = (unsigned int)(p0_6 >> 32ull) ^ key1_5;
                        st16_w[3] = (unsigned int)p0_6;
                        key0_5 = key0_5 + 2654435769u;
                        key1_5 = key1_5 + 3144134277u;
                        unsigned long long q0_6 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long q1_6 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(q1_6 >> 32ull) ^ key0_5;
                        st16_w[1] = (unsigned int)q1_6;
                        st16_w[2] = (unsigned int)(q0_6 >> 32ull) ^ st16_w[3] ^ key1_5;
                        st16_w[3] = (unsigned int)q0_6;
                        key0_5 = key0_5 + 2654435769u;
                        key1_5 = key1_5 + 3144134277u;
                        unsigned long long s0_18 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_18 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_18 >> 32ull) ^ st16_w[1] ^ key0_5;
                        st16_w[1] = (unsigned int)s1_18;
                        st16_w[2] = (unsigned int)(s0_18 >> 32ull) ^ st16_w[3] ^ key1_5;
                        st16_w[3] = (unsigned int)s0_18;
                        key0_5 = key0_5 + 2654435769u;
                        key1_5 = key1_5 + 3144134277u;
                        unsigned long long s0_19 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_19 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_19 >> 32ull) ^ st16_w[1] ^ key0_5;
                        st16_w[1] = (unsigned int)s1_19;
                        st16_w[2] = (unsigned int)(s0_19 >> 32ull) ^ st16_w[3] ^ key1_5;
                        st16_w[3] = (unsigned int)s0_19;
                        key0_5 = key0_5 + 2654435769u;
                        key1_5 = key1_5 + 3144134277u;
                        unsigned long long s0_20 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_20 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_20 >> 32ull) ^ st16_w[1] ^ key0_5;
                        st16_w[1] = (unsigned int)s1_20;
                        st16_w[2] = (unsigned int)(s0_20 >> 32ull) ^ st16_w[3] ^ key1_5;
                        st16_w[3] = (unsigned int)s0_20;
                        key0_5 = key0_5 + 2654435769u;
                        key1_5 = key1_5 + 3144134277u;
                        #pragma unroll 1
                        for (int _round_6 = 5; _round_6 < philox_rounds; _round_6++) {
                            unsigned long long t0_6 = (unsigned long long)st16_w[0] * 3528531795ull;
                            unsigned long long t1_6 = (unsigned long long)st16_w[2] * 3449720151ull;
                            st16_w[0] = (unsigned int)(t1_6 >> 32ull) ^ st16_w[1] ^ key0_5;
                            st16_w[1] = (unsigned int)t1_6;
                            st16_w[2] = (unsigned int)(t0_6 >> 32ull) ^ st16_w[3] ^ key1_5;
                            st16_w[3] = (unsigned int)t0_6;
                            key0_5 = key0_5 + 2654435769u;
                            key1_5 = key1_5 + 3144134277u;
                        }
                        unsigned int st16_give0_6 = ((st16_par != 0) ? st16_w[0] : st16_w[1]);
                        unsigned int st16_give1_6 = ((st16_par != 0) ? st16_w[2] : st16_w[3]);
                        unsigned int st16_own0_6 = ((st16_par != 0) ? st16_w[1] : st16_w[0]);
                        unsigned int st16_own1_6 = ((st16_par != 0) ? st16_w[3] : st16_w[2]);
                        unsigned int st16_got0_6 = __shfl_xor_sync(0xFFFFFFFF, st16_give0_6, 1);
                        unsigned int st16_got1_6 = __shfl_xor_sync(0xFFFFFFFF, st16_give1_6, 1);
                        st16_rb[0] = ((st16_par != 0) ? st16_got0_6 : st16_own0_6);
                        st16_rb[1] = ((st16_par != 0) ? st16_own0_6 : st16_got0_6);
                        st16_rb[2] = ((st16_par != 0) ? st16_got1_6 : st16_own1_6);
                        st16_rb[3] = ((st16_par != 0) ? st16_own1_6 : st16_got1_6);
                        float st16_lo_24 = state[24];
                        float st16_hi_24 = state[25];
                        unsigned int st16_da_24 = (__as_u32(st16_lo_24) & 2147483647) - 947912704;
                        unsigned int st16_db_24 = (__as_u32(st16_hi_24) & 2147483647) - 947912704;
                        float flo_23 = __uint_as_float(__as_u32(st16_lo_24) + (st16_rb[0] & 8191) & 4294959104u);
                        float fhi_23 = __uint_as_float(__as_u32(st16_hi_24) + (st16_rb[0] >> 16 & 8191) & 4294959104u);
                        uint32_t word_47;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_47) : "f"(fhi_23), "f"(flo_23));
                        unsigned int st16_fw_24 = word_47;
                        float _abs_266;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_266) : "f"(st16_lo_24));
                        unsigned int s_47 = __as_u32(mul_f32_rz_noftz(_abs_266, __uint_as_float(125829120)));
                        unsigned int _min_267 = ((s_47) < (260046848) ? (s_47) : (260046848));
                        unsigned int h_47 = _min_267 + (st16_rb[0] & 8191) >> 13;
                        unsigned int mag_47 = ((s_47 <= 2139095040) ? h_47 : 32256);
                        float _abs_268;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_268) : "f"(st16_hi_24));
                        unsigned int s_48 = __as_u32(mul_f32_rz_noftz(_abs_268, __uint_as_float(125829120)));
                        unsigned int _min_269 = ((s_48) < (260046848) ? (s_48) : (260046848));
                        unsigned int h_48 = _min_269 + ((st16_rb[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_48 = ((s_48 <= 2139095040) ? h_48 : 32256);
                        uint32_t mags_23;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_23) : "r"(mag_47), "r"(mag_48));
                        uint32_t signs_23;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_23) : "r"(__as_u32(st16_lo_24)), "r"(__as_u32(st16_hi_24)));
                        unsigned int word_48 = signs_23 & 2147516416u | mags_23;
                        unsigned int st16_sw_24 = word_48;
                        unsigned int _max_270 = ((st16_da_24) > (st16_db_24) ? (st16_da_24) : (st16_db_24));
                        st16_pk[0] = ((_max_270 <= 1191182336) ? st16_fw_24 : st16_sw_24);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 48ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_25 = state[26];
                        float st16_hi_25 = state[27];
                        unsigned int st16_da_25 = (__as_u32(st16_lo_25) & 2147483647) - 947912704;
                        unsigned int st16_db_25 = (__as_u32(st16_hi_25) & 2147483647) - 947912704;
                        float flo_24 = __uint_as_float(__as_u32(st16_lo_25) + (st16_rb[1] & 8191) & 4294959104u);
                        float fhi_24 = __uint_as_float(__as_u32(st16_hi_25) + (st16_rb[1] >> 16 & 8191) & 4294959104u);
                        uint32_t word_49;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_49) : "f"(fhi_24), "f"(flo_24));
                        unsigned int st16_fw_25 = word_49;
                        float _abs_271;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_271) : "f"(st16_lo_25));
                        unsigned int s_49 = __as_u32(mul_f32_rz_noftz(_abs_271, __uint_as_float(125829120)));
                        unsigned int _min_272 = ((s_49) < (260046848) ? (s_49) : (260046848));
                        unsigned int h_49 = _min_272 + (st16_rb[1] & 8191) >> 13;
                        unsigned int mag_49 = ((s_49 <= 2139095040) ? h_49 : 32256);
                        float _abs_273;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_273) : "f"(st16_hi_25));
                        unsigned int s_50 = __as_u32(mul_f32_rz_noftz(_abs_273, __uint_as_float(125829120)));
                        unsigned int _min_274 = ((s_50) < (260046848) ? (s_50) : (260046848));
                        unsigned int h_50 = _min_274 + ((st16_rb[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_50 = ((s_50 <= 2139095040) ? h_50 : 32256);
                        uint32_t mags_24;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_24) : "r"(mag_49), "r"(mag_50));
                        uint32_t signs_24;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_24) : "r"(__as_u32(st16_lo_25)), "r"(__as_u32(st16_hi_25)));
                        unsigned int word_50 = signs_24 & 2147516416u | mags_24;
                        unsigned int st16_sw_25 = word_50;
                        unsigned int _max_275 = ((st16_da_25) > (st16_db_25) ? (st16_da_25) : (st16_db_25));
                        st16_pk[0] = ((_max_275 <= 1191182336) ? st16_fw_25 : st16_sw_25);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1072ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_26 = state[56];
                        float st16_hi_26 = state[57];
                        unsigned int st16_da_26 = (__as_u32(st16_lo_26) & 2147483647) - 947912704;
                        unsigned int st16_db_26 = (__as_u32(st16_hi_26) & 2147483647) - 947912704;
                        float flo_25 = __uint_as_float(__as_u32(st16_lo_26) + (st16_rb[2] & 8191) & 4294959104u);
                        float fhi_25 = __uint_as_float(__as_u32(st16_hi_26) + (st16_rb[2] >> 16 & 8191) & 4294959104u);
                        uint32_t word_51;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_51) : "f"(fhi_25), "f"(flo_25));
                        unsigned int st16_fw_26 = word_51;
                        float _abs_276;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_276) : "f"(st16_lo_26));
                        unsigned int s_51 = __as_u32(mul_f32_rz_noftz(_abs_276, __uint_as_float(125829120)));
                        unsigned int _min_277 = ((s_51) < (260046848) ? (s_51) : (260046848));
                        unsigned int h_51 = _min_277 + (st16_rb[2] & 8191) >> 13;
                        unsigned int mag_51 = ((s_51 <= 2139095040) ? h_51 : 32256);
                        float _abs_278;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_278) : "f"(st16_hi_26));
                        unsigned int s_52 = __as_u32(mul_f32_rz_noftz(_abs_278, __uint_as_float(125829120)));
                        unsigned int _min_279 = ((s_52) < (260046848) ? (s_52) : (260046848));
                        unsigned int h_52 = _min_279 + ((st16_rb[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_52 = ((s_52 <= 2139095040) ? h_52 : 32256);
                        uint32_t mags_25;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_25) : "r"(mag_51), "r"(mag_52));
                        uint32_t signs_25;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_25) : "r"(__as_u32(st16_lo_26)), "r"(__as_u32(st16_hi_26)));
                        unsigned int word_52 = signs_25 & 2147516416u | mags_25;
                        unsigned int st16_sw_26 = word_52;
                        unsigned int _max_280 = ((st16_da_26) > (st16_db_26) ? (st16_da_26) : (st16_db_26));
                        st16_pk[0] = ((_max_280 <= 1191182336) ? st16_fw_26 : st16_sw_26);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 112ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_27 = state[58];
                        float st16_hi_27 = state[59];
                        unsigned int st16_da_27 = (__as_u32(st16_lo_27) & 2147483647) - 947912704;
                        unsigned int st16_db_27 = (__as_u32(st16_hi_27) & 2147483647) - 947912704;
                        float flo_26 = __uint_as_float(__as_u32(st16_lo_27) + (st16_rb[3] & 8191) & 4294959104u);
                        float fhi_26 = __uint_as_float(__as_u32(st16_hi_27) + (st16_rb[3] >> 16 & 8191) & 4294959104u);
                        uint32_t word_53;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_53) : "f"(fhi_26), "f"(flo_26));
                        unsigned int st16_fw_27 = word_53;
                        float _abs_281;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_281) : "f"(st16_lo_27));
                        unsigned int s_53 = __as_u32(mul_f32_rz_noftz(_abs_281, __uint_as_float(125829120)));
                        unsigned int _min_282 = ((s_53) < (260046848) ? (s_53) : (260046848));
                        unsigned int h_53 = _min_282 + (st16_rb[3] & 8191) >> 13;
                        unsigned int mag_53 = ((s_53 <= 2139095040) ? h_53 : 32256);
                        float _abs_283;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_283) : "f"(st16_hi_27));
                        unsigned int s_54 = __as_u32(mul_f32_rz_noftz(_abs_283, __uint_as_float(125829120)));
                        unsigned int _min_284 = ((s_54) < (260046848) ? (s_54) : (260046848));
                        unsigned int h_54 = _min_284 + ((st16_rb[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_54 = ((s_54 <= 2139095040) ? h_54 : 32256);
                        uint32_t mags_26;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_26) : "r"(mag_53), "r"(mag_54));
                        uint32_t signs_26;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_26) : "r"(__as_u32(st16_lo_27)), "r"(__as_u32(st16_hi_27)));
                        unsigned int word_54 = signs_26 & 2147516416u | mags_26;
                        unsigned int st16_sw_27 = word_54;
                        unsigned int _max_285 = ((st16_da_27) > (st16_db_27) ? (st16_da_27) : (st16_db_27));
                        st16_pk[0] = ((_max_285 <= 1191182336) ? st16_fw_27 : st16_sw_27);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1136ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        unsigned int key0_6 = st16_key_lo;
                        unsigned int key1_6 = st16_key_hi;
                        long long o_7 = st16_base + 56ll;
                        unsigned long long p0_7 = (unsigned long long)(unsigned int)o_7 * 3528531795ull;
                        st16_w[0] = (unsigned int)(o_7 >> 32ll) ^ key0_6;
                        st16_w[2] = (unsigned int)(p0_7 >> 32ull) ^ key1_6;
                        st16_w[3] = (unsigned int)p0_7;
                        key0_6 = key0_6 + 2654435769u;
                        key1_6 = key1_6 + 3144134277u;
                        unsigned long long q0_7 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long q1_7 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(q1_7 >> 32ull) ^ key0_6;
                        st16_w[1] = (unsigned int)q1_7;
                        st16_w[2] = (unsigned int)(q0_7 >> 32ull) ^ st16_w[3] ^ key1_6;
                        st16_w[3] = (unsigned int)q0_7;
                        key0_6 = key0_6 + 2654435769u;
                        key1_6 = key1_6 + 3144134277u;
                        unsigned long long s0_21 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_21 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_21 >> 32ull) ^ st16_w[1] ^ key0_6;
                        st16_w[1] = (unsigned int)s1_21;
                        st16_w[2] = (unsigned int)(s0_21 >> 32ull) ^ st16_w[3] ^ key1_6;
                        st16_w[3] = (unsigned int)s0_21;
                        key0_6 = key0_6 + 2654435769u;
                        key1_6 = key1_6 + 3144134277u;
                        unsigned long long s0_22 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_22 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_22 >> 32ull) ^ st16_w[1] ^ key0_6;
                        st16_w[1] = (unsigned int)s1_22;
                        st16_w[2] = (unsigned int)(s0_22 >> 32ull) ^ st16_w[3] ^ key1_6;
                        st16_w[3] = (unsigned int)s0_22;
                        key0_6 = key0_6 + 2654435769u;
                        key1_6 = key1_6 + 3144134277u;
                        unsigned long long s0_23 = (unsigned long long)st16_w[0] * 3528531795ull;
                        unsigned long long s1_23 = (unsigned long long)st16_w[2] * 3449720151ull;
                        st16_w[0] = (unsigned int)(s1_23 >> 32ull) ^ st16_w[1] ^ key0_6;
                        st16_w[1] = (unsigned int)s1_23;
                        st16_w[2] = (unsigned int)(s0_23 >> 32ull) ^ st16_w[3] ^ key1_6;
                        st16_w[3] = (unsigned int)s0_23;
                        key0_6 = key0_6 + 2654435769u;
                        key1_6 = key1_6 + 3144134277u;
                        #pragma unroll 1
                        for (int _round_7 = 5; _round_7 < philox_rounds; _round_7++) {
                            unsigned long long t0_7 = (unsigned long long)st16_w[0] * 3528531795ull;
                            unsigned long long t1_7 = (unsigned long long)st16_w[2] * 3449720151ull;
                            st16_w[0] = (unsigned int)(t1_7 >> 32ull) ^ st16_w[1] ^ key0_6;
                            st16_w[1] = (unsigned int)t1_7;
                            st16_w[2] = (unsigned int)(t0_7 >> 32ull) ^ st16_w[3] ^ key1_6;
                            st16_w[3] = (unsigned int)t0_7;
                            key0_6 = key0_6 + 2654435769u;
                            key1_6 = key1_6 + 3144134277u;
                        }
                        unsigned int st16_give0_7 = ((st16_par != 0) ? st16_w[0] : st16_w[1]);
                        unsigned int st16_give1_7 = ((st16_par != 0) ? st16_w[2] : st16_w[3]);
                        unsigned int st16_own0_7 = ((st16_par != 0) ? st16_w[1] : st16_w[0]);
                        unsigned int st16_own1_7 = ((st16_par != 0) ? st16_w[3] : st16_w[2]);
                        unsigned int st16_got0_7 = __shfl_xor_sync(0xFFFFFFFF, st16_give0_7, 1);
                        unsigned int st16_got1_7 = __shfl_xor_sync(0xFFFFFFFF, st16_give1_7, 1);
                        st16_rb[0] = ((st16_par != 0) ? st16_got0_7 : st16_own0_7);
                        st16_rb[1] = ((st16_par != 0) ? st16_own0_7 : st16_got0_7);
                        st16_rb[2] = ((st16_par != 0) ? st16_got1_7 : st16_own1_7);
                        st16_rb[3] = ((st16_par != 0) ? st16_own1_7 : st16_got1_7);
                        float st16_lo_28 = state[28];
                        float st16_hi_28 = state[29];
                        unsigned int st16_da_28 = (__as_u32(st16_lo_28) & 2147483647) - 947912704;
                        unsigned int st16_db_28 = (__as_u32(st16_hi_28) & 2147483647) - 947912704;
                        float flo_27 = __uint_as_float(__as_u32(st16_lo_28) + (st16_rb[0] & 8191) & 4294959104u);
                        float fhi_27 = __uint_as_float(__as_u32(st16_hi_28) + (st16_rb[0] >> 16 & 8191) & 4294959104u);
                        uint32_t word_55;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_55) : "f"(fhi_27), "f"(flo_27));
                        unsigned int st16_fw_28 = word_55;
                        float _abs_286;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_286) : "f"(st16_lo_28));
                        unsigned int s_55 = __as_u32(mul_f32_rz_noftz(_abs_286, __uint_as_float(125829120)));
                        unsigned int _min_287 = ((s_55) < (260046848) ? (s_55) : (260046848));
                        unsigned int h_55 = _min_287 + (st16_rb[0] & 8191) >> 13;
                        unsigned int mag_55 = ((s_55 <= 2139095040) ? h_55 : 32256);
                        float _abs_288;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_288) : "f"(st16_hi_28));
                        unsigned int s_56 = __as_u32(mul_f32_rz_noftz(_abs_288, __uint_as_float(125829120)));
                        unsigned int _min_289 = ((s_56) < (260046848) ? (s_56) : (260046848));
                        unsigned int h_56 = _min_289 + ((st16_rb[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_56 = ((s_56 <= 2139095040) ? h_56 : 32256);
                        uint32_t mags_27;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_27) : "r"(mag_55), "r"(mag_56));
                        uint32_t signs_27;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_27) : "r"(__as_u32(st16_lo_28)), "r"(__as_u32(st16_hi_28)));
                        unsigned int word_56 = signs_27 & 2147516416u | mags_27;
                        unsigned int st16_sw_28 = word_56;
                        unsigned int _max_290 = ((st16_da_28) > (st16_db_28) ? (st16_da_28) : (st16_db_28));
                        st16_pk[0] = ((_max_290 <= 1191182336) ? st16_fw_28 : st16_sw_28);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 56ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_29 = state[30];
                        float st16_hi_29 = state[31];
                        unsigned int st16_da_29 = (__as_u32(st16_lo_29) & 2147483647) - 947912704;
                        unsigned int st16_db_29 = (__as_u32(st16_hi_29) & 2147483647) - 947912704;
                        float flo_28 = __uint_as_float(__as_u32(st16_lo_29) + (st16_rb[1] & 8191) & 4294959104u);
                        float fhi_28 = __uint_as_float(__as_u32(st16_hi_29) + (st16_rb[1] >> 16 & 8191) & 4294959104u);
                        uint32_t word_57;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_57) : "f"(fhi_28), "f"(flo_28));
                        unsigned int st16_fw_29 = word_57;
                        float _abs_291;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_291) : "f"(st16_lo_29));
                        unsigned int s_57 = __as_u32(mul_f32_rz_noftz(_abs_291, __uint_as_float(125829120)));
                        unsigned int _min_292 = ((s_57) < (260046848) ? (s_57) : (260046848));
                        unsigned int h_57 = _min_292 + (st16_rb[1] & 8191) >> 13;
                        unsigned int mag_57 = ((s_57 <= 2139095040) ? h_57 : 32256);
                        float _abs_293;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_293) : "f"(st16_hi_29));
                        unsigned int s_58 = __as_u32(mul_f32_rz_noftz(_abs_293, __uint_as_float(125829120)));
                        unsigned int _min_294 = ((s_58) < (260046848) ? (s_58) : (260046848));
                        unsigned int h_58 = _min_294 + ((st16_rb[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_58 = ((s_58 <= 2139095040) ? h_58 : 32256);
                        uint32_t mags_28;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_28) : "r"(mag_57), "r"(mag_58));
                        uint32_t signs_28;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_28) : "r"(__as_u32(st16_lo_29)), "r"(__as_u32(st16_hi_29)));
                        unsigned int word_58 = signs_28 & 2147516416u | mags_28;
                        unsigned int st16_sw_29 = word_58;
                        unsigned int _max_295 = ((st16_da_29) > (st16_db_29) ? (st16_da_29) : (st16_db_29));
                        st16_pk[0] = ((_max_295 <= 1191182336) ? st16_fw_29 : st16_sw_29);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1080ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_30 = state[60];
                        float st16_hi_30 = state[61];
                        unsigned int st16_da_30 = (__as_u32(st16_lo_30) & 2147483647) - 947912704;
                        unsigned int st16_db_30 = (__as_u32(st16_hi_30) & 2147483647) - 947912704;
                        float flo_29 = __uint_as_float(__as_u32(st16_lo_30) + (st16_rb[2] & 8191) & 4294959104u);
                        float fhi_29 = __uint_as_float(__as_u32(st16_hi_30) + (st16_rb[2] >> 16 & 8191) & 4294959104u);
                        uint32_t word_59;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_59) : "f"(fhi_29), "f"(flo_29));
                        unsigned int st16_fw_30 = word_59;
                        float _abs_296;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_296) : "f"(st16_lo_30));
                        unsigned int s_59 = __as_u32(mul_f32_rz_noftz(_abs_296, __uint_as_float(125829120)));
                        unsigned int _min_297 = ((s_59) < (260046848) ? (s_59) : (260046848));
                        unsigned int h_59 = _min_297 + (st16_rb[2] & 8191) >> 13;
                        unsigned int mag_59 = ((s_59 <= 2139095040) ? h_59 : 32256);
                        float _abs_298;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_298) : "f"(st16_hi_30));
                        unsigned int s_60 = __as_u32(mul_f32_rz_noftz(_abs_298, __uint_as_float(125829120)));
                        unsigned int _min_299 = ((s_60) < (260046848) ? (s_60) : (260046848));
                        unsigned int h_60 = _min_299 + ((st16_rb[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_60 = ((s_60 <= 2139095040) ? h_60 : 32256);
                        uint32_t mags_29;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_29) : "r"(mag_59), "r"(mag_60));
                        uint32_t signs_29;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_29) : "r"(__as_u32(st16_lo_30)), "r"(__as_u32(st16_hi_30)));
                        unsigned int word_60 = signs_29 & 2147516416u | mags_29;
                        unsigned int st16_sw_30 = word_60;
                        unsigned int _max_300 = ((st16_da_30) > (st16_db_30) ? (st16_da_30) : (st16_db_30));
                        st16_pk[0] = ((_max_300 <= 1191182336) ? st16_fw_30 : st16_sw_30);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 120ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                        float st16_lo_31 = state[62];
                        float st16_hi_31 = state[63];
                        unsigned int st16_da_31 = (__as_u32(st16_lo_31) & 2147483647) - 947912704;
                        unsigned int st16_db_31 = (__as_u32(st16_hi_31) & 2147483647) - 947912704;
                        float flo_30 = __uint_as_float(__as_u32(st16_lo_31) + (st16_rb[3] & 8191) & 4294959104u);
                        float fhi_30 = __uint_as_float(__as_u32(st16_hi_31) + (st16_rb[3] >> 16 & 8191) & 4294959104u);
                        uint32_t word_61;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_61) : "f"(fhi_30), "f"(flo_30));
                        unsigned int st16_fw_31 = word_61;
                        float _abs_301;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_301) : "f"(st16_lo_31));
                        unsigned int s_61 = __as_u32(mul_f32_rz_noftz(_abs_301, __uint_as_float(125829120)));
                        unsigned int _min_302 = ((s_61) < (260046848) ? (s_61) : (260046848));
                        unsigned int h_61 = _min_302 + (st16_rb[3] & 8191) >> 13;
                        unsigned int mag_61 = ((s_61 <= 2139095040) ? h_61 : 32256);
                        float _abs_303;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_303) : "f"(st16_hi_31));
                        unsigned int s_62 = __as_u32(mul_f32_rz_noftz(_abs_303, __uint_as_float(125829120)));
                        unsigned int _min_304 = ((s_62) < (260046848) ? (s_62) : (260046848));
                        unsigned int h_62 = _min_304 + ((st16_rb[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_62 = ((s_62 <= 2139095040) ? h_62 : 32256);
                        uint32_t mags_30;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_30) : "r"(mag_61), "r"(mag_62));
                        uint32_t signs_30;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_30) : "r"(__as_u32(st16_lo_31)), "r"(__as_u32(st16_hi_31)));
                        unsigned int word_62 = signs_30 & 2147516416u | mags_30;
                        unsigned int st16_sw_31 = word_62;
                        unsigned int _max_305 = ((st16_da_31) > (st16_db_31) ? (st16_da_31) : (st16_db_31));
                        st16_pk[0] = ((_max_305 <= 1191182336) ? st16_fw_31 : st16_sw_31);
                        reinterpret_cast<unsigned*>(output_state + (st16_top + 1144ll))[0] = reinterpret_cast<unsigned*>(st16_pk)[0];
                    }
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
                        : "l"(pa_lo + 2ull), "l"(pb_lo + 2ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                        : "l"(pa_lo + 4ull), "l"(pb_lo + 4ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                        : "l"(pa_lo + 6ull), "l"(pb_lo + 6ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                        : "l"(pa_hi), "l"(pb_hi)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                        : "l"(pa_hi + 2ull), "l"(pb_hi + 2ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                        : "l"(pa_hi + 4ull), "l"(pb_hi + 4ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                        : "l"(pa_hi + 6ull), "l"(pb_hi + 6ull)
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
                        : "l"(pa_lo_0 + 2ull), "l"(pb_lo_0 + 2ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                        : "l"(pa_lo_0 + 4ull), "l"(pb_lo_0 + 4ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                        : "l"(pa_lo_0 + 6ull), "l"(pb_lo_0 + 6ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                        : "l"(pa_hi_0), "l"(pb_hi_0)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                        : "l"(pa_hi_0 + 2ull), "l"(pb_hi_0 + 2ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                        : "l"(pa_hi_0 + 4ull), "l"(pb_hi_0 + 4ull)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                        : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                        : "l"(pa_hi_0 + 6ull), "l"(pb_hi_0 + 6ull)
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
                                cs16[2 * i_3 + e_3] = smem_gate[a_gb + i_3 * 8 + a_quad * 2 + e_3];
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
                        float scs0 = smem_gate[a_gb + srow0];
                        float scs1 = smem_gate[a_gb + srow1];
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
                            for (int word_63 = 0; word_63 < 4; word_63++) {
                                unsigned int hi_bits_3 = words_1[word_63] >> 16;
                                __half lo_f16_3 = 0.0f;
                                __half hi_f16_3 = 0.0f;
                                unsigned int _bit_cast_src_12 = words_1[word_63];
                                lo_f16_3 = reinterpret_cast<__half*>(&_bit_cast_src_12)[0];
                                hi_f16_3 = reinterpret_cast<__half*>(&hi_bits_3)[0];
                                inv_row[word_63 * 2] = (float)lo_f16_3;
                                inv_row[word_63 * 2 + 1] = (float)hi_f16_3;
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
                            : "l"(pa_lo_1 + 2ull), "l"(pb_lo_1 + 2ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                            : "l"(pa_lo_1 + 4ull), "l"(pb_lo_1 + 4ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                            : "l"(pa_lo_1 + 6ull), "l"(pb_lo_1 + 6ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                            : "l"(pa_hi_1), "l"(pb_hi_1)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                            : "l"(pa_hi_1 + 2ull), "l"(pb_hi_1 + 2ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                            : "l"(pa_hi_1 + 4ull), "l"(pb_hi_1 + 4ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_kk[0]), "+f"(acc_kk[1]), "+f"(acc_kk[2]), "+f"(acc_kk[3]), "+f"(acc_kk[4]), "+f"(acc_kk[5]), "+f"(acc_kk[6]), "+f"(acc_kk[7]), "+f"(acc_kk[8]), "+f"(acc_kk[9]), "+f"(acc_kk[10]), "+f"(acc_kk[11]), "+f"(acc_kk[12]), "+f"(acc_kk[13]), "+f"(acc_kk[14]), "+f"(acc_kk[15]), "+f"(acc_kk[16]), "+f"(acc_kk[17]), "+f"(acc_kk[18]), "+f"(acc_kk[19]), "+f"(acc_kk[20]), "+f"(acc_kk[21]), "+f"(acc_kk[22]), "+f"(acc_kk[23]), "+f"(acc_kk[24]), "+f"(acc_kk[25]), "+f"(acc_kk[26]), "+f"(acc_kk[27]), "+f"(acc_kk[28]), "+f"(acc_kk[29]), "+f"(acc_kk[30]), "+f"(acc_kk[31])
                            : "l"(pa_hi_1 + 6ull), "l"(pb_hi_1 + 6ull)
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
                            : "l"(pa_lo_2 + 2ull), "l"(pb_lo_2 + 2ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                            : "l"(pa_lo_2 + 4ull), "l"(pb_lo_2 + 4ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                            : "l"(pa_lo_2 + 6ull), "l"(pb_lo_2 + 6ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                            : "l"(pa_hi_2), "l"(pb_hi_2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                            : "l"(pa_hi_2 + 2ull), "l"(pb_hi_2 + 2ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                            : "l"(pa_hi_2 + 4ull), "l"(pb_hi_2 + 4ull)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(acc_qk[0]), "+f"(acc_qk[1]), "+f"(acc_qk[2]), "+f"(acc_qk[3]), "+f"(acc_qk[4]), "+f"(acc_qk[5]), "+f"(acc_qk[6]), "+f"(acc_qk[7]), "+f"(acc_qk[8]), "+f"(acc_qk[9]), "+f"(acc_qk[10]), "+f"(acc_qk[11]), "+f"(acc_qk[12]), "+f"(acc_qk[13]), "+f"(acc_qk[14]), "+f"(acc_qk[15]), "+f"(acc_qk[16]), "+f"(acc_qk[17]), "+f"(acc_qk[18]), "+f"(acc_qk[19]), "+f"(acc_qk[20]), "+f"(acc_qk[21]), "+f"(acc_qk[22]), "+f"(acc_qk[23]), "+f"(acc_qk[24]), "+f"(acc_qk[25]), "+f"(acc_qk[26]), "+f"(acc_qk[27]), "+f"(acc_qk[28]), "+f"(acc_qk[29]), "+f"(acc_qk[30]), "+f"(acc_qk[31])
                            : "l"(pa_hi_2 + 6ull), "l"(pb_hi_2 + 6ull)
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
                                cs16_1[2 * i_4 + e_4] = smem_gate[a_gb_1 + i_4 * 8 + a_quad * 2 + e_4];
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
                        float scs0_1 = smem_gate[a_gb_1 + srow0_1];
                        float scs1_1 = smem_gate[a_gb_1 + srow1_1];
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
                            for (int word_64 = 0; word_64 < 4; word_64++) {
                                unsigned int hi_bits_4 = words_3[word_64] >> 16;
                                __half lo_f16_4 = 0.0f;
                                __half hi_f16_4 = 0.0f;
                                unsigned int _bit_cast_src_27 = words_3[word_64];
                                lo_f16_4 = reinterpret_cast<__half*>(&_bit_cast_src_27)[0];
                                hi_f16_4 = reinterpret_cast<__half*>(&hi_bits_4)[0];
                                inv_row_1[word_64 * 2] = (float)lo_f16_4;
                                inv_row_1[word_64 * 2 + 1] = (float)hi_f16_4;
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
                        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xFFFFFFFF;" : "=f"(al_carry) : "f"(al_la), "r"(31));
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
                        smem_gate[ac_gb + ac_m] = ac_l;
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
