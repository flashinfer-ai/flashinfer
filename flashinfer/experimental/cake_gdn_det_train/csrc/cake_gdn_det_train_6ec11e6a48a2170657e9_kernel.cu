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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TM_STATE_OFFSET 0
#define TMEM_TM_ACCV_OFFSET 128
#define TMEM_TM_DVL_OFFSET 192
#define TMEM_TM_S_OFFSET 320
#define NUM_RING_STAGES 2
#define NUM_ONE_STAGES 1
#define SMEM_S_KN_OFF 0
#define SMEM_S_KN_STAGE_BYTES 16384
#define SMEM_S_KN_STRIDE 65536
#define SMEM_S_QN_OFF 16384
#define SMEM_S_QN_STAGE_BYTES 16384
#define SMEM_S_QN_STRIDE 65536
#define SMEM_S_QNT_OFF 16384
#define SMEM_S_QNT_STAGE_BYTES 16384
#define SMEM_S_QNT_STRIDE 65536
#define SMEM_S_W_OFF 32768
#define SMEM_S_W_STAGE_BYTES 16384
#define SMEM_S_W_STRIDE 65536
#define SMEM_S_WT_OFF 32768
#define SMEM_S_WT_STAGE_BYTES 16384
#define SMEM_S_WT_STRIDE 65536
#define SMEM_S_DO_OFF 49152
#define SMEM_S_DO_STAGE_BYTES 16384
#define SMEM_S_DO_STRIDE 65536
#define SMEM_S_DOT_OFF 49152
#define SMEM_S_DOT_STAGE_BYTES 16384
#define SMEM_S_DOT_STRIDE 65536
#define SMEM_S_DOG_OFF 131072
#define SMEM_S_DOG_STAGE_BYTES 16384
#define SMEM_S_DOG_STRIDE 16384
#define SMEM_S_DOGT_OFF 131072
#define SMEM_S_DOGT_STAGE_BYTES 16384
#define SMEM_S_DOGT_STRIDE 16384
#define SMEM_S_DHS_OFF 147456
#define SMEM_S_DHS_STAGE_BYTES 32768
#define SMEM_S_DHS_STRIDE 32768
#define SMEM_S_BDVT_OFF 180224
#define SMEM_S_BDVT_STAGE_BYTES 16384
#define SMEM_S_BDVT_STRIDE 16384
#define SMEM_S_BDVS_OFF 196608
#define SMEM_S_BDVS_STAGE_BYTES 16384
#define SMEM_S_BDVS_STRIDE 16384
#define SMEM_S_ST_OFF 212992
#define SMEM_S_ST_STAGE_BYTES 8192
#define SMEM_S_ST_STRIDE 8192
#define SMEM_S_STT_OFF 212992
#define SMEM_S_STT_STAGE_BYTES 8192
#define SMEM_S_STT_STRIDE 8192
#define SMEM_S_G_OFF 221184
#define SMEM_S_G_STAGE_BYTES 512
#define SMEM_S_G_STRIDE 512
#define SMEM_S_DECAY_OFF 221696
#define SMEM_S_DECAY_STAGE_BYTES 512
#define SMEM_S_DECAY_STRIDE 512
#define SMEM_S_GAIN_OFF 222208
#define SMEM_S_GAIN_STAGE_BYTES 512
#define SMEM_S_GAIN_STRIDE 512
#define SMEM_S_GL_OFF 222720
#define SMEM_S_GL_STAGE_BYTES 8
#define SMEM_S_GL_STRIDE 8
#define SMEM_TOTAL 222976
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


__device__ __forceinline__ void tmem_st_x16_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]));
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


extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_det_train_6ec11e6a48a2170657e9(const __grid_constant__ CUtensorMap qn_tma, const __grid_constant__ CUtensorMap kn_tma, const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap do_tma, float* __restrict__ g_cs, float* __restrict__ dht, __nv_bfloat16* __restrict__ dh_out, __nv_bfloat16* __restrict__ dv2, float* __restrict__ dh0, float* __restrict__ seg_state, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int* __restrict__ seg_chunk_start, int* __restrict__ seg_seq, int* __restrict__ seq_chunk_start, int num_heads, int num_v_heads, int use_final_state_grad, int store_initial_state_grad, float scale)
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

    const int mbar_base = smem + 222736;
    #define e_full_addr (mbar_base + 0)
    #define g_full_addr (mbar_base + 16)
    #define q_full_addr (mbar_base + 32)
    #define w_full_addr (mbar_base + 48)
    #define e_empty_addr (mbar_base + 64)
    #define e_free_addr (mbar_base + 80)
    #define l_empty_addr (mbar_base + 96)
    #define tables_ready_addr (mbar_base + 112)
    #define dvl_full_addr (mbar_base + 128)
    #define doG_ready_addr (mbar_base + 144)
    #define tile0_ready_addr (mbar_base + 152)
    #define s_full_addr (mbar_base + 160)
    #define dhs_ready_addr (mbar_base + 168)
    #define dhs_free_addr (mbar_base + 176)
    #define acc_full_addr (mbar_base + 184)
    #define bdv_ready_addr (mbar_base + 192)
    #define bdv_free_addr (mbar_base + 200)
    #define dh_ready_addr (mbar_base + 208)
    #define pu_issued_addr (mbar_base + 216)
    #define tmem_free_addr (mbar_base + 224)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* s_kn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_kn_addr = smem + 0;
    __nv_bfloat16* s_qn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_qn_addr = smem + 16384;
    __nv_bfloat16* s_qnT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_qnT_addr = smem + 16384;
    __nv_bfloat16* s_w = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_w_addr = smem + 32768;
    __nv_bfloat16* s_wT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_wT_addr = smem + 32768;
    __nv_bfloat16* s_do = reinterpret_cast<__nv_bfloat16*>(smem_raw + 49152);
    const int s_do_addr = smem + 49152;
    __nv_bfloat16* s_doT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 49152);
    const int s_doT_addr = smem + 49152;
    __nv_bfloat16* s_dog = reinterpret_cast<__nv_bfloat16*>(smem_raw + 131072);
    const int s_dog_addr = smem + 131072;
    __nv_bfloat16* s_dogT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 131072);
    const int s_dogT_addr = smem + 131072;
    __nv_bfloat16* s_dhs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 147456);
    const int s_dhs_addr = smem + 147456;
    __nv_bfloat16* s_bdvT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 180224);
    const int s_bdvT_addr = smem + 180224;
    __nv_bfloat16* s_bdvs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 196608);
    const int s_bdvs_addr = smem + 196608;
    __nv_bfloat16* s_st = reinterpret_cast<__nv_bfloat16*>(smem_raw + 212992);
    const int s_st_addr = smem + 212992;
    __nv_bfloat16* s_stT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 212992);
    const int s_stT_addr = smem + 212992;
    float* s_g = reinterpret_cast<float*>(smem_raw + 221184);
    const int s_g_addr = smem + 221184;
    float* s_decay = reinterpret_cast<float*>(smem_raw + 221696);
    const int s_decay_addr = smem + 221696;
    float* s_gain = reinterpret_cast<float*>(smem_raw + 222208);
    const int s_gain_addr = smem + 222208;
    float* s_gl = reinterpret_cast<float*>(smem_raw + 222720);
    const int s_gl_addr = smem + 222720;

    // Mbarrier init (20 pipeline groups, 0 ordered-sequence groups, 29 barriers)
    // Mbarriers at smem_raw[222736..222968)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // e_full: 2 barriers, init_count=1
            mbarrier_init(smem + 222736, 1);
            mbarrier_init(smem + 222744, 1);
            // g_full: 2 barriers, init_count=32
            mbarrier_init(smem + 222752, 32);
            mbarrier_init(smem + 222760, 32);
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 222768, 1);
            mbarrier_init(smem + 222776, 1);
            // w_full: 2 barriers, init_count=1
            mbarrier_init(smem + 222784, 1);
            mbarrier_init(smem + 222792, 1);
            // e_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 222800, 1);
            mbarrier_init(smem + 222808, 1);
            // e_free: 2 barriers, init_count=128
            mbarrier_init(smem + 222816, 128);
            mbarrier_init(smem + 222824, 128);
            // l_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 222832, 1);
            mbarrier_init(smem + 222840, 1);
            // tables_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 222848, 128);
            mbarrier_init(smem + 222856, 128);
            // dvl_full: 2 barriers, init_count=1
            mbarrier_init(smem + 222864, 1);
            mbarrier_init(smem + 222872, 1);
            // --- pipeline 'one' ---
            // doG_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 222880, 128);
            // tile0_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 222888, 128);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 222896, 1);
            // dhs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 222904, 128);
            // dhs_free: 1 barriers, init_count=64
            mbarrier_init(smem + 222912, 64);
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 222920, 1);
            // bdv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 222928, 128);
            // bdv_free: 1 barriers, init_count=64
            mbarrier_init(smem + 222936, 64);
            // dh_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 222944, 1);
            // pu_issued: 1 barriers, init_count=1
            mbarrier_init(smem + 222952, 1);
            // tmem_free: 1 barriers, init_count=128
            mbarrier_init(smem + 222960, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 384 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 222968);
    if (warp == 9) {
        int _tmem_hold = smem + 222968;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tm_state = taddr;
    const int tm_accv = taddr + 128;
    const int tm_dvl = taddr + 192;
    const int tm_s = taddr + 320;

    // ---- Role: chain ----
    if (warp <= 3) {
        { // chain_main
            int group = num_v_heads / num_heads;
            int linear = blockIdx.x + blockIdx.y * gridDim.x;
            int span = group * gridDim.x;
            int hq = linear / span;
            int seg = linear % span / group;
            int hv = hq * group + linear % group;
            int c_begin = seg_chunk_start[seg];
            int c_end = seg_chunk_start[seg + 1];
            int nc = c_end - c_begin;
            int c_last = ((c_end - 1) > (0) ? (c_end - 1) : (0));
            int last_tok0 = chunk_start[c_last];
            int last_len = chunk_len[c_last];
            int seq = seg_seq[seg];
            int is_first = 0;
            if (seq >= 0) {
                if (c_begin == seq_chunk_start[seq]) {
                    is_first = 1;
                }
            }
            int lane_c = lane;
            int warp_c = warp;
            int vrow = warp_c * 32 + lane_c;
            int quad = lane_c % 4;
            unsigned int ta_state = (unsigned int)tm_state + (unsigned int)(warp_c * 32 << 16);
            unsigned int ta_accv = (unsigned int)tm_accv + (unsigned int)(warp_c * 32 << 16);
            unsigned int ta_dvl = (unsigned int)tm_dvl + (unsigned int)(warp_c * 32 << 16);
            int st_v = lane_c / 8 % 2 * 8 + lane_c % 8;
            int st_t = lane_c / 16 * 8;
            int stT_t = lane_c / 16 * 8 + lane_c % 8;
            int stT_v = lane_c / 8 % 2 * 8;
            float init_row[16];
            long long in_base = ((long long)seg * (long long)num_v_heads + (long long)hv) * 16384 + (long long)vrow;
            if (use_final_state_grad != 0) {
                #pragma unroll
                for (int blk = 0; blk < 8; blk++) {
                    #pragma unroll
                    for (int kk = 0; kk < 16; kk++) {
                        init_row[kk] = dht[in_base + (long long)((blk * 16 + kk) * 128)];
                    }
                    tmem_st_x16_f32(ta_state + (unsigned int)(blk * 16), init_row);
                }
            } else {
                #pragma unroll
                for (int kk2 = 0; kk2 < 16; kk2++) {
                    init_row[kk2] = 0.0f;
                }
                #pragma unroll
                for (int blk2 = 0; blk2 < 8; blk2++) {
                    tmem_st_x16_f32(ta_state + (unsigned int)(blk2 * 16), init_row);
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int cc = 0;
            float snap[64];
            unsigned int snap_pk[32];
            float accv[32];
            float dvl[32];
            unsigned int dvl_pk[8];
            float ndv[16];
            unsigned int ndv_pk[8];
            unsigned int dv_pk[8];
            unsigned int _phase_tables_ready = 0;
            unsigned int _phase_dhs_free_0 = 1;
            unsigned int _phase_acc_full_0 = 0;
            unsigned int _phase_bdv_free_0 = 1;
            unsigned int _phase_dh_ready_0 = 0;
            #pragma unroll 1
            for (int r = 0; r < nc; r++) {
                int n_c = 64;
                if (r == 0) {
                    n_c = last_len;
                }
                int t_off = (int)cc * 64;
                mbarrier_wait(tables_ready_addr + (cc) * 8, _phase_tables_ready);
                mbarrier_wait(dhs_free_addr, _phase_dhs_free_0);
                _phase_dhs_free_0 ^= 1;
                float gl = s_gl[(int)cc];
                #pragma unroll
                for (int grp = 0; grp < 2; grp++) {
                    tmem_ld_x16(&snap[0], ta_state + (unsigned int)(grp * 64 + 0));
                    tmem_ld_x16(&snap[16], ta_state + (unsigned int)(grp * 64 + 16));
                    tmem_ld_x16(&snap[32], ta_state + (unsigned int)(grp * 64 + 32));
                    tmem_ld_x16(&snap[48], ta_state + (unsigned int)(grp * 64 + 48));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(snap[_lp*2 + 0], snap[_lp*2+1 + 0]));
                        snap_pk[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int c8 = 0; c8 < 8; c8++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_dhs_addr + (unsigned int)((grp * 64 + c8 * 8) / 64 * 16384 + (vrow * 128 + (grp * 64 + c8 * 8) % 64 * 2 ^ (vrow * 128 + (grp * 64 + c8 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(snap_pk[c8 * 4]), "r"(snap_pk[c8 * 4 + 1]), "r"(snap_pk[c8 * 4 + 2]), "r"(snap_pk[c8 * 4 + 3]) : "memory");
                    }
                    const float2 _scale2_0 = {gl, gl};
                    #pragma unroll
                    for (int _ls = 0; _ls < 32; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(snap)[_ls], _scale2_0);
                    #pragma unroll
                    for (int q2 = 0; q2 < 4; q2++) {
                        tmem_st_x16_f32(ta_state + (unsigned int)(grp * 64 + q2 * 16), (snap + q2 * 16));
                    }
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dhs_ready_addr);
                mbarrier_wait(acc_full_addr, _phase_acc_full_0);
                _phase_acc_full_0 ^= 1;
                mbarrier_wait(dvl_full_addr + (cc) * 8, _phase_tables_ready);
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait(bdv_free_addr, _phase_bdv_free_0);
                _phase_bdv_free_0 ^= 1;
                #pragma unroll
                for (int p = 0; p < 2; p++) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&accv[0])), "=r"(*reinterpret_cast<uint32_t*>(&accv[1])), "=r"(*reinterpret_cast<uint32_t*>(&accv[2])), "=r"(*reinterpret_cast<uint32_t*>(&accv[3])), "=r"(*reinterpret_cast<uint32_t*>(&accv[4])), "=r"(*reinterpret_cast<uint32_t*>(&accv[5])), "=r"(*reinterpret_cast<uint32_t*>(&accv[6])), "=r"(*reinterpret_cast<uint32_t*>(&accv[7])), "=r"(*reinterpret_cast<uint32_t*>(&accv[8])), "=r"(*reinterpret_cast<uint32_t*>(&accv[9])), "=r"(*reinterpret_cast<uint32_t*>(&accv[10])), "=r"(*reinterpret_cast<uint32_t*>(&accv[11])), "=r"(*reinterpret_cast<uint32_t*>(&accv[12])), "=r"(*reinterpret_cast<uint32_t*>(&accv[13])), "=r"(*reinterpret_cast<uint32_t*>(&accv[14])), "=r"(*reinterpret_cast<uint32_t*>(&accv[15]))
                        : "r"(ta_accv + (unsigned int)(0 + p * 32)));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dvl[0])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[1])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[2])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[3])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[4])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[5])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[6])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[7])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[8])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[9])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[10])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[11])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[12])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[13])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[14])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[15]))
                        : "r"(ta_dvl + (unsigned int)(t_off + 0 + p * 32)));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&accv[16])), "=r"(*reinterpret_cast<uint32_t*>(&accv[17])), "=r"(*reinterpret_cast<uint32_t*>(&accv[18])), "=r"(*reinterpret_cast<uint32_t*>(&accv[19])), "=r"(*reinterpret_cast<uint32_t*>(&accv[20])), "=r"(*reinterpret_cast<uint32_t*>(&accv[21])), "=r"(*reinterpret_cast<uint32_t*>(&accv[22])), "=r"(*reinterpret_cast<uint32_t*>(&accv[23])), "=r"(*reinterpret_cast<uint32_t*>(&accv[24])), "=r"(*reinterpret_cast<uint32_t*>(&accv[25])), "=r"(*reinterpret_cast<uint32_t*>(&accv[26])), "=r"(*reinterpret_cast<uint32_t*>(&accv[27])), "=r"(*reinterpret_cast<uint32_t*>(&accv[28])), "=r"(*reinterpret_cast<uint32_t*>(&accv[29])), "=r"(*reinterpret_cast<uint32_t*>(&accv[30])), "=r"(*reinterpret_cast<uint32_t*>(&accv[31]))
                        : "r"(ta_accv + (unsigned int)(1048576 + p * 32)));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dvl[16])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[17])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[18])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[19])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[20])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[21])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[22])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[23])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[24])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[25])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[26])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[27])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[28])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[29])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[30])), "=r"(*reinterpret_cast<uint32_t*>(&dvl[31]))
                        : "r"(ta_dvl + (unsigned int)(t_off + 1048576 + p * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int lh2 = 0; lh2 < 2; lh2++) {
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((dvl + lh2 * 16)[_lp*2 + 0], (dvl + lh2 * 16)[_lp*2+1 + 0]));
                            dvl_pk[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            #pragma unroll
                            for (int e = 0; e < 2; e++) {
                                int t_e = p * 32 + i * 8 + quad * 2 + e;
                                float decay_e = s_decay[t_off + t_e];
                                float dl_a = 0.0f;
                                float dl_b = 0.0f;
                                if (t_e < n_c) {
                                    dl_a = ((e == 0) ? __uint_as_float(dvl_pk[i * 2] << 16) : __uint_as_float(dvl_pk[i * 2] & 4294901760u));
                                    dl_b = ((e == 0) ? __uint_as_float(dvl_pk[i * 2 + 1] << 16) : __uint_as_float(dvl_pk[i * 2 + 1] & 4294901760u));
                                }
                                ndv[i * 4 + e] = -__fmaf_rn(accv[lh2 * 16 + i * 4 + e], decay_e, dl_a);
                                ndv[i * 4 + 2 + e] = -__fmaf_rn(accv[lh2 * 16 + i * 4 + 2 + e], decay_e, dl_b);
                            }
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ndv[_lp*2 + 0], ndv[_lp*2+1 + 0]));
                            ndv_pk[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int k8 = 0; k8 < 8; k8++) {
                            dv_pk[k8] = ndv_pk[k8] ^ 2147516416u;
                        }
                        #pragma unroll
                        for (int pp = 0; pp < 2; pp++) {
                            uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(s_bdvT_addr + (unsigned int)((p * 32 + pp * 16 + st_t) / 64 * 16384 + ((warp_c * 32 + lh2 * 16 + st_v) * 128 + (p * 32 + pp * 16 + st_t) % 64 * 2 ^ ((warp_c * 32 + lh2 * 16 + st_v) * 128 + (p * 32 + pp * 16 + st_t) % 64 * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&ndv_pk[pp * 4])), "r"(*reinterpret_cast<const uint32_t*>(&ndv_pk[pp * 4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&ndv_pk[pp * 4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&ndv_pk[pp * 4 + 3]))
                                : "memory");
                            uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(s_bdvs_addr + (unsigned int)((warp_c * 32 + lh2 * 16 + stT_v) / 64 * 8192 + ((p * 32 + pp * 16 + stT_t) * 128 + (warp_c * 32 + lh2 * 16 + stT_v) % 64 * 2 ^ ((p * 32 + pp * 16 + stT_t) * 128 + (warp_c * 32 + lh2 * 16 + stT_v) % 64 * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&dv_pk[pp * 4])), "r"(*reinterpret_cast<const uint32_t*>(&dv_pk[pp * 4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&dv_pk[pp * 4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&dv_pk[pp * 4 + 3]))
                                : "memory");
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(bdv_ready_addr);
                cc += 1;
                if (cc == 2) { cc = 0; _phase_tables_ready ^= 1; }
                mbarrier_wait(dh_ready_addr, _phase_dh_ready_0);
                _phase_dh_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            if (store_initial_state_grad != 0) {
                if (is_first != 0) {
                    int _max_3 = ((seq) > (0) ? (seq) : (0));
                    long long out_base = ((long long)_max_3 * (long long)num_v_heads + (long long)hv) * 16384 + (long long)vrow;
                    #pragma unroll
                    for (int blk3 = 0; blk3 < 8; blk3++) {
                        tmem_ld_x16(&init_row[0], ta_state + (unsigned int)(blk3 * 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int kk3 = 0; kk3 < 16; kk3++) {
                            dh0[out_base + (long long)((blk3 * 16 + kk3) * 128)] = init_row[kk3];
                        }
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_free_addr);
        }
    }
    // ---- Role: aux ----
    if (warp >= 4 && warp <= 7) {
        { // aux_main
            int group_1 = num_v_heads / num_heads;
            int linear_1 = blockIdx.x + blockIdx.y * gridDim.x;
            int span_1 = group_1 * gridDim.x;
            int hq_1 = linear_1 / span_1;
            int seg_1 = linear_1 % span_1 / group_1;
            int hv_1 = hq_1 * group_1 + linear_1 % group_1;
            int c_begin_1 = seg_chunk_start[seg_1];
            int c_end_1 = seg_chunk_start[seg_1 + 1];
            int nc_1 = c_end_1 - c_begin_1;
            int c_last_1 = ((c_end_1 - 1) > (0) ? (c_end_1 - 1) : (0));
            int last_tok0_1 = chunk_start[c_last_1];
            int last_len_1 = chunk_len[c_last_1];
            unsigned int ac = 0;
            int tid_a = tid - 128;
            int lane_a = lane;
            int q_a = warp % 4;
            int row_a = q_a * 16 + lane_a / 4;
            int row_b = row_a + 8;
            int col_q = lane_a % 4 * 2;
            unsigned int ta_s = (unsigned int)tm_s + (unsigned int)(q_a * 32 << 16);
            int st_row = q_a * 16 + lane_a / 8 % 2 * 8 + lane_a % 8;
            int st_col = lane_a / 16 * 8;
            int t_dog = tid_a / 16;
            int c_dog = tid_a % 16 * 8;
            unsigned int fx_cell = (unsigned int)(tid_a % 16 / 8 * 8192 + tid_a % 8 * 16);
            int fx_r0 = tid_a / 16;
            float sacc[32];
            float gex[32];
            float vals[8];
            unsigned int spk[4];
            unsigned int dog[4];
            #pragma unroll
            for (int zk = 0; zk < 4; zk++) {
                spk[zk] = 0;
            }
            if (0 > q_a) {
                uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(s_st_addr + (unsigned int)((0 + st_col) / 64 * 8192 + (st_row * 128 + (0 + st_col) % 64 * 2 ^ (st_row * 128 + (0 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&spk[0])), "r"(*reinterpret_cast<const uint32_t*>(&spk[1])), "r"(*reinterpret_cast<const uint32_t*>(&spk[2])), "r"(*reinterpret_cast<const uint32_t*>(&spk[3]))
                    : "memory");
            }
            if (1 > q_a) {
                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(s_st_addr + (unsigned int)((16 + st_col) / 64 * 8192 + (st_row * 128 + (16 + st_col) % 64 * 2 ^ (st_row * 128 + (16 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&spk[0])), "r"(*reinterpret_cast<const uint32_t*>(&spk[1])), "r"(*reinterpret_cast<const uint32_t*>(&spk[2])), "r"(*reinterpret_cast<const uint32_t*>(&spk[3]))
                    : "memory");
            }
            if (2 > q_a) {
                uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(s_st_addr + (unsigned int)((32 + st_col) / 64 * 8192 + (st_row * 128 + (32 + st_col) % 64 * 2 ^ (st_row * 128 + (32 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&spk[0])), "r"(*reinterpret_cast<const uint32_t*>(&spk[1])), "r"(*reinterpret_cast<const uint32_t*>(&spk[2])), "r"(*reinterpret_cast<const uint32_t*>(&spk[3]))
                    : "memory");
            }
            if (3 > q_a) {
                uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(s_st_addr + (unsigned int)((48 + st_col) / 64 * 8192 + (st_row * 128 + (48 + st_col) % 64 * 2 ^ (st_row * 128 + (48 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&spk[0])), "r"(*reinterpret_cast<const uint32_t*>(&spk[1])), "r"(*reinterpret_cast<const uint32_t*>(&spk[2])), "r"(*reinterpret_cast<const uint32_t*>(&spk[3]))
                    : "memory");
            }
            unsigned int _phase_g_full = 0;
            unsigned int _phase_acc_full_0_1 = 0;
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_bdv_ready_0 = 0;
            unsigned int _phase_pu_issued_0 = 0;
            unsigned int _phase_dh_ready_0_1 = 0;
            #pragma unroll 1
            for (int j = 0; j < nc_1; j++) {
                int n_a = 64;
                if (j == 0) {
                    n_a = last_len_1;
                }
                int g_off = (int)ac * 64;
                mbarrier_wait(g_full_addr + (ac) * 8, _phase_g_full);
                int _max_4 = ((n_a) > (1) ? (n_a) : (1));
                float g_last = s_g[g_off + _max_4 - 1];
                if (tid_a < 64) {
                    float decay = 0.0f;
                    if (tid_a < n_a) {
                        float _exp2_5 = approx_exp2(g_last - s_g[g_off + tid_a]);
                        decay = _exp2_5;
                    }
                    s_decay[g_off + tid_a] = decay;
                } else {
                    float _exp2_6 = approx_exp2(s_g[g_off + tid_a - 64]);
                    s_gain[g_off + tid_a - 64] = _exp2_6 * scale;
                    if (tid_a == 64) {
                        float _exp2_7 = approx_exp2(g_last);
                        s_gl[(int)ac] = _exp2_7;
                    }
                }
                if (j > 0) {
                    mbarrier_wait(acc_full_addr, _phase_acc_full_0_1);
                    _phase_acc_full_0_1 ^= 1;
                }
                mbarrier_arrive(tables_ready_addr + (ac) * 8);
                float g_ra = s_g[g_off + row_a];
                float g_rb = s_g[g_off + row_b];
                if (0 <= q_a) {
                    #pragma unroll
                    for (int qg = 0; qg < 2; qg++) {
                        #pragma unroll
                        for (int eg = 0; eg < 2; eg++) {
                            float g_col = s_g[g_off + 0 + qg * 8 + col_q + eg];
                            float _exp2_8 = approx_exp2(g_ra - g_col);
                            gex[(0 + qg) * 4 + eg] = _exp2_8;
                            float _exp2_9 = approx_exp2(g_rb - g_col);
                            gex[(0 + qg) * 4 + 2 + eg] = _exp2_9;
                        }
                    }
                }
                if (1 <= q_a) {
                    #pragma unroll
                    for (int qg_1 = 0; qg_1 < 2; qg_1++) {
                        #pragma unroll
                        for (int eg_1 = 0; eg_1 < 2; eg_1++) {
                            float g_col_1 = s_g[g_off + 16 + qg_1 * 8 + col_q + eg_1];
                            float _exp2_10 = approx_exp2(g_ra - g_col_1);
                            gex[(2 + qg_1) * 4 + eg_1] = _exp2_10;
                            float _exp2_11 = approx_exp2(g_rb - g_col_1);
                            gex[(2 + qg_1) * 4 + 2 + eg_1] = _exp2_11;
                        }
                    }
                }
                if (2 <= q_a) {
                    #pragma unroll
                    for (int qg_2 = 0; qg_2 < 2; qg_2++) {
                        #pragma unroll
                        for (int eg_2 = 0; eg_2 < 2; eg_2++) {
                            float g_col_2 = s_g[g_off + 32 + qg_2 * 8 + col_q + eg_2];
                            float _exp2_12 = approx_exp2(g_ra - g_col_2);
                            gex[(4 + qg_2) * 4 + eg_2] = _exp2_12;
                            float _exp2_13 = approx_exp2(g_rb - g_col_2);
                            gex[(4 + qg_2) * 4 + 2 + eg_2] = _exp2_13;
                        }
                    }
                }
                if (3 <= q_a) {
                    #pragma unroll
                    for (int qg_3 = 0; qg_3 < 2; qg_3++) {
                        #pragma unroll
                        for (int eg_3 = 0; eg_3 < 2; eg_3++) {
                            float g_col_3 = s_g[g_off + 48 + qg_3 * 8 + col_q + eg_3];
                            float _exp2_14 = approx_exp2(g_ra - g_col_3);
                            gex[(6 + qg_3) * 4 + eg_3] = _exp2_14;
                            float _exp2_15 = approx_exp2(g_rb - g_col_3);
                            gex[(6 + qg_3) * 4 + 2 + eg_3] = _exp2_15;
                        }
                    }
                }
                if (j == 0) {
                    if (n_a < 64) {
                        mbarrier_wait(e_full_addr + (ac) * 8, _phase_g_full);
                        mbarrier_wait(q_full_addr + (ac) * 8, _phase_g_full);
                        mbarrier_wait(w_full_addr + (ac) * 8, _phase_g_full);
                        #pragma unroll 1
                        for (int fx = 0; fx < 8; fx++) {
                            int fx_row = fx * 8 + fx_r0;
                            if (fx_row >= n_a) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + ac * 65536 + fx_cell + (unsigned int)(fx_row * 128)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_qn_addr + ac * 65536 + fx_cell + (unsigned int)(fx_row * 128)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_w_addr + ac * 65536 + fx_cell + (unsigned int)(fx_row * 128)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_do_addr + ac * 65536 + fx_cell + (unsigned int)(fx_row * 128)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    mbarrier_arrive(tile0_ready_addr);
                }
                mbarrier_wait(e_full_addr + (ac) * 8, _phase_g_full);
                mbarrier_wait(q_full_addr + (ac) * 8, _phase_g_full);
                if (j == 0) {
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                }
                if (warp == 4) {
                    uint32_t _elected_16 = elect_sync();
                    asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_16));
                    int _smem_desc_lo_17 = (((s_qn_addr) >> 4) & 0x3FFF) + (ac) * 4096;
                    int _smem_desc_lo_18 = (((s_kn_addr) >> 4) & 0x3FFF) + (ac) * 4096;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    :: "r"(_smem_desc_lo_17), "r"(_smem_desc_lo_18), "r"(tm_s), "r"(0), "r"(_elected_16) : "memory");
                    tcgen05_commit_leader(s_full_addr, _elected_16);
                }
                mbarrier_wait(s_full_addr, _phase_s_full_0);
                _phase_s_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc[0])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[3])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[4])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[5])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[6])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[7])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[8])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[9])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[10])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[11])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[12])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[13])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[14])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[15]))
                    : "r"(ta_s));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                if (q_a >= 2) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[16])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[17])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[18])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[19])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[20])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[21])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[22])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[23])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[24])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[25])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[26])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[27])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[28])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[29])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[30])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[31]))
                        : "r"(ta_s + 32));
                }
                if (0 <= q_a) {
                    #pragma unroll
                    for (int qb = 0; qb < 2; qb++) {
                        #pragma unroll
                        for (int e_1 = 0; e_1 < 2; e_1++) {
                            int col = 0 + qb * 8 + col_q + e_1;
                            float val_a = 0.0f;
                            float val_b = 0.0f;
                            if (col <= row_a) {
                                if (row_a < n_a) {
                                    val_a = sacc[(0 + qb) * 4 + e_1] * gex[(0 + qb) * 4 + e_1] * scale;
                                }
                            }
                            if (col <= row_b) {
                                if (row_b < n_a) {
                                    val_b = sacc[(0 + qb) * 4 + 2 + e_1] * gex[(0 + qb) * 4 + 2 + e_1] * scale;
                                }
                            }
                            vals[qb * 4 + e_1] = val_a;
                            vals[qb * 4 + 2 + e_1] = val_b;
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                        spk[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(s_st_addr + (unsigned int)((0 + st_col) / 64 * 8192 + (st_row * 128 + (0 + st_col) % 64 * 2 ^ (st_row * 128 + (0 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&spk[0])), "r"(*reinterpret_cast<const uint32_t*>(&spk[1])), "r"(*reinterpret_cast<const uint32_t*>(&spk[2])), "r"(*reinterpret_cast<const uint32_t*>(&spk[3]))
                        : "memory");
                }
                if (1 <= q_a) {
                    #pragma unroll
                    for (int qb_1 = 0; qb_1 < 2; qb_1++) {
                        #pragma unroll
                        for (int e_2 = 0; e_2 < 2; e_2++) {
                            int col_1 = 16 + qb_1 * 8 + col_q + e_2;
                            float val_a_1 = 0.0f;
                            float val_b_1 = 0.0f;
                            if (col_1 <= row_a) {
                                if (row_a < n_a) {
                                    val_a_1 = sacc[(2 + qb_1) * 4 + e_2] * gex[(2 + qb_1) * 4 + e_2] * scale;
                                }
                            }
                            if (col_1 <= row_b) {
                                if (row_b < n_a) {
                                    val_b_1 = sacc[(2 + qb_1) * 4 + 2 + e_2] * gex[(2 + qb_1) * 4 + 2 + e_2] * scale;
                                }
                            }
                            vals[qb_1 * 4 + e_2] = val_a_1;
                            vals[qb_1 * 4 + 2 + e_2] = val_b_1;
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                        spk[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_20 = static_cast<uint32_t>(s_st_addr + (unsigned int)((16 + st_col) / 64 * 8192 + (st_row * 128 + (16 + st_col) % 64 * 2 ^ (st_row * 128 + (16 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_20), "r"(*reinterpret_cast<const uint32_t*>(&spk[0])), "r"(*reinterpret_cast<const uint32_t*>(&spk[1])), "r"(*reinterpret_cast<const uint32_t*>(&spk[2])), "r"(*reinterpret_cast<const uint32_t*>(&spk[3]))
                        : "memory");
                }
                if (q_a >= 2) {
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                }
                if (2 <= q_a) {
                    #pragma unroll
                    for (int qb_2 = 0; qb_2 < 2; qb_2++) {
                        #pragma unroll
                        for (int e_3 = 0; e_3 < 2; e_3++) {
                            int col_2 = 32 + qb_2 * 8 + col_q + e_3;
                            float val_a_2 = 0.0f;
                            float val_b_2 = 0.0f;
                            if (col_2 <= row_a) {
                                if (row_a < n_a) {
                                    val_a_2 = sacc[(4 + qb_2) * 4 + e_3] * gex[(4 + qb_2) * 4 + e_3] * scale;
                                }
                            }
                            if (col_2 <= row_b) {
                                if (row_b < n_a) {
                                    val_b_2 = sacc[(4 + qb_2) * 4 + 2 + e_3] * gex[(4 + qb_2) * 4 + 2 + e_3] * scale;
                                }
                            }
                            vals[qb_2 * 4 + e_3] = val_a_2;
                            vals[qb_2 * 4 + 2 + e_3] = val_b_2;
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                        spk[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_21 = static_cast<uint32_t>(s_st_addr + (unsigned int)((32 + st_col) / 64 * 8192 + (st_row * 128 + (32 + st_col) % 64 * 2 ^ (st_row * 128 + (32 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_21), "r"(*reinterpret_cast<const uint32_t*>(&spk[0])), "r"(*reinterpret_cast<const uint32_t*>(&spk[1])), "r"(*reinterpret_cast<const uint32_t*>(&spk[2])), "r"(*reinterpret_cast<const uint32_t*>(&spk[3]))
                        : "memory");
                }
                if (3 <= q_a) {
                    #pragma unroll
                    for (int qb_3 = 0; qb_3 < 2; qb_3++) {
                        #pragma unroll
                        for (int e_4 = 0; e_4 < 2; e_4++) {
                            int col_3 = 48 + qb_3 * 8 + col_q + e_4;
                            float val_a_3 = 0.0f;
                            float val_b_3 = 0.0f;
                            if (col_3 <= row_a) {
                                if (row_a < n_a) {
                                    val_a_3 = sacc[(6 + qb_3) * 4 + e_4] * gex[(6 + qb_3) * 4 + e_4] * scale;
                                }
                            }
                            if (col_3 <= row_b) {
                                if (row_b < n_a) {
                                    val_b_3 = sacc[(6 + qb_3) * 4 + 2 + e_4] * gex[(6 + qb_3) * 4 + 2 + e_4] * scale;
                                }
                            }
                            vals[qb_3 * 4 + e_4] = val_a_3;
                            vals[qb_3 * 4 + 2 + e_4] = val_b_3;
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                        spk[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(s_st_addr + (unsigned int)((48 + st_col) / 64 * 8192 + (st_row * 128 + (48 + st_col) % 64 * 2 ^ (st_row * 128 + (48 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&spk[0])), "r"(*reinterpret_cast<const uint32_t*>(&spk[1])), "r"(*reinterpret_cast<const uint32_t*>(&spk[2])), "r"(*reinterpret_cast<const uint32_t*>(&spk[3]))
                        : "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (j > 0) {
                    mbarrier_wait(bdv_ready_addr, _phase_bdv_ready_0);
                    _phase_bdv_ready_0 ^= 1;
                    mbarrier_wait(pu_issued_addr, _phase_pu_issued_0);
                    _phase_pu_issued_0 ^= 1;
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (warp == 4) {
                    uint32_t _elected_23 = elect_sync();
                    asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_23));
                    int _smem_desc_lo_24 = ((((s_doT_addr) >> 4) & 0x3FFF) | 0x2000000) + (ac) * 4096;
                    int _smem_desc_lo_25 = (((s_stT_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 135365776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_24), "r"(_smem_desc_lo_25), "r"((tm_dvl + ((int)ac * 64))), "r"(0), "r"(_elected_23) : "memory");
                    tcgen05_commit_leader(dvl_full_addr + (ac) * 8, _elected_23);
                }
                if (j > 0) {
                    mbarrier_wait(dh_ready_addr, _phase_dh_ready_0_1);
                    _phase_dh_ready_0_1 ^= 1;
                }
                #pragma unroll
                for (int it = 0; it < 8; it++) {
                    unsigned int dog_off = (unsigned int)(c_dog / 64 * 8192 + ((it * 8 + t_dog) * 128 + c_dog % 64 * 2 ^ ((it * 8 + t_dog) * 128 + c_dog % 64 * 2 >> 7 & 7) << 4));
                    float gain_t = s_gain[g_off + it * 8 + t_dog];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dog[0])), "=r"(*reinterpret_cast<uint32_t*>(&dog[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dog[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dog[(0) + 3]))
                        : "r"(s_do_addr + ac * 65536 + dog_off));
                    uint32_t _bf16x2_scale_pair_26;
                    {
                        uint32_t _bf16x2_pair_27 = dog[0];
                        float _bf16x2_lo_27;
                        float _bf16x2_hi_27;
                        asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_27) : "h"((uint16_t)(_bf16x2_pair_27 & 0xFFFFu)));
                        asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_27) : "h"((uint16_t)(_bf16x2_pair_27 >> 16)));
                        _bf16x2_lo_27 *= gain_t;
                        _bf16x2_hi_27 *= gain_t;
                        uint32_t _bf16x2_out_27;
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_27) : "f"(_bf16x2_hi_27), "f"(_bf16x2_lo_27));
                        _bf16x2_scale_pair_26 = _bf16x2_out_27;
                    }
                    uint32_t _bf16x2_scale_pair_28;
                    {
                        uint32_t _bf16x2_pair_29 = dog[1];
                        float _bf16x2_lo_29;
                        float _bf16x2_hi_29;
                        asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_29) : "h"((uint16_t)(_bf16x2_pair_29 & 0xFFFFu)));
                        asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_29) : "h"((uint16_t)(_bf16x2_pair_29 >> 16)));
                        _bf16x2_lo_29 *= gain_t;
                        _bf16x2_hi_29 *= gain_t;
                        uint32_t _bf16x2_out_29;
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_29) : "f"(_bf16x2_hi_29), "f"(_bf16x2_lo_29));
                        _bf16x2_scale_pair_28 = _bf16x2_out_29;
                    }
                    uint32_t _bf16x2_scale_pair_30;
                    {
                        uint32_t _bf16x2_pair_31 = dog[2];
                        float _bf16x2_lo_31;
                        float _bf16x2_hi_31;
                        asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_31) : "h"((uint16_t)(_bf16x2_pair_31 & 0xFFFFu)));
                        asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_31) : "h"((uint16_t)(_bf16x2_pair_31 >> 16)));
                        _bf16x2_lo_31 *= gain_t;
                        _bf16x2_hi_31 *= gain_t;
                        uint32_t _bf16x2_out_31;
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_31) : "f"(_bf16x2_hi_31), "f"(_bf16x2_lo_31));
                        _bf16x2_scale_pair_30 = _bf16x2_out_31;
                    }
                    uint32_t _bf16x2_scale_pair_32;
                    {
                        uint32_t _bf16x2_pair_33 = dog[3];
                        float _bf16x2_lo_33;
                        float _bf16x2_hi_33;
                        asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_33) : "h"((uint16_t)(_bf16x2_pair_33 & 0xFFFFu)));
                        asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_33) : "h"((uint16_t)(_bf16x2_pair_33 >> 16)));
                        _bf16x2_lo_33 *= gain_t;
                        _bf16x2_hi_33 *= gain_t;
                        uint32_t _bf16x2_out_33;
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_33) : "f"(_bf16x2_hi_33), "f"(_bf16x2_lo_33));
                        _bf16x2_scale_pair_32 = _bf16x2_out_33;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_dog_addr + dog_off), "r"(_bf16x2_scale_pair_26), "r"(_bf16x2_scale_pair_28), "r"(_bf16x2_scale_pair_30), "r"(_bf16x2_scale_pair_32) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(doG_ready_addr);
                mbarrier_wait(dvl_full_addr + (ac) * 8, _phase_g_full);
                mbarrier_arrive(e_free_addr + (ac) * 8);
                ac += 1;
                if (ac == 2) { ac = 0; _phase_g_full ^= 1; }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            int group_2 = num_v_heads / num_heads;
            int linear_2 = blockIdx.x + blockIdx.y * gridDim.x;
            int span_2 = group_2 * gridDim.x;
            int hq_2 = linear_2 / span_2;
            int seg_2 = linear_2 % span_2 / group_2;
            int hv_2 = hq_2 * group_2 + linear_2 % group_2;
            int c_begin_2 = seg_chunk_start[seg_2];
            int c_end_2 = seg_chunk_start[seg_2 + 1];
            int nc_2 = c_end_2 - c_begin_2;
            int c_last_2 = ((c_end_2 - 1) > (0) ? (c_end_2 - 1) : (0));
            int last_tok0_2 = chunk_start[c_last_2];
            int last_len_2 = chunk_len[c_last_2];
            unsigned int lc = 0;
            int lane_l = lane;
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&kn_tma))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&do_tma))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&qn_tma))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&w_tma))) : "memory");
            }
            unsigned int _phase_e_empty = 1;
            #pragma unroll 1
            for (int j_1 = 0; j_1 < nc_2; j_1++) {
                int n_l = 64;
                if (j_1 == 0) {
                    n_l = last_len_2;
                }
                int tok_l = last_tok0_2 - j_1 * 64;
                unsigned int stage_l = lc * 65536;
                mbarrier_wait(e_empty_addr + (lc) * 8, _phase_e_empty);
                mbarrier_wait(e_free_addr + (lc) * 8, _phase_e_empty);
                #pragma unroll
                for (int e_5 = 0; e_5 < 2; e_5++) {
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                        :: "r"(s_g_addr + lc * 256 + (unsigned int)((lane_l + e_5 * 32) * 4)), "l"(g_cs + ((long long)(tok_l + lane_l + e_5 * 32) * (long long)num_v_heads + (long long)hv_2)), "r"((n_l > lane_l + e_5 * 32) ? 4 : 0));
                }
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(g_full_addr + (lc) * 8) : "memory");
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(e_full_addr + (lc) * 8, 32768);
                    tma_3d_gmem2smem(s_kn_addr + stage_l + 0, (&kn_tma), 0, hq_2, tok_l, e_full_addr + (lc) * 8);
                    tma_3d_gmem2smem(s_do_addr + stage_l + 0, (&do_tma), 0, hv_2, tok_l, e_full_addr + (lc) * 8);
                    tma_3d_gmem2smem(s_kn_addr + stage_l + 8192, (&kn_tma), 64, hq_2, tok_l, e_full_addr + (lc) * 8);
                    tma_3d_gmem2smem(s_do_addr + stage_l + 8192, (&do_tma), 64, hv_2, tok_l, e_full_addr + (lc) * 8);
                }
                mbarrier_wait(l_empty_addr + (lc) * 8, _phase_e_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + (lc) * 8, 16384);
                    tma_3d_gmem2smem(s_qn_addr + stage_l + 0, (&qn_tma), 0, hq_2, tok_l, q_full_addr + (lc) * 8);
                    tma_3d_gmem2smem(s_qn_addr + stage_l + 8192, (&qn_tma), 64, hq_2, tok_l, q_full_addr + (lc) * 8);
                    mbarrier_arrive_expect_tx(w_full_addr + (lc) * 8, 16384);
                    tma_3d_gmem2smem(s_w_addr + stage_l + 0, (&w_tma), 0, hv_2, tok_l, w_full_addr + (lc) * 8);
                    tma_3d_gmem2smem(s_w_addr + stage_l + 8192, (&w_tma), 64, hv_2, tok_l, w_full_addr + (lc) * 8);
                }
                lc += 1;
                if (lc == 2) { lc = 0; _phase_e_empty ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            int group_3 = num_v_heads / num_heads;
            int linear_3 = blockIdx.x + blockIdx.y * gridDim.x;
            int span_3 = group_3 * gridDim.x;
            int hq_3 = linear_3 / span_3;
            int seg_3 = linear_3 % span_3 / group_3;
            int hv_3 = hq_3 * group_3 + linear_3 % group_3;
            int c_begin_3 = seg_chunk_start[seg_3];
            int c_end_3 = seg_chunk_start[seg_3 + 1];
            int nc_3 = c_end_3 - c_begin_3;
            int c_last_3 = ((c_end_3 - 1) > (0) ? (c_end_3 - 1) : (0));
            int last_tok0_3 = chunk_start[c_last_3];
            int last_len_3 = chunk_len[c_last_3];
            unsigned int mc = 0;
            unsigned int _phase_e_full = 0;
            unsigned int _phase_dhs_ready_0 = 0;
            unsigned int _phase_doG_ready_0 = 0;
            unsigned int _phase_bdv_ready_0_1 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int r_1 = 0; r_1 < nc_3; r_1++) {
                    mbarrier_wait(e_full_addr + (mc) * 8, _phase_e_full);
                    if (r_1 == 0) {
                        mbarrier_wait(tile0_ready_addr, 0);
                    }
                    mbarrier_wait(dhs_ready_addr, _phase_dhs_ready_0);
                    _phase_dhs_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _smem_desc_lo_0 = ((s_dhs_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_1 = (((s_kn_addr) >> 4) & 0x3FFF) + (mc) * 4096;
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
                    "mov.b32 id, 135267472;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tm_accv), "r"(0) : "memory");
                    tcgen05_commit(acc_full_addr);
                    tcgen05_commit(e_empty_addr + (mc) * 8);
                    mbarrier_wait(q_full_addr + (mc) * 8, _phase_e_full);
                    mbarrier_wait(w_full_addr + (mc) * 8, _phase_e_full);
                    mbarrier_wait(doG_ready_addr, _phase_doG_ready_0);
                    _phase_doG_ready_0 ^= 1;
                    mbarrier_wait(bdv_ready_addr, _phase_bdv_ready_0_1);
                    _phase_bdv_ready_0_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _smem_desc_lo_2 = (((s_dogT_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _smem_desc_lo_3 = ((((s_qnT_addr) >> 4) & 0x3FFF) | 0x2000000) + (mc) * 4096;
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
                    "mov.b32 id, 136414352;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tm_state), "r"(1) : "memory");
                    int _smem_desc_lo_4 = ((s_bdvT_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_5 = ((((s_wT_addr) >> 4) & 0x3FFF) | 0x2000000) + (mc) * 4096;
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
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tm_state), "r"(1) : "memory");
                    int _smem_desc_lo_6 = (((s_dogT_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _smem_desc_lo_7 = ((((s_qnT_addr) >> 4) & 0x3FFF) | 0x2000000) + (mc) * 4096;
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
                    "mov.b32 id, 136414352;\n\t"
                    "add.u32 alo, %0, 128;\n\t"
                    "add.u32 blo, %1, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tm_state), "r"(1) : "memory");
                    int _smem_desc_lo_8 = ((s_bdvT_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_9 = ((((s_wT_addr) >> 4) & 0x3FFF) | 0x2000000) + (mc) * 4096;
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
                    "mov.b32 id, 136381584;\n\t"
                    "add.u32 alo, %0, 2;\n\t"
                    "add.u32 blo, %1, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_8), "r"(_smem_desc_lo_9), "r"(tm_state), "r"(1) : "memory");
                    int _smem_desc_lo_10 = (((s_dogT_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _smem_desc_lo_11 = ((((s_qnT_addr) >> 4) & 0x3FFF) | 0x2000000) + (mc) * 4096;
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
                    "mov.b32 id, 136414352;\n\t"
                    "add.u32 alo, %0, 256;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"(tm_state), "r"(1) : "memory");
                    int _smem_desc_lo_12 = ((s_bdvT_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_13 = ((((s_wT_addr) >> 4) & 0x3FFF) | 0x2000000) + (mc) * 4096;
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
                    "mov.b32 id, 136381584;\n\t"
                    "add.u32 alo, %0, 4;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_12), "r"(_smem_desc_lo_13), "r"(tm_state), "r"(1) : "memory");
                    int _smem_desc_lo_14 = (((s_dogT_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _smem_desc_lo_15 = ((((s_qnT_addr) >> 4) & 0x3FFF) | 0x2000000) + (mc) * 4096;
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
                    "mov.b32 id, 136414352;\n\t"
                    "add.u32 alo, %0, 384;\n\t"
                    "add.u32 blo, %1, 384;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_14), "r"(_smem_desc_lo_15), "r"(tm_state), "r"(1) : "memory");
                    int _smem_desc_lo_16 = ((s_bdvT_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_17_1 = ((((s_wT_addr) >> 4) & 0x3FFF) | 0x2000000) + (mc) * 4096;
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
                    "mov.b32 id, 136381584;\n\t"
                    "add.u32 alo, %0, 6;\n\t"
                    "add.u32 blo, %1, 384;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_16), "r"(_smem_desc_lo_17_1), "r"(tm_state), "r"(1) : "memory");
                    tcgen05_commit(dh_ready_addr);
                    tcgen05_commit(l_empty_addr + (mc) * 8);
                    mbarrier_arrive(pu_issued_addr);
                    mc += 1;
                    if (mc == 2) { mc = 0; _phase_e_full ^= 1; }
                }
            }
            unsigned int _phase_tmem_free_0 = 0;
            mbarrier_wait(tmem_free_addr, _phase_tmem_free_0);
            _phase_tmem_free_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
    // ---- Role: store ----
    if (warp >= 10 && warp <= 11) {
        { // store_main
            int group_4 = num_v_heads / num_heads;
            int linear_4 = blockIdx.x + blockIdx.y * gridDim.x;
            int span_4 = group_4 * gridDim.x;
            int hq_4 = linear_4 / span_4;
            int seg_4 = linear_4 % span_4 / group_4;
            int hv_4 = hq_4 * group_4 + linear_4 % group_4;
            int c_begin_4 = seg_chunk_start[seg_4];
            int c_end_4 = seg_chunk_start[seg_4 + 1];
            int nc_4 = c_end_4 - c_begin_4;
            int c_last_4 = ((c_end_4 - 1) > (0) ? (c_end_4 - 1) : (0));
            int last_tok0_4 = chunk_start[c_last_4];
            int last_len_4 = chunk_len[c_last_4];
            int tid_s = tid - 320;
            int row_s = tid_s / 16;
            int col_s = tid_s % 16 * 8;
            unsigned int hold[64];
            unsigned int _phase_dhs_ready_0_1 = 0;
            unsigned int _phase_bdv_ready_0_2 = 0;
            #pragma unroll 1
            for (int r_2 = 0; r_2 < nc_4; r_2++) {
                int gc = c_end_4 - 1 - r_2;
                int tok_s = last_tok0_4 - r_2 * 64;
                int n_s = 64;
                if (r_2 == 0) {
                    n_s = last_len_4;
                }
                mbarrier_wait(dhs_ready_addr, _phase_dhs_ready_0_1);
                _phase_dhs_ready_0_1 ^= 1;
                long long tape_base = ((long long)gc * (long long)num_v_heads + (long long)hv_4) * 16384 + (long long)(row_s * 128 + col_s);
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    #pragma unroll
                    for (int it_1 = 0; it_1 < 16; it_1++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&hold[it_1 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&hold[(it_1 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&hold[(it_1 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&hold[(it_1 * 4) + 3]))
                            : "r"(s_dhs_addr + (unsigned int)(col_s / 64 * 16384 + ((half * 64 + it_1 * 4 + row_s) * 128 + col_s % 64 * 2 ^ ((half * 64 + it_1 * 4 + row_s) * 128 + col_s % 64 * 2 >> 7 & 7) << 4))));
                    }
                    if (half == 1) {
                        mbarrier_arrive(dhs_free_addr);
                    }
                    #pragma unroll
                    for (int it2 = 0; it2 < 16; it2++) {
                        reinterpret_cast<int4*>(dh_out + (tape_base + (long long)((half * 16 + it2) * 512)))[0] = reinterpret_cast<int4*>(hold + (it2 * 4))[0];
                    }
                }
                mbarrier_wait(bdv_ready_addr, _phase_bdv_ready_0_2);
                _phase_bdv_ready_0_2 ^= 1;
                #pragma unroll
                for (int it3 = 0; it3 < 16; it3++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&hold[it3 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&hold[(it3 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&hold[(it3 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&hold[(it3 * 4) + 3]))
                        : "r"(s_bdvs_addr + (unsigned int)(col_s / 64 * 8192 + ((it3 * 4 + row_s) * 128 + col_s % 64 * 2 ^ ((it3 * 4 + row_s) * 128 + col_s % 64 * 2 >> 7 & 7) << 4))));
                }
                mbarrier_arrive(bdv_free_addr);
                long long dv_base = ((long long)(tok_s + row_s) * (long long)num_v_heads + (long long)hv_4) * 128 + (long long)col_s;
                #pragma unroll
                for (int it4 = 0; it4 < 16; it4++) {
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(dv2 + (dv_base + (long long)(it4 * 4) * (long long)num_v_heads * 128)), "r"(hold[(it4 * 4) + 0]), "r"(hold[(it4 * 4) + 1]), "r"(hold[(it4 * 4) + 2]), "r"(hold[(it4 * 4) + 3]), "r"((unsigned int)(it4 * 4 + row_s < n_s)) : "memory");
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
