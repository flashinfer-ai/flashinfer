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

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
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

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_sage_block_sparse_attention_ef25366cb9d28aad7efb(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, const __grid_constant__ CUtensorMap v, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, float* __restrict__ q_scale, float* __restrict__ k_scale, float* __restrict__ v_scale, int* __restrict__ block_index, int* __restrict__ block_nums, int* __restrict__ block_sizes, int seqlen_q, int seqlen_k, int num_heads, int num_kv_heads, int q_blocks, int k_blocks, int k_scale_groups, int capacity, int block_sparse_num, int has_block_nums, int block_sizes_mode, int v_scale_batch_stride, int total_tiles, float softmax_scale_log2, int return_lse, int split_from)
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
#if __CUDA_ARCH__ == 1030
    #define kv_empty_addr (mbar_base + 96)
    #define s_full_addr (mbar_base + 160)
    #define p_full_addr (mbar_base + 176)
    #define pv_done_addr (mbar_base + 208)
    #define corr_sig_addr (mbar_base + 224)
    #define corr_done_addr (mbar_base + 240)
    #define o_full_addr (mbar_base + 256)
    #define tile_done_addr (mbar_base + 264)
    #define xfer_addr (mbar_base + 272)
    #define s_empty_addr (mbar_base + 280)
    #define meta_full_addr (mbar_base + 296)
    #define meta_empty_addr (mbar_base + 312)
    #define step_full_addr (mbar_base + 328)
    #define step_empty_addr (mbar_base + 392)
#else
    #define kv_empty_addr (mbar_base + 80)
    #define s_full_addr (mbar_base + 128)
    #define p_full_addr (mbar_base + 144)
    #define pv_done_addr (mbar_base + 176)
    #define corr_sig_addr (mbar_base + 192)
    #define corr_done_addr (mbar_base + 208)
    #define o_full_addr (mbar_base + 224)
    #define tile_done_addr (mbar_base + 232)
    #define xfer_addr (mbar_base + 240)
    #define s_empty_addr (mbar_base + 248)
    #define meta_full_addr (mbar_base + 264)
    #define meta_empty_addr (mbar_base + 280)
    #define step_full_addr (mbar_base + 296)
    #define step_empty_addr (mbar_base + 360)
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    uint8_t* q_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    uint8_t* kv_smem = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int kv_smem_addr = smem + 17408;
    uint8_t* v_smem = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int v_smem_addr = smem + 17408;
#if __CUDA_ARCH__ == 1030
    float* scale_smem = reinterpret_cast<float*>(smem_raw + 148480);
    const int scale_smem_addr = smem + 148480;
    float* pf_sink_f = reinterpret_cast<float*>(smem_raw + 151552);
    const int pf_sink_f_addr = smem + 151552;
    int* pf_sink_i = reinterpret_cast<int*>(smem_raw + 151680);
    const int pf_sink_i_addr = smem + 151680;
    float* meta_f = reinterpret_cast<float*>(smem_raw + 151808);
    const int meta_f_addr = smem + 151808;
    int* meta_i = reinterpret_cast<int*>(smem_raw + 151808);
    const int meta_i_addr = smem + 151808;
    float* ring_f = reinterpret_cast<float*>(smem_raw + 152608);
    const int ring_f_addr = smem + 152608;
    int* ring_i = reinterpret_cast<int*>(smem_raw + 152608);
    const int ring_i_addr = smem + 152608;
    float* vs_f = reinterpret_cast<float*>(smem_raw + 152992);
    const int vs_f_addr = smem + 152992;
    float* xbuf = reinterpret_cast<float*>(smem_raw + 154016);
    const int xbuf_addr = smem + 154016;
#else
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
    float* vs_f = reinterpret_cast<float*>(smem_raw + 120224);
    const int vs_f_addr = smem + 120224;
    float* xbuf = reinterpret_cast<float*>(smem_raw + 121248);
    const int xbuf_addr = smem + 121248;
#endif

#if __CUDA_ARCH__ == 1030
    // Mbarrier init (57 barriers at smem_raw[0..456))
#else
    // Mbarrier init (53 barriers at smem_raw[0..424))
#endif

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
#if __CUDA_ARCH__ == 1030
            // kv_full: 8 barriers, init_count=1
#else
            // kv_full: 6 barriers, init_count=1
#endif
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
#if __CUDA_ARCH__ == 1030
#else
            // kv_empty: 6 barriers, init_count=1
#endif
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
#if __CUDA_ARCH__ == 1030
            // kv_empty: 8 barriers, init_count=1
#else
#endif
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
#if __CUDA_ARCH__ == 1030
#else
            // s_full: 2 barriers, init_count=1
#endif
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
#else
#endif
            // p_full: 4 barriers, init_count=256
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 176, 256);
            mbarrier_init(smem + 184, 256);
            mbarrier_init(smem + 192, 256);
            mbarrier_init(smem + 200, 256);
#else
            mbarrier_init(smem + 144, 256);
            mbarrier_init(smem + 152, 256);
            mbarrier_init(smem + 160, 256);
            mbarrier_init(smem + 168, 256);
#endif
            // pv_done: 2 barriers, init_count=1
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
#else
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
#endif
            // corr_sig: 2 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 224, 128);
            mbarrier_init(smem + 232, 128);
#else
            mbarrier_init(smem + 192, 128);
            mbarrier_init(smem + 200, 128);
#endif
            // corr_done: 2 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 240, 128);
            mbarrier_init(smem + 248, 128);
#else
            mbarrier_init(smem + 208, 128);
            mbarrier_init(smem + 216, 128);
#endif
            // o_full: 1 barriers, init_count=1
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 256, 1);
#else
            mbarrier_init(smem + 224, 1);
#endif
            // tile_done: 1 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 264, 128);
#else
            mbarrier_init(smem + 232, 128);
#endif
            // xfer: 1 barriers, init_count=1
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 272, 1);
#else
            mbarrier_init(smem + 240, 1);
#endif
            // s_empty: 2 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 280, 128);
            mbarrier_init(smem + 288, 128);
#else
            mbarrier_init(smem + 248, 128);
            mbarrier_init(smem + 256, 128);
#endif
            // meta_full: 2 barriers, init_count=32
#if __CUDA_ARCH__ == 1030
#else
            mbarrier_init(smem + 264, 32);
            mbarrier_init(smem + 272, 32);
            // meta_empty: 2 barriers, init_count=384
            mbarrier_init(smem + 280, 384);
            mbarrier_init(smem + 288, 384);
            // step_full: 8 barriers, init_count=32
#endif
            mbarrier_init(smem + 296, 32);
            mbarrier_init(smem + 304, 32);
#if __CUDA_ARCH__ == 1030
            // meta_empty: 2 barriers, init_count=384
            mbarrier_init(smem + 312, 384);
            mbarrier_init(smem + 320, 384);
            // step_full: 8 barriers, init_count=32
#else
            mbarrier_init(smem + 312, 32);
            mbarrier_init(smem + 320, 32);
#endif
            mbarrier_init(smem + 328, 32);
            mbarrier_init(smem + 336, 32);
            mbarrier_init(smem + 344, 32);
            mbarrier_init(smem + 352, 32);
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 360, 32);
            mbarrier_init(smem + 368, 32);
            mbarrier_init(smem + 376, 32);
            mbarrier_init(smem + 384, 32);
#else
#endif
            // step_empty: 8 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
#else
            mbarrier_init(smem + 360, 128);
            mbarrier_init(smem + 368, 128);
            mbarrier_init(smem + 376, 128);
            mbarrier_init(smem + 384, 128);
#endif
            mbarrier_init(smem + 392, 128);
            mbarrier_init(smem + 400, 128);
            mbarrier_init(smem + 408, 128);
            mbarrier_init(smem + 416, 128);
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 424, 128);
            mbarrier_init(smem + 432, 128);
            mbarrier_init(smem + 440, 128);
            mbarrier_init(smem + 448, 128);
#else
#endif
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
#if __CUDA_ARCH__ == 1030
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 456);
#else
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 424);
#endif
    if (warp == 0) {
#if __CUDA_ARCH__ == 1030
        int _tmem_hold = smem + 456;
#else
        int _tmem_hold = smem + 424;
#endif
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_scores = taddr;
    const unsigned int tmem_probs_0 = taddr;
    const unsigned int tmem_probs_1 = taddr + 32;
    const unsigned int tmem_output_0 = taddr + 256;
    const unsigned int tmem_output_1 = taddr + 384;

    // Kernel post-init ops
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
            unsigned int p_sd_cursor = 0;
            int steps_before = 0;
            unsigned int _phase_step_full = 0;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_corr_done = 1;
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
                int first_pair = (group + steps_before) % 2;
                #pragma unroll 1
                for (int pair_index = first_pair; pair_index < pair_count; pair_index += 2) {
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
#if __CUDA_ARCH__ == 1030
                    float score_values[64];
                    float red_max_0 = -CUDART_INF_F;
                    float red_max_1 = -CUDART_INF_F;
                    float red_max_2 = -CUDART_INF_F;
                    float red_max_3 = -CUDART_INF_F;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
#else
                    float _tmem_load_0[64];
#endif
                    asm volatile(
#if __CUDA_ARCH__ == 1030
                        "tcgen05.ld.red.sync.aligned.16x32bx2.x16.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, [%17], 64;"
                        : "=f"(score_values[0]), "=f"(score_values[1]), "=f"(score_values[2]), "=f"(score_values[3]), "=f"(score_values[4]), "=f"(score_values[5]), "=f"(score_values[6]), "=f"(score_values[7]), "=f"(score_values[8]), "=f"(score_values[9]), "=f"(score_values[10]), "=f"(score_values[11]), "=f"(score_values[12]), "=f"(score_values[13]), "=f"(score_values[14]), "=f"(score_values[15]), "=f"(red_max_0)
#else
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
#endif
                        : "r"(score_addr));
#if __CUDA_ARCH__ == 1030
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
#else
#endif
                    asm volatile(
#if __CUDA_ARCH__ == 1030
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
#else
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[63]))
#endif
                        : "r"(score_addr + 32));
#if __CUDA_ARCH__ == 1030
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.16x32bx2.x16.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, [%17], 64;"
                        : "=f"(score_values[48]), "=f"(score_values[49]), "=f"(score_values[50]), "=f"(score_values[51]), "=f"(score_values[52]), "=f"(score_values[53]), "=f"(score_values[54]), "=f"(score_values[55]), "=f"(score_values[56]), "=f"(score_values[57]), "=f"(score_values[58]), "=f"(score_values[59]), "=f"(score_values[60]), "=f"(score_values[61]), "=f"(score_values[62]), "=f"(score_values[63]), "=f"(red_max_3)
                        : "r"(score_addr + 48));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
#else
#endif
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(s_empty_addr + (group) * 8);
#if __CUDA_ARCH__ == 1030
                    float chunk_max[4];
                    chunk_max[0] = red_max_0 * my_scales[0];
                    chunk_max[1] = red_max_1 * my_scales[1];
                    chunk_max[2] = red_max_2 * my_scales[2];
                    chunk_max[3] = red_max_3 * my_scales[3];
#else
#endif
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
#if __CUDA_ARCH__ == 1030
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
#else
                        if (!(_thresh_mask_0 & (1u << 0))) _tmem_load_0[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 1))) _tmem_load_0[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 2))) _tmem_load_0[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 3))) _tmem_load_0[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 4))) _tmem_load_0[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 5))) _tmem_load_0[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 6))) _tmem_load_0[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 7))) _tmem_load_0[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 8))) _tmem_load_0[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 9))) _tmem_load_0[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 10))) _tmem_load_0[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 11))) _tmem_load_0[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 12))) _tmem_load_0[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 13))) _tmem_load_0[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 14))) _tmem_load_0[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 15))) _tmem_load_0[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 16))) _tmem_load_0[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 17))) _tmem_load_0[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 18))) _tmem_load_0[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 19))) _tmem_load_0[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 20))) _tmem_load_0[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 21))) _tmem_load_0[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 22))) _tmem_load_0[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 23))) _tmem_load_0[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 24))) _tmem_load_0[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 25))) _tmem_load_0[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 26))) _tmem_load_0[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 27))) _tmem_load_0[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 28))) _tmem_load_0[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 29))) _tmem_load_0[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 30))) _tmem_load_0[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 31))) _tmem_load_0[31] = -CUDART_INF_F;
#endif
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
#if __CUDA_ARCH__ == 1030
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
#else
                        if (!(_thresh_mask_2 & (1u << 0))) _tmem_load_0[32] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 1))) _tmem_load_0[33] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 2))) _tmem_load_0[34] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 3))) _tmem_load_0[35] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 4))) _tmem_load_0[36] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 5))) _tmem_load_0[37] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 6))) _tmem_load_0[38] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 7))) _tmem_load_0[39] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 8))) _tmem_load_0[40] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 9))) _tmem_load_0[41] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 10))) _tmem_load_0[42] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 11))) _tmem_load_0[43] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 12))) _tmem_load_0[44] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 13))) _tmem_load_0[45] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 14))) _tmem_load_0[46] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 15))) _tmem_load_0[47] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 16))) _tmem_load_0[48] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 17))) _tmem_load_0[49] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 18))) _tmem_load_0[50] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 19))) _tmem_load_0[51] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 20))) _tmem_load_0[52] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 21))) _tmem_load_0[53] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 22))) _tmem_load_0[54] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 23))) _tmem_load_0[55] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 24))) _tmem_load_0[56] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 25))) _tmem_load_0[57] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 26))) _tmem_load_0[58] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 27))) _tmem_load_0[59] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 28))) _tmem_load_0[60] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 29))) _tmem_load_0[61] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 30))) _tmem_load_0[62] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 31))) _tmem_load_0[63] = -CUDART_INF_F;
                    }
                    float chunk_max[4];
                    #pragma unroll
                    for (int scale_group_1 = 0; scale_group_1 < 4; scale_group_1++) {
                        float _max3_5;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_5) : "f"(_tmem_load_0[scale_group_1 * 16]), "f"(_tmem_load_0[scale_group_1 * 16 + 1]), "f"(_tmem_load_0[scale_group_1 * 16 + 2]));
                        float _max3_6;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_6) : "f"(_tmem_load_0[scale_group_1 * 16 + 3]), "f"(_tmem_load_0[scale_group_1 * 16 + 4]), "f"(_tmem_load_0[scale_group_1 * 16 + 5]));
                        float _max3_7;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_7) : "f"(_tmem_load_0[scale_group_1 * 16 + 6]), "f"(_tmem_load_0[scale_group_1 * 16 + 7]), "f"(_tmem_load_0[scale_group_1 * 16 + 8]));
                        float _max3_4;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_4) : "f"(_max3_5), "f"(_max3_6), "f"(_max3_7));
                        float _max3_9;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_9) : "f"(_tmem_load_0[scale_group_1 * 16 + 9]), "f"(_tmem_load_0[scale_group_1 * 16 + 10]), "f"(_tmem_load_0[scale_group_1 * 16 + 11]));
                        float _max3_10;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_10) : "f"(_tmem_load_0[scale_group_1 * 16 + 12]), "f"(_tmem_load_0[scale_group_1 * 16 + 13]), "f"(_tmem_load_0[scale_group_1 * 16 + 14]));
                        float _max3_8;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_8) : "f"(_max3_9), "f"(_max3_10), "f"(_tmem_load_0[scale_group_1 * 16 + 15]));
                        chunk_max[scale_group_1] = max_noftz(_max3_4, _max3_8) * my_scales[scale_group_1];
#endif
                    }
                    float _max3_11;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_11) : "f"(chunk_max[0]), "f"(chunk_max[1]), "f"(chunk_max[2]));
                    float tile_max;
                    asm("max.f32 %0, %1, %2;" : "=f"(tile_max) : "f"(_max3_11), "f"(chunk_max[3]));
                    tile_max = ((m_valid > 0) ? tile_max : -CUDART_INF_F);
                    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, tile_max, 16);
                    tile_max = max_noftz(tile_max, _shfl_xor_sync_12);
                    float new_max;
                    asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(tile_max), "f"(row_max));
                    float acc_scale = 1.0f;
                    float old_max = row_max;
                    float grow = new_max - old_max;
                    row_max = ((grow > 0.8073549f) ? new_max : old_max);
                    float _exp2_13 = approx_exp2(old_max - new_max);
                    acc_scale = ((grow > 0.8073549f && old_max > -CUDART_INF_F) ? _exp2_13 : 1.0f);
                    mbarrier_wait(corr_done_addr + (group) * 8, _phase_corr_done);
                    scale_smem[slot] = acc_scale;
#if __CUDA_ARCH__ == 1030
#else
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(corr_sig_addr + (group) * 8);
                    float score_bias = ((row_max > -CUDART_INF_F) ? 8.0f - row_max : 0.0f);
                    #pragma unroll
                    for (int scale_group_2 = 0; scale_group_2 < 4; scale_group_2++) {
                        const float2 _fma_b2_14 = {my_scales[scale_group_2], my_scales[scale_group_2]};
                        const float2 _fma_c2_15 = {score_bias, score_bias};
                        #pragma unroll
                        for (int _lf = 0; _lf < 8; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((score_values + scale_group_2 * 16))[_lf], _fma_b2_14, _fma_c2_15);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_0 + scale_group_2 * 16))[_lf], _fma_b2_14, _fma_c2_15);
#endif
                    }
                    unsigned int p_addr = taddr + (unsigned int)((int)((unsigned int)group + p_sd_cursor) * 32 + 1048576) + (unsigned int)(tmem_row_origin << 16);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
#if __CUDA_ARCH__ == 1030
                        score_values[_le] = approx_exp2(score_values[_le]);
#else
                        _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
#endif
                    }
                    float2 _reg_reduce_sum2_16 = make_float2(0.0f, 0.0f);
#if __CUDA_ARCH__ == 1030
                    softmax_block_sum(&score_values[0], &_reg_reduce_sum2_16);
                    softmax_block_sum(&score_values[32], &_reg_reduce_sum2_16);
#else
                    softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_16);
                    softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_16);
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[0]), "f"(score_values[1]),
                                               "f"(score_values[2]), "f"(score_values[3]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[0]), "f"(_tmem_load_0[1]),
                                               "f"(_tmem_load_0[2]), "f"(_tmem_load_0[3]));
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[4]), "f"(score_values[5]),
                                               "f"(score_values[6]), "f"(score_values[7]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[4]), "f"(_tmem_load_0[5]),
                                               "f"(_tmem_load_0[6]), "f"(_tmem_load_0[7]));
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[8]), "f"(score_values[9]),
                                               "f"(score_values[10]), "f"(score_values[11]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[8]), "f"(_tmem_load_0[9]),
                                               "f"(_tmem_load_0[10]), "f"(_tmem_load_0[11]));
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[12]), "f"(score_values[13]),
                                               "f"(score_values[14]), "f"(score_values[15]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[12]), "f"(_tmem_load_0[13]),
                                               "f"(_tmem_load_0[14]), "f"(_tmem_load_0[15]));
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[16]), "f"(score_values[17]),
                                               "f"(score_values[18]), "f"(score_values[19]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[16]), "f"(_tmem_load_0[17]),
                                               "f"(_tmem_load_0[18]), "f"(_tmem_load_0[19]));
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[20]), "f"(score_values[21]),
                                               "f"(score_values[22]), "f"(score_values[23]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[20]), "f"(_tmem_load_0[21]),
                                               "f"(_tmem_load_0[22]), "f"(_tmem_load_0[23]));
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[24]), "f"(score_values[25]),
                                               "f"(score_values[26]), "f"(score_values[27]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[24]), "f"(_tmem_load_0[25]),
                                               "f"(_tmem_load_0[26]), "f"(_tmem_load_0[27]));
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[28]), "f"(score_values[29]),
                                               "f"(score_values[30]), "f"(score_values[31]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[28]), "f"(_tmem_load_0[29]),
                                               "f"(_tmem_load_0[30]), "f"(_tmem_load_0[31]));
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[32]), "f"(score_values[33]),
                                               "f"(score_values[34]), "f"(score_values[35]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[32]), "f"(_tmem_load_0[33]),
                                               "f"(_tmem_load_0[34]), "f"(_tmem_load_0[35]));
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[36]), "f"(score_values[37]),
                                               "f"(score_values[38]), "f"(score_values[39]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[36]), "f"(_tmem_load_0[37]),
                                               "f"(_tmem_load_0[38]), "f"(_tmem_load_0[39]));
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[40]), "f"(score_values[41]),
                                               "f"(score_values[42]), "f"(score_values[43]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[40]), "f"(_tmem_load_0[41]),
                                               "f"(_tmem_load_0[42]), "f"(_tmem_load_0[43]));
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[44]), "f"(score_values[45]),
                                               "f"(score_values[46]), "f"(score_values[47]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[44]), "f"(_tmem_load_0[45]),
                                               "f"(_tmem_load_0[46]), "f"(_tmem_load_0[47]));
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[48]), "f"(score_values[49]),
                                               "f"(score_values[50]), "f"(score_values[51]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[48]), "f"(_tmem_load_0[49]),
                                               "f"(_tmem_load_0[50]), "f"(_tmem_load_0[51]));
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[52]), "f"(score_values[53]),
                                               "f"(score_values[54]), "f"(score_values[55]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[52]), "f"(_tmem_load_0[53]),
                                               "f"(_tmem_load_0[54]), "f"(_tmem_load_0[55]));
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[56]), "f"(score_values[57]),
                                               "f"(score_values[58]), "f"(score_values[59]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[56]), "f"(_tmem_load_0[57]),
                                               "f"(_tmem_load_0[58]), "f"(_tmem_load_0[59]));
#endif
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
#if __CUDA_ARCH__ == 1030
                            : "=r"(_packed) : "f"(score_values[60]), "f"(score_values[61]),
                                               "f"(score_values[62]), "f"(score_values[63]));
#else
                            : "=r"(_packed) : "f"(_tmem_load_0[60]), "f"(_tmem_load_0[61]),
                                               "f"(_tmem_load_0[62]), "f"(_tmem_load_0[63]));
#endif
                        _fp8_0[15] = _packed;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 16, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + ((unsigned int)group + p_sd_cursor) * 8);
                    p_sd_cursor += 2;
                    if (p_sd_cursor >= 4) { p_sd_cursor -= 4; }
                    _phase_corr_done ^= 1;
                    row_sum = __fmaf_rn(row_sum, acc_scale, step_sum);
                }
                mbarrier_wait(corr_done_addr + (group) * 8, _phase_corr_done);
                scale_smem[256 + slot] = row_sum;
                scale_smem[512 + slot] = row_max;
#if __CUDA_ARCH__ == 1030
#else
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
                mbarrier_arrive(corr_sig_addr + (group) * 8);
                _phase_corr_done ^= 1;
                steps_before = steps_before + pair_count;
            }
#if __CUDA_ARCH__ == 1030
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            unsigned int kv_stage = 0;
            unsigned int q_stage = 0;
            int count_1 = (total_tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
            int my_tiles_1 = count_1;
            unsigned int s_cur = 0;
            unsigned int p_cur = 0;
            int q_iter = 0;
            int q_step = 0;
            int q_pairs = 0;
            unsigned int sd_e_cur = 0;
            unsigned int pv_cur = 0;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_kv_full = 0;
            if (q_iter < my_tiles_1) {
                if (q_step == 0) {
                    int split_d = blockIdx.x + q_iter * gridDim.x - split_from;
                    if (split_d < 0) {
                        split_d = 0;
                    }
                    int tile_id = blockIdx.x + q_iter * gridDim.x - (split_d + 1) / 2;
                    int q_block = tile_id % q_blocks;
                    int bh = tile_id / q_blocks;
                    int head = bh % num_heads;
                    int batch_1 = bh / num_heads;
                    int kv_head_1 = head / (num_heads / num_kv_heads);
                    int query_base = q_block * 64;
                    int q_valid_1 = seqlen_q - query_base;
                    if (q_valid_1 > 64) {
                        q_valid_1 = 64;
                    }
                    if (q_valid_1 < 0) {
                        q_valid_1 = 0;
                    }
                    int row_slot = bh * q_blocks + q_block;
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
                    int pair_count_1 = (selected_count_1 + 1) / 2;
                    if (pair_count_1 < 1) {
                        pair_count_1 = 1;
                    }
                    int index_base_1 = row_slot * capacity;
                    if (blockIdx.x + q_iter * gridDim.x >= split_from) {
                        int half_pairs = (pair_count_1 + 1) / 2;
                        if ((blockIdx.x + q_iter * gridDim.x - split_from) % 2 == 0) {
                            pair_count_1 = half_pairs;
                        } else {
                            pair_count_1 = pair_count_1 - half_pairs;
                            if (pair_count_1 < 1) {
                                pair_count_1 = 1;
                            }
                            index_base_1 = index_base_1 + half_pairs * 2;
                            selected_count_1 = selected_count_1 - half_pairs * 2;
                            if (selected_count_1 < 0) {
                                selected_count_1 = 0;
                            }
                        }
                    }
                    q_pairs = pair_count_1;
                    mbarrier_wait(q_full_addr + (q_stage) * 8, _phase_q_full);
                }
                unsigned int k_stage = kv_stage;
                mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage) * 512);
                int mma_b_lo = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                    }
                }
                elect_commit(s_full_addr + (s_cur) * 8);
                s_cur += 1;
                if (s_cur == 2) { s_cur = 0; }
                elect_commit(kv_empty_addr + (kv_stage) * 8);
                kv_stage += 1;
                if (kv_stage == 8) { kv_stage = 0; _phase_kv_full ^= 1; }
                q_step = q_step + 1;
                if (q_step == q_pairs) {
                    if (my_tiles_1 > q_iter + 1) {
                        elect_commit(q_empty_addr + (q_stage) * 8);
                    }
                    q_stage += 1;
                    if (q_stage == 2) { q_stage = 0; _phase_q_full ^= 1; }
                    q_step = 0;
                    q_iter = q_iter + 1;
                }
            }
            if (q_iter < my_tiles_1) {
                if (q_step == 0) {
                    int split_d_1 = blockIdx.x + q_iter * gridDim.x - split_from;
                    if (split_d_1 < 0) {
                        split_d_1 = 0;
                    }
                    int tile_id_1 = blockIdx.x + q_iter * gridDim.x - (split_d_1 + 1) / 2;
                    int q_block_1 = tile_id_1 % q_blocks;
                    int bh_1 = tile_id_1 / q_blocks;
                    int head_1 = bh_1 % num_heads;
                    int batch_2 = bh_1 / num_heads;
                    int kv_head_2 = head_1 / (num_heads / num_kv_heads);
                    int query_base_1 = q_block_1 * 64;
                    int q_valid_2 = seqlen_q - query_base_1;
                    if (q_valid_2 > 64) {
                        q_valid_2 = 64;
                    }
                    if (q_valid_2 < 0) {
                        q_valid_2 = 0;
                    }
                    int row_slot_1 = bh_1 * q_blocks + q_block_1;
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
                    int pair_count_2 = (selected_count_2 + 1) / 2;
                    if (pair_count_2 < 1) {
                        pair_count_2 = 1;
                    }
                    int index_base_2 = row_slot_1 * capacity;
                    if (blockIdx.x + q_iter * gridDim.x >= split_from) {
                        int half_pairs_1 = (pair_count_2 + 1) / 2;
                        if ((blockIdx.x + q_iter * gridDim.x - split_from) % 2 == 0) {
                            pair_count_2 = half_pairs_1;
                        } else {
                            pair_count_2 = pair_count_2 - half_pairs_1;
                            if (pair_count_2 < 1) {
                                pair_count_2 = 1;
                            }
                            index_base_2 = index_base_2 + half_pairs_1 * 2;
                            selected_count_2 = selected_count_2 - half_pairs_1 * 2;
                            if (selected_count_2 < 0) {
                                selected_count_2 = 0;
                            }
                        }
                    }
                    q_pairs = pair_count_2;
                    mbarrier_wait(q_full_addr + (q_stage) * 8, _phase_q_full);
                }
                unsigned int k_stage_1 = kv_stage;
                mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo_1 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage) * 512);
                int mma_b_lo_1 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage_1) * 1024);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                    }
                }
                elect_commit(s_full_addr + (s_cur) * 8);
                s_cur += 1;
                if (s_cur == 2) { s_cur = 0; }
                elect_commit(kv_empty_addr + (kv_stage) * 8);
                kv_stage += 1;
                if (kv_stage == 8) { kv_stage = 0; _phase_kv_full ^= 1; }
                q_step = q_step + 1;
                if (q_step == q_pairs) {
                    if (my_tiles_1 > q_iter + 1) {
                        elect_commit(q_empty_addr + (q_stage) * 8);
                    }
                    q_stage += 1;
                    if (q_stage == 2) { q_stage = 0; _phase_q_full ^= 1; }
                    q_step = 0;
                    q_iter = q_iter + 1;
                }
            }
            unsigned int _phase_s_empty = 0;
            unsigned int _phase_p_full = 0;
            unsigned int _phase_tile_done_0 = 0;
            #pragma unroll 1
            for (int tile_iter_1 = 0; tile_iter_1 < my_tiles_1; tile_iter_1++) {
                int tile_idx = blockIdx.x + tile_iter_1 * gridDim.x;
                int split_d_2 = tile_idx - split_from;
                if (split_d_2 < 0) {
                    split_d_2 = 0;
                }
                int tile_id_2 = tile_idx - (split_d_2 + 1) / 2;
                int q_block_2 = tile_id_2 % q_blocks;
                int bh_2 = tile_id_2 / q_blocks;
                int head_2 = bh_2 % num_heads;
                int batch_3 = bh_2 / num_heads;
                int kv_head_3 = head_2 / (num_heads / num_kv_heads);
                int query_base_2 = q_block_2 * 64;
                int q_valid_3 = seqlen_q - query_base_2;
                if (q_valid_3 > 64) {
                    q_valid_3 = 64;
                }
                if (q_valid_3 < 0) {
                    q_valid_3 = 0;
                }
                int row_slot_2 = bh_2 * q_blocks + q_block_2;
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
                int pair_count_3 = (selected_count_3 + 1) / 2;
                if (pair_count_3 < 1) {
                    pair_count_3 = 1;
                }
                int index_base_3 = row_slot_2 * capacity;
                if (tile_idx >= split_from) {
                    int half_pairs_2 = (pair_count_3 + 1) / 2;
                    if ((tile_idx - split_from) % 2 == 0) {
                        pair_count_3 = half_pairs_2;
                    } else {
                        pair_count_3 = pair_count_3 - half_pairs_2;
                        if (pair_count_3 < 1) {
                            pair_count_3 = 1;
                        }
                        index_base_3 = index_base_3 + half_pairs_2 * 2;
                        selected_count_3 = selected_count_3 - half_pairs_2 * 2;
                        if (selected_count_3 < 0) {
                            selected_count_3 = 0;
                        }
                    }
                }
                #pragma unroll 1
                for (int step = 0; step < pair_count_3; step++) {
                    if (q_iter < my_tiles_1) {
                        if (q_step == 0) {
                            int split_d_1_1 = blockIdx.x + q_iter * gridDim.x - split_from;
                            if (split_d_1_1 < 0) {
                                split_d_1_1 = 0;
                            }
                            int tile_id_1_1 = blockIdx.x + q_iter * gridDim.x - (split_d_1_1 + 1) / 2;
                            int q_block_1_1 = tile_id_1_1 % q_blocks;
                            int bh_1_1 = tile_id_1_1 / q_blocks;
                            int head_1_1 = bh_1_1 % num_heads;
                            int batch_1_1 = bh_1_1 / num_heads;
                            int kv_head_1_1 = head_1_1 / (num_heads / num_kv_heads);
                            int query_base_1_1 = q_block_1_1 * 64;
                            int q_valid_1_1 = seqlen_q - query_base_1_1;
                            if (q_valid_1_1 > 64) {
                                q_valid_1_1 = 64;
                            }
                            if (q_valid_1_1 < 0) {
                                q_valid_1_1 = 0;
                            }
                            int row_slot_1_1 = bh_1_1 * q_blocks + q_block_1_1;
                            int selected_count_1_1 = block_sparse_num;
                            if (has_block_nums != 0) {
                                selected_count_1_1 = block_nums[row_slot_1_1];
                            }
                            if (selected_count_1_1 < 0) {
                                selected_count_1_1 = 0;
                            }
                            if (selected_count_1_1 > capacity) {
                                selected_count_1_1 = capacity;
                            }
                            int pair_count_1_1 = (selected_count_1_1 + 1) / 2;
                            if (pair_count_1_1 < 1) {
                                pair_count_1_1 = 1;
                            }
                            int index_base_1_1 = row_slot_1_1 * capacity;
                            if (blockIdx.x + q_iter * gridDim.x >= split_from) {
                                int half_pairs_3 = (pair_count_1_1 + 1) / 2;
                                if ((blockIdx.x + q_iter * gridDim.x - split_from) % 2 == 0) {
                                    pair_count_1_1 = half_pairs_3;
                                } else {
                                    pair_count_1_1 = pair_count_1_1 - half_pairs_3;
                                    if (pair_count_1_1 < 1) {
                                        pair_count_1_1 = 1;
                                    }
                                    index_base_1_1 = index_base_1_1 + half_pairs_3 * 2;
                                    selected_count_1_1 = selected_count_1_1 - half_pairs_3 * 2;
                                    if (selected_count_1_1 < 0) {
                                        selected_count_1_1 = 0;
                                    }
                                }
                            }
                            q_pairs = pair_count_1_1;
                            mbarrier_wait(q_full_addr + (q_stage) * 8, _phase_q_full);
                        }
                        unsigned int k_stage_2 = kv_stage;
                        mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                        mbarrier_wait(s_empty_addr + (sd_e_cur) * 8, _phase_s_empty);
                        sd_e_cur += 1;
                        if (sd_e_cur == 2) { sd_e_cur = 0; _phase_s_empty ^= 1; }
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int mma_a_lo_2 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage) * 512);
                        int mma_b_lo_2 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage_2) * 1024);
                        {
                            uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_2);
                            uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_2);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                            }
                        }
                        elect_commit(s_full_addr + (s_cur) * 8);
                        s_cur += 1;
                        if (s_cur == 2) { s_cur = 0; }
                        elect_commit(kv_empty_addr + (kv_stage) * 8);
                        kv_stage += 1;
                        if (kv_stage == 8) { kv_stage = 0; _phase_kv_full ^= 1; }
                        q_step = q_step + 1;
                        if (q_step == q_pairs) {
                            if (my_tiles_1 > q_iter + 1) {
                                elect_commit(q_empty_addr + (q_stage) * 8);
                            }
                            q_stage += 1;
                            if (q_stage == 2) { q_stage = 0; _phase_q_full ^= 1; }
                            q_step = 0;
                            q_iter = q_iter + 1;
                        }
                    }
                    int first_pv = ((step < 2) ? 1 : 0);
                    unsigned int v_stage = kv_stage;
                    mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                    mbarrier_wait(p_full_addr + (p_cur) * 8, _phase_p_full);
                    unsigned int group_cols = p_cur % 2 * 128 + 1048576;
                    unsigned int p_cols = p_cur * 32 + 1048576;
                    p_cur += 1;
                    if (p_cur == 4) { p_cur = 0; _phase_p_full ^= 1; }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_b_lo_3 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024);
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
                    :: "r"((tmem_output_0 + ((int)group_cols))), "r"(mma_b_lo_3), "r"(tmem_probs_0 + p_cols), "r"(((first_pv) ? 0 : 1)) : "memory");
                    elect_commit(kv_empty_addr + (kv_stage) * 8);
                    kv_stage += 1;
                    if (kv_stage == 8) { kv_stage = 0; _phase_kv_full ^= 1; }
                    elect_commit(pv_done_addr + (pv_cur) * 8);
                    pv_cur += 1;
                    if (pv_cur == 2) { pv_cur = 0; }
                    if (step == pair_count_3 - 1) {
                        elect_commit(o_full_addr);
                    }
                }
                mbarrier_wait(tile_done_addr, _phase_tile_done_0);
                _phase_tile_done_0 ^= 1;
            }
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
#else
#endif
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 104;");
        { // correction_main
            const int warp_in_role = warp - 8;
            const int tmem_row_origin_1 = warp_in_role * 32;
            int my_row_1 = warp_in_role * 16 + lane % 16;
            int col_half_1 = lane / 16;
            int row_addr = tmem_row_origin_1 + 16 << 16;
            int slot_0 = col_half_1 * 64 + my_row_1;
            int slot_1 = 128 + col_half_1 * 64 + my_row_1;
#if __CUDA_ARCH__ == 1030
            int count_2 = (total_tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
            int my_tiles_2 = count_2;
#else
            int count_1 = (total_tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
            int my_tiles_1 = count_1;
#endif
            unsigned int corr_p_full_cursor0 = 0;
            unsigned int corr_p_full_cursor1 = 0;
            int corr_tiles_done = 0;
            int steps_before_1 = 0;
#if __CUDA_ARCH__ == 1030
            if (my_tiles_2 > 0) {
#else
            if (my_tiles_1 > 0) {
#endif
                mbarrier_arrive(p_full_addr + (0 + corr_p_full_cursor0) * 8);
                corr_p_full_cursor0 += 2;
                if (corr_p_full_cursor0 >= 4) { corr_p_full_cursor0 -= 4; }
                mbarrier_arrive(p_full_addr + (1 + corr_p_full_cursor1) * 8);
                corr_p_full_cursor1 += 2;
                if (corr_p_full_cursor1 >= 4) { corr_p_full_cursor1 -= 4; }
            }
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_corr_sig_1 = 0;
            unsigned int _phase_pv_done_0 = 1;
            unsigned int _phase_pv_done_1 = 1;
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
#if __CUDA_ARCH__ == 1030
            for (int tile_iter_2 = 0; tile_iter_2 < my_tiles_2; tile_iter_2++) {
                int tile_idx_1 = blockIdx.x + tile_iter_2 * gridDim.x;
                int cm_off = tile_iter_2 % 2 * 100;
#else
            for (int tile_iter_1 = 0; tile_iter_1 < my_tiles_1; tile_iter_1++) {
                int tile_idx = blockIdx.x + tile_iter_1 * gridDim.x;
                int cm_off = tile_iter_1 % 2 * 100;
#endif
                int _stage_meta_full_1 = (int)((unsigned int)corr_tiles_done % 2);
                int _parity_meta_full_1 = (int)((unsigned int)corr_tiles_done / 2 % 2);
                mbarrier_wait(meta_full_addr + (_stage_meta_full_1) * 8, _parity_meta_full_1);
#if __CUDA_ARCH__ == 1030
                int batch_4 = meta_i[cm_off + 88];
                int kv_head_4 = meta_i[cm_off + 88 + 1];
                int q_valid_4 = meta_i[cm_off + 88 + 3];
                int pair_count_4 = meta_i[cm_off + 88 + 5];
                int head_3 = meta_i[cm_off + 88 + 7];
                int bh_3 = meta_i[cm_off + 88 + 8];
                int query_base_3 = meta_i[cm_off + 88 + 9];
#else
                int batch_1 = meta_i[cm_off + 88];
                int kv_head_1 = meta_i[cm_off + 88 + 1];
                int q_valid_1 = meta_i[cm_off + 88 + 3];
                int pair_count_1 = meta_i[cm_off + 88 + 5];
                int head = meta_i[cm_off + 88 + 7];
                int bh = meta_i[cm_off + 88 + 8];
                int query_base = meta_i[cm_off + 88 + 9];
#endif
                int first_group = steps_before_1 % 2;
#if __CUDA_ARCH__ == 1030
                int last_group = (steps_before_1 + pair_count_4 - 1) % 2;
#else
                int last_group = (steps_before_1 + pair_count_1 - 1) % 2;
#endif
                float max_g0 = 0.0f;
                float sum_g0 = 0.0f;
                float max_g1 = 0.0f;
                float sum_g1 = 0.0f;
#if __CUDA_ARCH__ == 1030
                if (pair_count_4 == 1) {
#else
                if (pair_count_1 == 1) {
#endif
                    if (1 - last_group == 0) {
                        mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                        _phase_corr_sig_0 ^= 1;
                        max_g0 = scale_smem[512 + slot_0];
                        sum_g0 = scale_smem[256 + slot_0];
                        mbarrier_arrive(corr_done_addr);
                    } else {
                        mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                        _phase_corr_sig_1 ^= 1;
                        max_g1 = scale_smem[512 + slot_1];
                        sum_g1 = scale_smem[256 + slot_1];
                        mbarrier_arrive(corr_done_addr + 8);
                    }
                }
                #pragma unroll 1
#if __CUDA_ARCH__ == 1030
                for (int step_1 = 0; step_1 < pair_count_4; step_1++) {
                    if ((steps_before_1 + step_1) % 2 == 0) {
#else
                for (int step = 0; step < pair_count_1; step++) {
                    if ((steps_before_1 + step) % 2 == 0) {
#endif
                        mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                        _phase_corr_sig_0 ^= 1;
#if __CUDA_ARCH__ == 1030
                        if (step_1 >= 2) {
#else
                        if (step >= 2) {
#endif
                            float acc_scale_c = scale_smem[slot_0];
                            mbarrier_arrive(corr_done_addr);
                            mbarrier_wait(pv_done_addr, _phase_pv_done_0);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _warp_vote_0 = __any_sync(0xFFFFFFFF, acc_scale_c < 1.0f);
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
                                const float2 _scale2_1 = {acc_scale_c, acc_scale_c};
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
                            mbarrier_arrive(p_full_addr + (0 + corr_p_full_cursor0) * 8);
                            corr_p_full_cursor0 += 2;
                            if (corr_p_full_cursor0 >= 4) { corr_p_full_cursor0 -= 4; }
                        } else {
                            mbarrier_arrive(corr_done_addr);
                        }
                        _phase_pv_done_0 ^= 1;
                    } else {
                        mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                        _phase_corr_sig_1 ^= 1;
#if __CUDA_ARCH__ == 1030
                        if (step_1 >= 2) {
#else
                        if (step >= 2) {
#endif
                            float acc_scale_c_1 = scale_smem[slot_1];
                            mbarrier_arrive(corr_done_addr + 8);
                            mbarrier_wait(pv_done_addr + 8, _phase_pv_done_1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _warp_vote_2 = __any_sync(0xFFFFFFFF, acc_scale_c_1 < 1.0f);
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
                                const float2 _scale2_3 = {acc_scale_c_1, acc_scale_c_1};
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
                            mbarrier_arrive(p_full_addr + (1 + corr_p_full_cursor1) * 8);
                            corr_p_full_cursor1 += 2;
                            if (corr_p_full_cursor1 >= 4) { corr_p_full_cursor1 -= 4; }
                        } else {
                            mbarrier_arrive(corr_done_addr + 8);
                        }
                        _phase_pv_done_1 ^= 1;
                    }
#if __CUDA_ARCH__ == 1030
                    if (step_1 == pair_count_4 - 2) {
#else
                    if (step == pair_count_1 - 2) {
#endif
                        if (1 - last_group == 0) {
                            mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                            _phase_corr_sig_0 ^= 1;
                            max_g0 = scale_smem[512 + slot_0];
                            sum_g0 = scale_smem[256 + slot_0];
                            mbarrier_arrive(corr_done_addr);
                            {
                                mbarrier_arrive(p_full_addr + (0 + corr_p_full_cursor0) * 8);
                                corr_p_full_cursor0 += 2;
                                if (corr_p_full_cursor0 >= 4) { corr_p_full_cursor0 -= 4; }
                            }
                        } else {
                            mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                            _phase_corr_sig_1 ^= 1;
                            max_g1 = scale_smem[512 + slot_1];
                            sum_g1 = scale_smem[256 + slot_1];
                            mbarrier_arrive(corr_done_addr + 8);
                            {
                                mbarrier_arrive(p_full_addr + (1 + corr_p_full_cursor1) * 8);
                                corr_p_full_cursor1 += 2;
                                if (corr_p_full_cursor1 >= 4) { corr_p_full_cursor1 -= 4; }
                            }
                        }
                    }
                }
                if (last_group == 0) {
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    max_g0 = scale_smem[512 + slot_0];
                    sum_g0 = scale_smem[256 + slot_0];
                    mbarrier_arrive(corr_done_addr);
                    {
                        mbarrier_arrive(p_full_addr + (0 + corr_p_full_cursor0) * 8);
                        corr_p_full_cursor0 += 2;
                        if (corr_p_full_cursor0 >= 4) { corr_p_full_cursor0 -= 4; }
                    }
                } else {
                    mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                    _phase_corr_sig_1 ^= 1;
                    max_g1 = scale_smem[512 + slot_1];
                    sum_g1 = scale_smem[256 + slot_1];
                    mbarrier_arrive(corr_done_addr + 8);
                    {
                        mbarrier_arrive(p_full_addr + (1 + corr_p_full_cursor1) * 8);
                        corr_p_full_cursor1 += 2;
                        if (corr_p_full_cursor1 >= 4) { corr_p_full_cursor1 -= 4; }
                    }
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
#if __CUDA_ARCH__ == 1030
                if (pair_count_4 >= 2 || last_group == 0) {
#else
                if (pair_count_1 >= 2 || last_group == 0) {
#endif
                    mbarrier_wait(pv_done_addr, _phase_pv_done_0);
                }
#if __CUDA_ARCH__ == 1030
                if (pair_count_4 >= 2 || last_group != 0) {
#else
                if (pair_count_1 >= 2 || last_group != 0) {
#endif
                    mbarrier_wait(pv_done_addr + 8, _phase_pv_done_1);
                }
                asm volatile("tcgen05.fence::after_thread_sync;");
                float max_0 = ((first_group != 0) ? max_g1 : max_g0);
                float sum_0 = ((first_group != 0) ? sum_g1 : sum_g0);
                float max_1 = ((first_group != 0) ? max_g0 : max_g1);
                float sum_1 = ((first_group != 0) ? sum_g0 : sum_g1);
                int o0_off = 256 + first_group * 128;
                int o1_off = 384 - first_group * 128;
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
#if __CUDA_ARCH__ == 1030
                int v_scale_base = batch_4 * v_scale_batch_stride + kv_head_4 * 128 + col_base;
                int query = query_base_3 + my_row_1;
                int output_row = ((batch_4 * seqlen_q + query) * num_heads + head_3) * 128;
#else
                int v_scale_base = batch_1 * v_scale_batch_stride + kv_head_1 * 128 + col_base;
                int query = query_base + my_row_1;
                int output_row = ((batch_1 * seqlen_q + query) * num_heads + head) * 128;
#endif
                unsigned int vs_smem = 0;
#if __CUDA_ARCH__ == 1030
                vs_smem = vs_f_addr + (unsigned int)((tile_iter_2 % 2 * 128 + col_base) * 4);
#else
                vs_smem = vs_f_addr + (unsigned int)((tile_iter_1 % 2 * 128 + col_base) * 4);
#endif
                int piece = meta_i[cm_off + 88 + 10];
                if (piece >= 0) {
                    int x_thread = warp_in_role * 32 + lane;
                    int rank = piece % 2;
                    float own_max = ((has_mass != 0) ? row_max_1 : -CUDART_INF_F);
                    float own_sum = ((has_mass != 0) ? total_sum : 0.0f);
                    if (x_thread == 0) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 _, [%0], %1;"
                            :: "r"(xfer_addr), "r"((uint32_t)(18432)) : "memory");
                    }
                    uint32_t _cluster_mapa_7;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_7) : "r"(xbuf_addr + (unsigned int)((16 * 128 + x_thread) * 16)), "r"(1 - rank));
                    uint32_t _mbar_peer_8;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_8) : "r"(xfer_addr), "r"(1 - rank));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_7), "r"(__float_as_uint(own_max)), "r"(__float_as_uint(own_sum)), "r"(__float_as_uint(0.0f)), "r"(__float_as_uint(0.0f)), "r"(_mbar_peer_8) : "memory");
                    float part_chunk[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[1])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[2])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[3])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[4])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[5])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[6])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[7])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[8])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[9])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[10])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[11])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[12])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[13])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[14])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[15])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[16])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[17])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[18])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[19])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[20])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[21])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[22])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[23])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[24])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[25])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[26])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[27])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[28])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[29])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[30])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk[31]))
                        : "r"(taddr + (unsigned int)o0_off + (unsigned int)row_addr + (unsigned int)((1 - rank) * 32)));
                    const float2 _scale2_9 = {merge_0, merge_0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(part_chunk)[_ls], _scale2_9);
#if __CUDA_ARCH__ == 1030
                    if (pair_count_4 > 1) {
#else
                    if (pair_count_1 > 1) {
#endif
                        float part_chunk_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1[31]))
                            : "r"(taddr + (unsigned int)o1_off + (unsigned int)row_addr + (unsigned int)((1 - rank) * 32)));
                        #pragma unroll
                        for (int item = 0; item < 32; item++) {
                            part_chunk[item] = __fmaf_rn(part_chunk_1[item], merge_1, part_chunk[item]);
                        }
                    }
                    if (has_mass != 0) {
                        #pragma unroll
                        for (int quad = 0; quad < 8; quad++) {
                            float part_vs[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_vs[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_vs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_vs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_vs[(0) + 3]))
                                : "r"(vs_smem + (unsigned int)(((1 - rank) * 32 + quad * 4) * 4)));
                            #pragma unroll
                            for (int item_1 = 0; item_1 < 4; item_1++) {
                                part_chunk[quad * 4 + item_1] = part_chunk[quad * 4 + item_1] * part_vs[item_1];
                            }
                        }
                    } else {
                        if (!(0 & (1u << 0))) part_chunk[0] = 0.0f;
                        if (!(0 & (1u << 1))) part_chunk[1] = 0.0f;
                        if (!(0 & (1u << 2))) part_chunk[2] = 0.0f;
                        if (!(0 & (1u << 3))) part_chunk[3] = 0.0f;
                        if (!(0 & (1u << 4))) part_chunk[4] = 0.0f;
                        if (!(0 & (1u << 5))) part_chunk[5] = 0.0f;
                        if (!(0 & (1u << 6))) part_chunk[6] = 0.0f;
                        if (!(0 & (1u << 7))) part_chunk[7] = 0.0f;
                        if (!(0 & (1u << 8))) part_chunk[8] = 0.0f;
                        if (!(0 & (1u << 9))) part_chunk[9] = 0.0f;
                        if (!(0 & (1u << 10))) part_chunk[10] = 0.0f;
                        if (!(0 & (1u << 11))) part_chunk[11] = 0.0f;
                        if (!(0 & (1u << 12))) part_chunk[12] = 0.0f;
                        if (!(0 & (1u << 13))) part_chunk[13] = 0.0f;
                        if (!(0 & (1u << 14))) part_chunk[14] = 0.0f;
                        if (!(0 & (1u << 15))) part_chunk[15] = 0.0f;
                        if (!(0 & (1u << 16))) part_chunk[16] = 0.0f;
                        if (!(0 & (1u << 17))) part_chunk[17] = 0.0f;
                        if (!(0 & (1u << 18))) part_chunk[18] = 0.0f;
                        if (!(0 & (1u << 19))) part_chunk[19] = 0.0f;
                        if (!(0 & (1u << 20))) part_chunk[20] = 0.0f;
                        if (!(0 & (1u << 21))) part_chunk[21] = 0.0f;
                        if (!(0 & (1u << 22))) part_chunk[22] = 0.0f;
                        if (!(0 & (1u << 23))) part_chunk[23] = 0.0f;
                        if (!(0 & (1u << 24))) part_chunk[24] = 0.0f;
                        if (!(0 & (1u << 25))) part_chunk[25] = 0.0f;
                        if (!(0 & (1u << 26))) part_chunk[26] = 0.0f;
                        if (!(0 & (1u << 27))) part_chunk[27] = 0.0f;
                        if (!(0 & (1u << 28))) part_chunk[28] = 0.0f;
                        if (!(0 & (1u << 29))) part_chunk[29] = 0.0f;
                        if (!(0 & (1u << 30))) part_chunk[30] = 0.0f;
                        if (!(0 & (1u << 31))) part_chunk[31] = 0.0f;
                    }
                    uint32_t _cluster_mapa_10;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_10) : "r"(xbuf_addr + (unsigned int)(((1 - rank) * 8 * 128 + x_thread) * 16)), "r"(1 - rank));
                    uint32_t _mbar_peer_11;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_11) : "r"(xfer_addr), "r"(1 - rank));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_10), "r"(__float_as_uint(part_chunk[0])), "r"(__float_as_uint(part_chunk[1])), "r"(__float_as_uint(part_chunk[2])), "r"(__float_as_uint(part_chunk[3])), "r"(_mbar_peer_11) : "memory");
                    uint32_t _cluster_mapa_12;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_12) : "r"(xbuf_addr + (unsigned int)((((1 - rank) * 8 + 1) * 128 + x_thread) * 16)), "r"(1 - rank));
                    uint32_t _mbar_peer_13;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_13) : "r"(xfer_addr), "r"(1 - rank));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_12), "r"(__float_as_uint(part_chunk[4])), "r"(__float_as_uint(part_chunk[5])), "r"(__float_as_uint(part_chunk[6])), "r"(__float_as_uint(part_chunk[7])), "r"(_mbar_peer_13) : "memory");
                    uint32_t _cluster_mapa_14;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_14) : "r"(xbuf_addr + (unsigned int)((((1 - rank) * 8 + 2) * 128 + x_thread) * 16)), "r"(1 - rank));
                    uint32_t _mbar_peer_15;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_15) : "r"(xfer_addr), "r"(1 - rank));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_14), "r"(__float_as_uint(part_chunk[8])), "r"(__float_as_uint(part_chunk[9])), "r"(__float_as_uint(part_chunk[10])), "r"(__float_as_uint(part_chunk[11])), "r"(_mbar_peer_15) : "memory");
                    uint32_t _cluster_mapa_16;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_16) : "r"(xbuf_addr + (unsigned int)((((1 - rank) * 8 + 3) * 128 + x_thread) * 16)), "r"(1 - rank));
                    uint32_t _mbar_peer_17;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_17) : "r"(xfer_addr), "r"(1 - rank));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_16), "r"(__float_as_uint(part_chunk[12])), "r"(__float_as_uint(part_chunk[13])), "r"(__float_as_uint(part_chunk[14])), "r"(__float_as_uint(part_chunk[15])), "r"(_mbar_peer_17) : "memory");
                    uint32_t _cluster_mapa_18;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_18) : "r"(xbuf_addr + (unsigned int)((((1 - rank) * 8 + 4) * 128 + x_thread) * 16)), "r"(1 - rank));
                    uint32_t _mbar_peer_19;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_19) : "r"(xfer_addr), "r"(1 - rank));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_18), "r"(__float_as_uint(part_chunk[16])), "r"(__float_as_uint(part_chunk[17])), "r"(__float_as_uint(part_chunk[18])), "r"(__float_as_uint(part_chunk[19])), "r"(_mbar_peer_19) : "memory");
                    uint32_t _cluster_mapa_20;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_20) : "r"(xbuf_addr + (unsigned int)((((1 - rank) * 8 + 5) * 128 + x_thread) * 16)), "r"(1 - rank));
                    uint32_t _mbar_peer_21;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_21) : "r"(xfer_addr), "r"(1 - rank));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_20), "r"(__float_as_uint(part_chunk[20])), "r"(__float_as_uint(part_chunk[21])), "r"(__float_as_uint(part_chunk[22])), "r"(__float_as_uint(part_chunk[23])), "r"(_mbar_peer_21) : "memory");
                    uint32_t _cluster_mapa_22;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_22) : "r"(xbuf_addr + (unsigned int)((((1 - rank) * 8 + 6) * 128 + x_thread) * 16)), "r"(1 - rank));
                    uint32_t _mbar_peer_23;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_23) : "r"(xfer_addr), "r"(1 - rank));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_22), "r"(__float_as_uint(part_chunk[24])), "r"(__float_as_uint(part_chunk[25])), "r"(__float_as_uint(part_chunk[26])), "r"(__float_as_uint(part_chunk[27])), "r"(_mbar_peer_23) : "memory");
                    uint32_t _cluster_mapa_24;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_24) : "r"(xbuf_addr + (unsigned int)((((1 - rank) * 8 + 7) * 128 + x_thread) * 16)), "r"(1 - rank));
                    uint32_t _mbar_peer_25;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_25) : "r"(xfer_addr), "r"(1 - rank));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_24), "r"(__float_as_uint(part_chunk[28])), "r"(__float_as_uint(part_chunk[29])), "r"(__float_as_uint(part_chunk[30])), "r"(__float_as_uint(part_chunk[31])), "r"(_mbar_peer_25) : "memory");
                    float part_chunk_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_0[31]))
                        : "r"(taddr + (unsigned int)o0_off + (unsigned int)row_addr + (unsigned int)(rank * 32)));
                    const float2 _scale2_26 = {merge_0, merge_0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(part_chunk_0)[_ls], _scale2_26);
#if __CUDA_ARCH__ == 1030
                    if (pair_count_4 > 1) {
#else
                    if (pair_count_1 > 1) {
#endif
                        float part_chunk_1_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&part_chunk_1_1[31]))
                            : "r"(taddr + (unsigned int)o1_off + (unsigned int)row_addr + (unsigned int)(rank * 32)));
                        #pragma unroll
                        for (int item_2 = 0; item_2 < 32; item_2++) {
                            part_chunk_0[item_2] = __fmaf_rn(part_chunk_1_1[item_2], merge_1, part_chunk_0[item_2]);
                        }
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(tile_done_addr);
                    if (has_mass != 0) {
                        #pragma unroll
                        for (int quad_1 = 0; quad_1 < 8; quad_1++) {
                            float part_vs_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_vs_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_vs_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_vs_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_vs_1[(0) + 3]))
                                : "r"(vs_smem + (unsigned int)((rank * 32 + quad_1 * 4) * 4)));
                            #pragma unroll
                            for (int item_3 = 0; item_3 < 4; item_3++) {
                                part_chunk_0[quad_1 * 4 + item_3] = part_chunk_0[quad_1 * 4 + item_3] * part_vs_1[item_3];
                            }
                        }
                    } else {
                        if (!(0 & (1u << 0))) part_chunk_0[0] = 0.0f;
                        if (!(0 & (1u << 1))) part_chunk_0[1] = 0.0f;
                        if (!(0 & (1u << 2))) part_chunk_0[2] = 0.0f;
                        if (!(0 & (1u << 3))) part_chunk_0[3] = 0.0f;
                        if (!(0 & (1u << 4))) part_chunk_0[4] = 0.0f;
                        if (!(0 & (1u << 5))) part_chunk_0[5] = 0.0f;
                        if (!(0 & (1u << 6))) part_chunk_0[6] = 0.0f;
                        if (!(0 & (1u << 7))) part_chunk_0[7] = 0.0f;
                        if (!(0 & (1u << 8))) part_chunk_0[8] = 0.0f;
                        if (!(0 & (1u << 9))) part_chunk_0[9] = 0.0f;
                        if (!(0 & (1u << 10))) part_chunk_0[10] = 0.0f;
                        if (!(0 & (1u << 11))) part_chunk_0[11] = 0.0f;
                        if (!(0 & (1u << 12))) part_chunk_0[12] = 0.0f;
                        if (!(0 & (1u << 13))) part_chunk_0[13] = 0.0f;
                        if (!(0 & (1u << 14))) part_chunk_0[14] = 0.0f;
                        if (!(0 & (1u << 15))) part_chunk_0[15] = 0.0f;
                        if (!(0 & (1u << 16))) part_chunk_0[16] = 0.0f;
                        if (!(0 & (1u << 17))) part_chunk_0[17] = 0.0f;
                        if (!(0 & (1u << 18))) part_chunk_0[18] = 0.0f;
                        if (!(0 & (1u << 19))) part_chunk_0[19] = 0.0f;
                        if (!(0 & (1u << 20))) part_chunk_0[20] = 0.0f;
                        if (!(0 & (1u << 21))) part_chunk_0[21] = 0.0f;
                        if (!(0 & (1u << 22))) part_chunk_0[22] = 0.0f;
                        if (!(0 & (1u << 23))) part_chunk_0[23] = 0.0f;
                        if (!(0 & (1u << 24))) part_chunk_0[24] = 0.0f;
                        if (!(0 & (1u << 25))) part_chunk_0[25] = 0.0f;
                        if (!(0 & (1u << 26))) part_chunk_0[26] = 0.0f;
                        if (!(0 & (1u << 27))) part_chunk_0[27] = 0.0f;
                        if (!(0 & (1u << 28))) part_chunk_0[28] = 0.0f;
                        if (!(0 & (1u << 29))) part_chunk_0[29] = 0.0f;
                        if (!(0 & (1u << 30))) part_chunk_0[30] = 0.0f;
                        if (!(0 & (1u << 31))) part_chunk_0[31] = 0.0f;
                    }
                    mbarrier_wait_cluster_hint(xfer_addr, 0, 10000000);
                    float stats_p[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&stats_p[0])), "=r"(*reinterpret_cast<uint32_t*>(&stats_p[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&stats_p[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&stats_p[(0) + 3]))
                        : "r"(xbuf_addr + (unsigned int)((16 * 128 + x_thread) * 16)));
                    float max_a = ((rank == 0) ? own_max : stats_p[0]);
                    float sum_a = ((rank == 0) ? own_sum : stats_p[1]);
                    float max_b = ((rank == 0) ? stats_p[0] : own_max);
                    float sum_b = ((rank == 0) ? stats_p[1] : own_sum);
                    float fold_max;
                    asm("max.f32 %0, %1, %2;" : "=f"(fold_max) : "f"(max_a), "f"(max_b));
                    float _exp2_27 = approx_exp2(max_a - fold_max);
                    float fold_a = ((max_a > -CUDART_INF_F) ? _exp2_27 : 0.0f);
                    float _exp2_28 = approx_exp2(max_b - fold_max);
                    float fold_b = ((max_b > -CUDART_INF_F) ? _exp2_28 : 0.0f);
                    float fold_sum = __fmaf_rn(sum_a, fold_a, sum_b * fold_b);
                    int fold_mass = ((fold_sum > 0.0f && fold_sum == fold_sum) ? 1 : 0);
                    float _rcp_29 = approx_rcp(fold_sum);
                    float fold_inv = ((fold_mass != 0) ? _rcp_29 : 0.0f);
                    float peer_o_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&peer_o_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_0[(0) + 3]))
                        : "r"(xbuf_addr + (unsigned int)((rank * 8 * 128 + x_thread) * 16)));
                    float o_a_0 = ((rank == 0) ? part_chunk_0[0] : peer_o_0[0]);
                    float o_b_0 = ((rank == 0) ? peer_o_0[0] : part_chunk_0[0]);
                    part_chunk_0[0] = __fmaf_rn(o_b_0, fold_b, o_a_0 * fold_a) * fold_inv;
                    float o_a_1 = ((rank == 0) ? part_chunk_0[1] : peer_o_0[1]);
                    float o_b_1 = ((rank == 0) ? peer_o_0[1] : part_chunk_0[1]);
                    part_chunk_0[1] = __fmaf_rn(o_b_1, fold_b, o_a_1 * fold_a) * fold_inv;
                    float o_a_2 = ((rank == 0) ? part_chunk_0[2] : peer_o_0[2]);
                    float o_b_2 = ((rank == 0) ? peer_o_0[2] : part_chunk_0[2]);
                    part_chunk_0[2] = __fmaf_rn(o_b_2, fold_b, o_a_2 * fold_a) * fold_inv;
                    float o_a_3 = ((rank == 0) ? part_chunk_0[3] : peer_o_0[3]);
                    float o_b_3 = ((rank == 0) ? peer_o_0[3] : part_chunk_0[3]);
                    part_chunk_0[3] = __fmaf_rn(o_b_3, fold_b, o_a_3 * fold_a) * fold_inv;
                    float peer_o_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&peer_o_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_1[(0) + 3]))
                        : "r"(xbuf_addr + (unsigned int)(((rank * 8 + 1) * 128 + x_thread) * 16)));
                    float o_a_4 = ((rank == 0) ? part_chunk_0[4] : peer_o_1[0]);
                    float o_b_4 = ((rank == 0) ? peer_o_1[0] : part_chunk_0[4]);
                    part_chunk_0[4] = __fmaf_rn(o_b_4, fold_b, o_a_4 * fold_a) * fold_inv;
                    float o_a_5 = ((rank == 0) ? part_chunk_0[5] : peer_o_1[1]);
                    float o_b_5 = ((rank == 0) ? peer_o_1[1] : part_chunk_0[5]);
                    part_chunk_0[5] = __fmaf_rn(o_b_5, fold_b, o_a_5 * fold_a) * fold_inv;
                    float o_a_6 = ((rank == 0) ? part_chunk_0[6] : peer_o_1[2]);
                    float o_b_6 = ((rank == 0) ? peer_o_1[2] : part_chunk_0[6]);
                    part_chunk_0[6] = __fmaf_rn(o_b_6, fold_b, o_a_6 * fold_a) * fold_inv;
                    float o_a_7 = ((rank == 0) ? part_chunk_0[7] : peer_o_1[3]);
                    float o_b_7 = ((rank == 0) ? peer_o_1[3] : part_chunk_0[7]);
                    part_chunk_0[7] = __fmaf_rn(o_b_7, fold_b, o_a_7 * fold_a) * fold_inv;
                    float peer_o_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&peer_o_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_2[(0) + 3]))
                        : "r"(xbuf_addr + (unsigned int)(((rank * 8 + 2) * 128 + x_thread) * 16)));
                    float o_a_8 = ((rank == 0) ? part_chunk_0[8] : peer_o_2[0]);
                    float o_b_8 = ((rank == 0) ? peer_o_2[0] : part_chunk_0[8]);
                    part_chunk_0[8] = __fmaf_rn(o_b_8, fold_b, o_a_8 * fold_a) * fold_inv;
                    float o_a_9 = ((rank == 0) ? part_chunk_0[9] : peer_o_2[1]);
                    float o_b_9 = ((rank == 0) ? peer_o_2[1] : part_chunk_0[9]);
                    part_chunk_0[9] = __fmaf_rn(o_b_9, fold_b, o_a_9 * fold_a) * fold_inv;
                    float o_a_10 = ((rank == 0) ? part_chunk_0[10] : peer_o_2[2]);
                    float o_b_10 = ((rank == 0) ? peer_o_2[2] : part_chunk_0[10]);
                    part_chunk_0[10] = __fmaf_rn(o_b_10, fold_b, o_a_10 * fold_a) * fold_inv;
                    float o_a_11 = ((rank == 0) ? part_chunk_0[11] : peer_o_2[3]);
                    float o_b_11 = ((rank == 0) ? peer_o_2[3] : part_chunk_0[11]);
                    part_chunk_0[11] = __fmaf_rn(o_b_11, fold_b, o_a_11 * fold_a) * fold_inv;
                    float peer_o_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&peer_o_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_3[(0) + 3]))
                        : "r"(xbuf_addr + (unsigned int)(((rank * 8 + 3) * 128 + x_thread) * 16)));
                    float o_a_12 = ((rank == 0) ? part_chunk_0[12] : peer_o_3[0]);
                    float o_b_12 = ((rank == 0) ? peer_o_3[0] : part_chunk_0[12]);
                    part_chunk_0[12] = __fmaf_rn(o_b_12, fold_b, o_a_12 * fold_a) * fold_inv;
                    float o_a_13 = ((rank == 0) ? part_chunk_0[13] : peer_o_3[1]);
                    float o_b_13 = ((rank == 0) ? peer_o_3[1] : part_chunk_0[13]);
                    part_chunk_0[13] = __fmaf_rn(o_b_13, fold_b, o_a_13 * fold_a) * fold_inv;
                    float o_a_14 = ((rank == 0) ? part_chunk_0[14] : peer_o_3[2]);
                    float o_b_14 = ((rank == 0) ? peer_o_3[2] : part_chunk_0[14]);
                    part_chunk_0[14] = __fmaf_rn(o_b_14, fold_b, o_a_14 * fold_a) * fold_inv;
                    float o_a_15 = ((rank == 0) ? part_chunk_0[15] : peer_o_3[3]);
                    float o_b_15 = ((rank == 0) ? peer_o_3[3] : part_chunk_0[15]);
                    part_chunk_0[15] = __fmaf_rn(o_b_15, fold_b, o_a_15 * fold_a) * fold_inv;
                    float peer_o_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&peer_o_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_4[(0) + 3]))
                        : "r"(xbuf_addr + (unsigned int)(((rank * 8 + 4) * 128 + x_thread) * 16)));
                    float o_a_16 = ((rank == 0) ? part_chunk_0[16] : peer_o_4[0]);
                    float o_b_16 = ((rank == 0) ? peer_o_4[0] : part_chunk_0[16]);
                    part_chunk_0[16] = __fmaf_rn(o_b_16, fold_b, o_a_16 * fold_a) * fold_inv;
                    float o_a_17 = ((rank == 0) ? part_chunk_0[17] : peer_o_4[1]);
                    float o_b_17 = ((rank == 0) ? peer_o_4[1] : part_chunk_0[17]);
                    part_chunk_0[17] = __fmaf_rn(o_b_17, fold_b, o_a_17 * fold_a) * fold_inv;
                    float o_a_18 = ((rank == 0) ? part_chunk_0[18] : peer_o_4[2]);
                    float o_b_18 = ((rank == 0) ? peer_o_4[2] : part_chunk_0[18]);
                    part_chunk_0[18] = __fmaf_rn(o_b_18, fold_b, o_a_18 * fold_a) * fold_inv;
                    float o_a_19 = ((rank == 0) ? part_chunk_0[19] : peer_o_4[3]);
                    float o_b_19 = ((rank == 0) ? peer_o_4[3] : part_chunk_0[19]);
                    part_chunk_0[19] = __fmaf_rn(o_b_19, fold_b, o_a_19 * fold_a) * fold_inv;
                    float peer_o_5[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&peer_o_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_5[(0) + 3]))
                        : "r"(xbuf_addr + (unsigned int)(((rank * 8 + 5) * 128 + x_thread) * 16)));
                    float o_a_20 = ((rank == 0) ? part_chunk_0[20] : peer_o_5[0]);
                    float o_b_20 = ((rank == 0) ? peer_o_5[0] : part_chunk_0[20]);
                    part_chunk_0[20] = __fmaf_rn(o_b_20, fold_b, o_a_20 * fold_a) * fold_inv;
                    float o_a_21 = ((rank == 0) ? part_chunk_0[21] : peer_o_5[1]);
                    float o_b_21 = ((rank == 0) ? peer_o_5[1] : part_chunk_0[21]);
                    part_chunk_0[21] = __fmaf_rn(o_b_21, fold_b, o_a_21 * fold_a) * fold_inv;
                    float o_a_22 = ((rank == 0) ? part_chunk_0[22] : peer_o_5[2]);
                    float o_b_22 = ((rank == 0) ? peer_o_5[2] : part_chunk_0[22]);
                    part_chunk_0[22] = __fmaf_rn(o_b_22, fold_b, o_a_22 * fold_a) * fold_inv;
                    float o_a_23 = ((rank == 0) ? part_chunk_0[23] : peer_o_5[3]);
                    float o_b_23 = ((rank == 0) ? peer_o_5[3] : part_chunk_0[23]);
                    part_chunk_0[23] = __fmaf_rn(o_b_23, fold_b, o_a_23 * fold_a) * fold_inv;
                    float peer_o_6[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&peer_o_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_6[(0) + 3]))
                        : "r"(xbuf_addr + (unsigned int)(((rank * 8 + 6) * 128 + x_thread) * 16)));
                    float o_a_24 = ((rank == 0) ? part_chunk_0[24] : peer_o_6[0]);
                    float o_b_24 = ((rank == 0) ? peer_o_6[0] : part_chunk_0[24]);
                    part_chunk_0[24] = __fmaf_rn(o_b_24, fold_b, o_a_24 * fold_a) * fold_inv;
                    float o_a_25 = ((rank == 0) ? part_chunk_0[25] : peer_o_6[1]);
                    float o_b_25 = ((rank == 0) ? peer_o_6[1] : part_chunk_0[25]);
                    part_chunk_0[25] = __fmaf_rn(o_b_25, fold_b, o_a_25 * fold_a) * fold_inv;
                    float o_a_26 = ((rank == 0) ? part_chunk_0[26] : peer_o_6[2]);
                    float o_b_26 = ((rank == 0) ? peer_o_6[2] : part_chunk_0[26]);
                    part_chunk_0[26] = __fmaf_rn(o_b_26, fold_b, o_a_26 * fold_a) * fold_inv;
                    float o_a_27 = ((rank == 0) ? part_chunk_0[27] : peer_o_6[3]);
                    float o_b_27 = ((rank == 0) ? peer_o_6[3] : part_chunk_0[27]);
                    part_chunk_0[27] = __fmaf_rn(o_b_27, fold_b, o_a_27 * fold_a) * fold_inv;
                    float peer_o_7[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&peer_o_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_o_7[(0) + 3]))
                        : "r"(xbuf_addr + (unsigned int)(((rank * 8 + 7) * 128 + x_thread) * 16)));
                    float o_a_28 = ((rank == 0) ? part_chunk_0[28] : peer_o_7[0]);
                    float o_b_28 = ((rank == 0) ? peer_o_7[0] : part_chunk_0[28]);
                    part_chunk_0[28] = __fmaf_rn(o_b_28, fold_b, o_a_28 * fold_a) * fold_inv;
                    float o_a_29 = ((rank == 0) ? part_chunk_0[29] : peer_o_7[1]);
                    float o_b_29 = ((rank == 0) ? peer_o_7[1] : part_chunk_0[29]);
                    part_chunk_0[29] = __fmaf_rn(o_b_29, fold_b, o_a_29 * fold_a) * fold_inv;
                    float o_a_30 = ((rank == 0) ? part_chunk_0[30] : peer_o_7[2]);
                    float o_b_30 = ((rank == 0) ? peer_o_7[2] : part_chunk_0[30]);
                    part_chunk_0[30] = __fmaf_rn(o_b_30, fold_b, o_a_30 * fold_a) * fold_inv;
                    float o_a_31 = ((rank == 0) ? part_chunk_0[31] : peer_o_7[3]);
                    float o_b_31 = ((rank == 0) ? peer_o_7[3] : part_chunk_0[31]);
                    part_chunk_0[31] = __fmaf_rn(o_b_31, fold_b, o_a_31 * fold_a) * fold_inv;
#if __CUDA_ARCH__ == 1030
                    if (my_row_1 < q_valid_4) {
#else
                    if (my_row_1 < q_valid_1) {
#endif
                        unsigned int fold_packed_0[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 0], part_chunk_0[_lp*2+1 + 0]));
                            fold_packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 2], part_chunk_0[_lp*2+1 + 2]));
                            fold_packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 4], part_chunk_0[_lp*2+1 + 4]));
                            fold_packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 6], part_chunk_0[_lp*2+1 + 6]));
                            fold_packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(out + (output_row + col_base + rank * 32))[0] = reinterpret_cast<int4*>(fold_packed_0)[0];
                        unsigned int fold_packed_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 8], part_chunk_0[_lp*2+1 + 8]));
                            fold_packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 10], part_chunk_0[_lp*2+1 + 10]));
                            fold_packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 12], part_chunk_0[_lp*2+1 + 12]));
                            fold_packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 14], part_chunk_0[_lp*2+1 + 14]));
                            fold_packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(out + (output_row + col_base + rank * 32 + 8))[0] = reinterpret_cast<int4*>(fold_packed_1)[0];
                        unsigned int fold_packed_2[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 16], part_chunk_0[_lp*2+1 + 16]));
                            fold_packed_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 18], part_chunk_0[_lp*2+1 + 18]));
                            fold_packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 20], part_chunk_0[_lp*2+1 + 20]));
                            fold_packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 22], part_chunk_0[_lp*2+1 + 22]));
                            fold_packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(out + (output_row + col_base + rank * 32 + 16))[0] = reinterpret_cast<int4*>(fold_packed_2)[0];
                        unsigned int fold_packed_3[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 24], part_chunk_0[_lp*2+1 + 24]));
                            fold_packed_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 26], part_chunk_0[_lp*2+1 + 26]));
                            fold_packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 28], part_chunk_0[_lp*2+1 + 28]));
                            fold_packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(part_chunk_0[_lp*2 + 30], part_chunk_0[_lp*2+1 + 30]));
                            fold_packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(out + (output_row + col_base + rank * 32 + 24))[0] = reinterpret_cast<int4*>(fold_packed_3)[0];
                    }
#if __CUDA_ARCH__ == 1030
                    if (rank == 0 && my_row_1 < q_valid_4 && col_half_1 == 0 && return_lse != 0) {
#else
                    if (rank == 0 && my_row_1 < q_valid_1 && col_half_1 == 0 && return_lse != 0) {
#endif
                        float _log2_30;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_30) : "f"(fold_sum));
#if __CUDA_ARCH__ == 1030
                        lse[bh_3 * seqlen_q + query] = ((fold_mass != 0) ? (fold_max + _log2_30 - 8.0f) * 0.6931472f : -CUDART_INF_F);
#else
                        lse[bh * seqlen_q + query] = ((fold_mass != 0) ? (fold_max + _log2_30 - 8.0f) * 0.6931472f : -CUDART_INF_F);
#endif
                    }
                } else {
                    const int chunk_cols_0 = 0;
                    float tmem_load_2[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[31]))
                        : "r"(taddr + (unsigned int)o0_off + (unsigned int)row_addr + (unsigned int)chunk_cols_0));
                    const float2 _scale2_31 = {merge_0, merge_0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(tmem_load_2)[_ls], _scale2_31);
#if __CUDA_ARCH__ == 1030
                    if (pair_count_4 > 1) {
#else
                    if (pair_count_1 > 1) {
#endif
                        float tmem_load_1_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[31]))
                            : "r"(taddr + (unsigned int)o1_off + (unsigned int)row_addr + (unsigned int)chunk_cols_0));
                        #pragma unroll
                        for (int item_4 = 0; item_4 < 32; item_4++) {
                            tmem_load_2[item_4] = __fmaf_rn(tmem_load_1_1[item_4], merge_1, tmem_load_2[item_4]);
                        }
                    }
                    if (has_mass != 0) {
                        #pragma unroll
                        for (int quad_2 = 0; quad_2 < 8; quad_2++) {
                            float _vec_load_0[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_vec_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_vec_load_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&_vec_load_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&_vec_load_0[(0) + 3]))
                                : "r"(vs_smem + (unsigned int)((chunk_cols_0 + quad_2 * 4) * 4)));
                            #pragma unroll
                            for (int item_5 = 0; item_5 < 4; item_5++) {
                                tmem_load_2[quad_2 * 4 + item_5] = tmem_load_2[quad_2 * 4 + item_5] * _vec_load_0[item_5];
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
#if __CUDA_ARCH__ == 1030
                    if (my_row_1 < q_valid_4) {
#else
                    if (my_row_1 < q_valid_1) {
#endif
                        bool out32 = ((unsigned long long)out & 31ull) == 0ull;
                        bool out16 = ((unsigned long long)out & 31ull) != 0ull;
                        unsigned int st_packed_0[8];
                        float st_scaled_0[8];
                        const float2 _scale2_32 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            reinterpret_cast<float2*>(st_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((tmem_load_2 + 0))[_ls], _scale2_32);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_0[_lp*2 + 0], st_scaled_0[_lp*2+1 + 0]));
                            st_packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_0[_lp*2 + 2], st_scaled_0[_lp*2+1 + 2]));
                            st_packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_0[_lp*2 + 4], st_scaled_0[_lp*2+1 + 4]));
                            st_packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_0[_lp*2 + 6], st_scaled_0[_lp*2+1 + 6]));
                            st_packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        float st_scaled_1[8];
                        const float2 _scale2_33 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            reinterpret_cast<float2*>(st_scaled_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((tmem_load_2 + 8))[_ls], _scale2_33);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_1[_lp*2 + 0], st_scaled_1[_lp*2+1 + 0]));
                            st_packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_1[_lp*2 + 2], st_scaled_1[_lp*2+1 + 2]));
                            st_packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_1[_lp*2 + 4], st_scaled_1[_lp*2+1 + 4]));
                            st_packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_1[_lp*2 + 6], st_scaled_1[_lp*2+1 + 6]));
                            st_packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(out + (output_row + col_base + chunk_cols_0)), "r"(st_packed_0[(0) + 0]), "r"(st_packed_0[(0) + 1]), "r"(st_packed_0[(0) + 2]), "r"(st_packed_0[(0) + 3]), "r"(st_packed_0[(0) + 4]), "r"(st_packed_0[(0) + 5]), "r"(st_packed_0[(0) + 6]), "r"(st_packed_0[(0) + 7]), "r"((unsigned int)(out32)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (output_row + col_base + chunk_cols_0)), "r"(st_packed_0[(0) + 0]), "r"(st_packed_0[(0) + 1]), "r"(st_packed_0[(0) + 2]), "r"(st_packed_0[(0) + 3]), "r"((unsigned int)(out16)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (output_row + col_base + chunk_cols_0 + 8)), "r"(st_packed_0[(4) + 0]), "r"(st_packed_0[(4) + 1]), "r"(st_packed_0[(4) + 2]), "r"(st_packed_0[(4) + 3]), "r"((unsigned int)(out16)) : "memory");
                        unsigned int st_packed_1[8];
                        float st_scaled_2[8];
                        const float2 _scale2_34 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            reinterpret_cast<float2*>(st_scaled_2)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((tmem_load_2 + 16))[_ls], _scale2_34);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_2[_lp*2 + 0], st_scaled_2[_lp*2+1 + 0]));
                            st_packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_2[_lp*2 + 2], st_scaled_2[_lp*2+1 + 2]));
                            st_packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_2[_lp*2 + 4], st_scaled_2[_lp*2+1 + 4]));
                            st_packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_2[_lp*2 + 6], st_scaled_2[_lp*2+1 + 6]));
                            st_packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        float st_scaled_3[8];
                        const float2 _scale2_35 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            reinterpret_cast<float2*>(st_scaled_3)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((tmem_load_2 + 24))[_ls], _scale2_35);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_3[_lp*2 + 0], st_scaled_3[_lp*2+1 + 0]));
                            st_packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_3[_lp*2 + 2], st_scaled_3[_lp*2+1 + 2]));
                            st_packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_3[_lp*2 + 4], st_scaled_3[_lp*2+1 + 4]));
                            st_packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_3[_lp*2 + 6], st_scaled_3[_lp*2+1 + 6]));
                            st_packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(out + (output_row + col_base + chunk_cols_0 + 16)), "r"(st_packed_1[(0) + 0]), "r"(st_packed_1[(0) + 1]), "r"(st_packed_1[(0) + 2]), "r"(st_packed_1[(0) + 3]), "r"(st_packed_1[(0) + 4]), "r"(st_packed_1[(0) + 5]), "r"(st_packed_1[(0) + 6]), "r"(st_packed_1[(0) + 7]), "r"((unsigned int)(out32)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (output_row + col_base + chunk_cols_0 + 16)), "r"(st_packed_1[(0) + 0]), "r"(st_packed_1[(0) + 1]), "r"(st_packed_1[(0) + 2]), "r"(st_packed_1[(0) + 3]), "r"((unsigned int)(out16)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (output_row + col_base + chunk_cols_0 + 16 + 8)), "r"(st_packed_1[(4) + 0]), "r"(st_packed_1[(4) + 1]), "r"(st_packed_1[(4) + 2]), "r"(st_packed_1[(4) + 3]), "r"((unsigned int)(out16)) : "memory");
                    }
                    const int chunk_cols_1 = 32;
                    float tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[31]))
                        : "r"(taddr + (unsigned int)o0_off + (unsigned int)row_addr + (unsigned int)chunk_cols_1));
                    const float2 _scale2_36 = {merge_0, merge_0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(tmem_load_0)[_ls], _scale2_36);
#if __CUDA_ARCH__ == 1030
                    if (pair_count_4 > 1) {
#else
                    if (pair_count_1 > 1) {
#endif
                        float tmem_load_1_2[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_2[31]))
                            : "r"(taddr + (unsigned int)o1_off + (unsigned int)row_addr + (unsigned int)chunk_cols_1));
                        #pragma unroll
                        for (int item_6 = 0; item_6 < 32; item_6++) {
                            tmem_load_0[item_6] = __fmaf_rn(tmem_load_1_2[item_6], merge_1, tmem_load_0[item_6]);
                        }
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(tile_done_addr);
                    if (has_mass != 0) {
                        #pragma unroll
                        for (int quad_3 = 0; quad_3 < 8; quad_3++) {
                            float _vec_load_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_vec_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_vec_load_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&_vec_load_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&_vec_load_1[(0) + 3]))
                                : "r"(vs_smem + (unsigned int)((chunk_cols_1 + quad_3 * 4) * 4)));
                            #pragma unroll
                            for (int item_7 = 0; item_7 < 4; item_7++) {
                                tmem_load_0[quad_3 * 4 + item_7] = tmem_load_0[quad_3 * 4 + item_7] * _vec_load_1[item_7];
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
#if __CUDA_ARCH__ == 1030
                    if (my_row_1 < q_valid_4) {
#else
                    if (my_row_1 < q_valid_1) {
#endif
                        bool out32_1 = ((unsigned long long)out & 31ull) == 0ull;
                        bool out16_1 = ((unsigned long long)out & 31ull) != 0ull;
                        unsigned int st_packed_2[8];
                        float st_scaled_4[8];
                        const float2 _scale2_37 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            reinterpret_cast<float2*>(st_scaled_4)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((tmem_load_0 + 0))[_ls], _scale2_37);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_4[_lp*2 + 0], st_scaled_4[_lp*2+1 + 0]));
                            st_packed_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_4[_lp*2 + 2], st_scaled_4[_lp*2+1 + 2]));
                            st_packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_4[_lp*2 + 4], st_scaled_4[_lp*2+1 + 4]));
                            st_packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_4[_lp*2 + 6], st_scaled_4[_lp*2+1 + 6]));
                            st_packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        float st_scaled_5[8];
                        const float2 _scale2_38 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            reinterpret_cast<float2*>(st_scaled_5)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((tmem_load_0 + 8))[_ls], _scale2_38);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_5[_lp*2 + 0], st_scaled_5[_lp*2+1 + 0]));
                            st_packed_2[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_5[_lp*2 + 2], st_scaled_5[_lp*2+1 + 2]));
                            st_packed_2[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_5[_lp*2 + 4], st_scaled_5[_lp*2+1 + 4]));
                            st_packed_2[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_5[_lp*2 + 6], st_scaled_5[_lp*2+1 + 6]));
                            st_packed_2[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(out + (output_row + col_base + chunk_cols_1)), "r"(st_packed_2[(0) + 0]), "r"(st_packed_2[(0) + 1]), "r"(st_packed_2[(0) + 2]), "r"(st_packed_2[(0) + 3]), "r"(st_packed_2[(0) + 4]), "r"(st_packed_2[(0) + 5]), "r"(st_packed_2[(0) + 6]), "r"(st_packed_2[(0) + 7]), "r"((unsigned int)(out32_1)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (output_row + col_base + chunk_cols_1)), "r"(st_packed_2[(0) + 0]), "r"(st_packed_2[(0) + 1]), "r"(st_packed_2[(0) + 2]), "r"(st_packed_2[(0) + 3]), "r"((unsigned int)(out16_1)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (output_row + col_base + chunk_cols_1 + 8)), "r"(st_packed_2[(4) + 0]), "r"(st_packed_2[(4) + 1]), "r"(st_packed_2[(4) + 2]), "r"(st_packed_2[(4) + 3]), "r"((unsigned int)(out16_1)) : "memory");
                        unsigned int st_packed_3[8];
                        float st_scaled_6[8];
                        const float2 _scale2_39 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            reinterpret_cast<float2*>(st_scaled_6)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((tmem_load_0 + 16))[_ls], _scale2_39);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_6[_lp*2 + 0], st_scaled_6[_lp*2+1 + 0]));
                            st_packed_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_6[_lp*2 + 2], st_scaled_6[_lp*2+1 + 2]));
                            st_packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_6[_lp*2 + 4], st_scaled_6[_lp*2+1 + 4]));
                            st_packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_6[_lp*2 + 6], st_scaled_6[_lp*2+1 + 6]));
                            st_packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        float st_scaled_7[8];
                        const float2 _scale2_40 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            reinterpret_cast<float2*>(st_scaled_7)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((tmem_load_0 + 24))[_ls], _scale2_40);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_7[_lp*2 + 0], st_scaled_7[_lp*2+1 + 0]));
                            st_packed_3[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_7[_lp*2 + 2], st_scaled_7[_lp*2+1 + 2]));
                            st_packed_3[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_7[_lp*2 + 4], st_scaled_7[_lp*2+1 + 4]));
                            st_packed_3[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(st_scaled_7[_lp*2 + 6], st_scaled_7[_lp*2+1 + 6]));
                            st_packed_3[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(out + (output_row + col_base + chunk_cols_1 + 16)), "r"(st_packed_3[(0) + 0]), "r"(st_packed_3[(0) + 1]), "r"(st_packed_3[(0) + 2]), "r"(st_packed_3[(0) + 3]), "r"(st_packed_3[(0) + 4]), "r"(st_packed_3[(0) + 5]), "r"(st_packed_3[(0) + 6]), "r"(st_packed_3[(0) + 7]), "r"((unsigned int)(out32_1)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (output_row + col_base + chunk_cols_1 + 16)), "r"(st_packed_3[(0) + 0]), "r"(st_packed_3[(0) + 1]), "r"(st_packed_3[(0) + 2]), "r"(st_packed_3[(0) + 3]), "r"((unsigned int)(out16_1)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (output_row + col_base + chunk_cols_1 + 16 + 8)), "r"(st_packed_3[(4) + 0]), "r"(st_packed_3[(4) + 1]), "r"(st_packed_3[(4) + 2]), "r"(st_packed_3[(4) + 3]), "r"((unsigned int)(out16_1)) : "memory");
                    }
#if __CUDA_ARCH__ == 1030
                    if (my_row_1 < q_valid_4 && col_half_1 == 0 && return_lse != 0) {
                        int stat_idx = bh_3 * seqlen_q + query;
#else
                    if (my_row_1 < q_valid_1 && col_half_1 == 0 && return_lse != 0) {
                        int stat_idx = bh * seqlen_q + query;
#endif
                        float _log2_41;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_41) : "f"(total_sum));
                        lse[stat_idx] = ((has_mass != 0) ? (row_max_1 + _log2_41 - 8.0f) * 0.6931472f : -CUDART_INF_F);
                    }
                }
#if __CUDA_ARCH__ == 1030
                if (my_tiles_2 > tile_iter_2 + 2) {
#else
                if (my_tiles_1 > tile_iter_1 + 2) {
#endif
                    mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)corr_tiles_done % 2)) * 8);
                }
                corr_tiles_done += 1;
#if __CUDA_ARCH__ == 1030
                steps_before_1 = steps_before_1 + pair_count_4;
#else
                steps_before_1 = steps_before_1 + pair_count_1;
#endif
            }
        }
    }
#if __CUDA_ARCH__ == 1030
    // ---- Role: empty ----
    if (warp >= 14 && warp <= 15) {
        { // empty_main
            int p_group = make_warp_uniform(warp - 14);
            int p_lane = lane;
            int p_blk = ((p_lane < 8) ? p_lane / 4 : p_lane % 2);
            int p_kg = p_lane % 4;
            int p_n = p_group * 4;
            int count_3 = (total_tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
            int my_tiles_3 = count_3;
            int steps_before_2 = 0;
            unsigned int _phase_step_empty = 1;
            #pragma unroll 1
            for (int tile_iter_3 = 0; tile_iter_3 < my_tiles_3; tile_iter_3++) {
                int tile_idx_2 = blockIdx.x + tile_iter_3 * gridDim.x;
                int split_d_3 = tile_idx_2 - split_from;
                if (split_d_3 < 0) {
                    split_d_3 = 0;
                }
                int tile_id_3 = tile_idx_2 - (split_d_3 + 1) / 2;
                int q_block_3 = tile_id_3 % q_blocks;
                int bh_4 = tile_id_3 / q_blocks;
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
                if (tile_idx_2 >= split_from) {
                    int half_pairs_4 = (pair_count_5 + 1) / 2;
                    if ((tile_idx_2 - split_from) % 2 == 0) {
                        pair_count_5 = half_pairs_4;
                    } else {
                        pair_count_5 = pair_count_5 - half_pairs_4;
                        if (pair_count_5 < 1) {
                            pair_count_5 = 1;
#else
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            unsigned int kv_stage = 0;
            unsigned int q_stage = 0;
            int count_2 = (total_tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
            int my_tiles_2 = count_2;
            unsigned int s_cur = 0;
            unsigned int p_cur = 0;
            int q_iter = 0;
            int q_step = 0;
            int q_pairs = 0;
            unsigned int sd_e_cur = 0;
            unsigned int pv_cur = 0;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_kv_full = 0;
            if (q_iter < my_tiles_2) {
                if (q_step == 0) {
                    int split_d = blockIdx.x + q_iter * gridDim.x - split_from;
                    if (split_d < 0) {
                        split_d = 0;
                    }
                    int tile_id = blockIdx.x + q_iter * gridDim.x - (split_d + 1) / 2;
                    int q_block = tile_id % q_blocks;
                    int bh_1 = tile_id / q_blocks;
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
                    if (blockIdx.x + q_iter * gridDim.x >= split_from) {
                        int half_pairs = (pair_count_2 + 1) / 2;
                        if ((blockIdx.x + q_iter * gridDim.x - split_from) % 2 == 0) {
                            pair_count_2 = half_pairs;
                        } else {
                            pair_count_2 = pair_count_2 - half_pairs;
                            if (pair_count_2 < 1) {
                                pair_count_2 = 1;
                            }
                            index_base_1 = index_base_1 + half_pairs * 2;
                            selected_count_1 = selected_count_1 - half_pairs * 2;
                            if (selected_count_1 < 0) {
                                selected_count_1 = 0;
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        index_base_4 = index_base_4 + half_pairs_4 * 2;
                        selected_count_4 = selected_count_4 - half_pairs_4 * 2;
                        if (selected_count_4 < 0) {
                            selected_count_4 = 0;
#else
                    }
                    q_pairs = pair_count_2;
                    mbarrier_wait(q_full_addr + (q_stage) * 8, _phase_q_full);
                }
                unsigned int k_stage = kv_stage;
                mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage) * 512);
                int mma_b_lo = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                    }
                }
                elect_commit(s_full_addr + (s_cur) * 8);
                s_cur += 1;
                if (s_cur == 2) { s_cur = 0; }
                elect_commit(kv_empty_addr + (kv_stage) * 8);
                kv_stage += 1;
                if (kv_stage == 6) { kv_stage = 0; _phase_kv_full ^= 1; }
                q_step = q_step + 1;
                if (q_step == q_pairs) {
                    if (my_tiles_2 > q_iter + 1) {
                        elect_commit(q_empty_addr + (q_stage) * 8);
                    }
                    q_stage += 1;
                    if (q_stage == 2) { q_stage = 0; _phase_q_full ^= 1; }
                    q_step = 0;
                    q_iter = q_iter + 1;
                }
            }
            if (q_iter < my_tiles_2) {
                if (q_step == 0) {
                    int split_d_1 = blockIdx.x + q_iter * gridDim.x - split_from;
                    if (split_d_1 < 0) {
                        split_d_1 = 0;
                    }
                    int tile_id_1 = blockIdx.x + q_iter * gridDim.x - (split_d_1 + 1) / 2;
                    int q_block_1 = tile_id_1 % q_blocks;
                    int bh_2 = tile_id_1 / q_blocks;
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
                    if (blockIdx.x + q_iter * gridDim.x >= split_from) {
                        int half_pairs_1 = (pair_count_3 + 1) / 2;
                        if ((blockIdx.x + q_iter * gridDim.x - split_from) % 2 == 0) {
                            pair_count_3 = half_pairs_1;
                        } else {
                            pair_count_3 = pair_count_3 - half_pairs_1;
                            if (pair_count_3 < 1) {
                                pair_count_3 = 1;
                            }
                            index_base_2 = index_base_2 + half_pairs_1 * 2;
                            selected_count_2 = selected_count_2 - half_pairs_1 * 2;
                            if (selected_count_2 < 0) {
                                selected_count_2 = 0;
                            }
                        }
                    }
                    q_pairs = pair_count_3;
                    mbarrier_wait(q_full_addr + (q_stage) * 8, _phase_q_full);
                }
                unsigned int k_stage_1 = kv_stage;
                mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo_1 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage) * 512);
                int mma_b_lo_1 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage_1) * 1024);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                    }
                }
                elect_commit(s_full_addr + (s_cur) * 8);
                s_cur += 1;
                if (s_cur == 2) { s_cur = 0; }
                elect_commit(kv_empty_addr + (kv_stage) * 8);
                kv_stage += 1;
                if (kv_stage == 6) { kv_stage = 0; _phase_kv_full ^= 1; }
                q_step = q_step + 1;
                if (q_step == q_pairs) {
                    if (my_tiles_2 > q_iter + 1) {
                        elect_commit(q_empty_addr + (q_stage) * 8);
                    }
                    q_stage += 1;
                    if (q_stage == 2) { q_stage = 0; _phase_q_full ^= 1; }
                    q_step = 0;
                    q_iter = q_iter + 1;
                }
            }
            unsigned int _phase_s_empty = 0;
            unsigned int _phase_p_full = 0;
            unsigned int _phase_tile_done_0 = 0;
            #pragma unroll 1
            for (int tile_iter_2 = 0; tile_iter_2 < my_tiles_2; tile_iter_2++) {
                int tile_idx_1 = blockIdx.x + tile_iter_2 * gridDim.x;
                int split_d_2 = tile_idx_1 - split_from;
                if (split_d_2 < 0) {
                    split_d_2 = 0;
                }
                int tile_id_2 = tile_idx_1 - (split_d_2 + 1) / 2;
                int q_block_2 = tile_id_2 % q_blocks;
                int bh_3 = tile_id_2 / q_blocks;
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
                if (tile_idx_1 >= split_from) {
                    int half_pairs_2 = (pair_count_4 + 1) / 2;
                    if ((tile_idx_1 - split_from) % 2 == 0) {
                        pair_count_4 = half_pairs_2;
                    } else {
                        pair_count_4 = pair_count_4 - half_pairs_2;
                        if (pair_count_4 < 1) {
                            pair_count_4 = 1;
                        }
                        index_base_3 = index_base_3 + half_pairs_2 * 2;
                        selected_count_3 = selected_count_3 - half_pairs_2 * 2;
                        if (selected_count_3 < 0) {
                            selected_count_3 = 0;
#endif
                        }
                    }
                }
#if __CUDA_ARCH__ == 1030
                int k_scale_base_1 = (batch_5 * num_kv_heads + kv_head_5) * k_scale_groups;
                int first_pair_1 = (p_group + steps_before_2) % 2;
                int e_entry0 = first_pair_1 * 2;
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
                    unsigned int s_stage = (unsigned int)(tile_iter_3 % 2);
                    int s_off = tile_iter_3 % 2 * 100;
                    float vs4[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(v_scale + (batch_5 * v_scale_batch_stride + kv_head_5 * 128 + p_lane * 4) + 0);
                        vs4[0 + 0] = _v4.x;
                        vs4[0 + 1] = _v4.y;
                        vs4[0 + 2] = _v4.z;
                        vs4[0 + 3] = _v4.w;
                    }
                    int _stage_meta_empty = (int)((unsigned int)tile_iter_3 % 2);
                    int _parity_meta_empty = (int)((unsigned int)tile_iter_3 / 2 % 2) ^ 1;
                    mbarrier_wait(meta_empty_addr + (_stage_meta_empty) * 8, _parity_meta_empty);
                    #pragma unroll
                    for (int vs_i = 0; vs_i < 4; vs_i++) {
                        vs_f[tile_iter_3 % 2 * 128 + p_lane * 4 + vs_i] = vs4[vs_i];
                    }
                    meta_f[s_off + p_lane] = s_q_lo;
                    meta_f[s_off + 32 + p_lane] = s_q_hi;
                    meta_i[s_off + 88] = batch_5;
                    meta_i[s_off + 88 + 1] = kv_head_5;
                    meta_i[s_off + 88 + 2] = k_scale_base_1;
                    meta_i[s_off + 88 + 3] = q_valid_5;
                    meta_i[s_off + 88 + 4] = selected_count_4;
                    meta_i[s_off + 88 + 5] = pair_count_5;
                    meta_i[s_off + 88 + 6] = index_base_4;
                    meta_i[s_off + 88 + 7] = head_4;
                    meta_i[s_off + 88 + 8] = bh_4;
                    meta_i[s_off + 88 + 9] = query_base_4;
                    int piece_1 = tile_idx_2 - split_from;
                    if (piece_1 < 0) {
                        piece_1 = -1;
                    }
                    meta_i[s_off + 88 + 10] = piece_1;
                    mbarrier_arrive(meta_full_addr + ((int)((unsigned int)tile_iter_3 % 2)) * 8);
                }
#else
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1030
                for (int pi = first_pair_1; pi < pair_count_5; pi += 2) {
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
                        w_f = k_scale[k_scale_base_1 + w_group];
                    } else if (p_lane < 10) {
                        w_i = cur_block;
                    } else {
                        if (p_lane < 12) {
                            int valid = seqlen_k - cur_block * 64;
                            if (valid > 64) {
                                valid = 64;
#else
                for (int step_1 = 0; step_1 < pair_count_4; step_1++) {
                    if (q_iter < my_tiles_2) {
                        if (q_step == 0) {
                            int split_d_1_1 = blockIdx.x + q_iter * gridDim.x - split_from;
                            if (split_d_1_1 < 0) {
                                split_d_1_1 = 0;
#endif
                            }
#if __CUDA_ARCH__ == 1030
                            if (valid < 0) {
                                valid = 0;
#else
                            int tile_id_1_1 = blockIdx.x + q_iter * gridDim.x - (split_d_1_1 + 1) / 2;
                            int q_block_1_1 = tile_id_1_1 % q_blocks;
                            int bh_1_1 = tile_id_1_1 / q_blocks;
                            int head_1_1 = bh_1_1 % num_heads;
                            int batch_1_1 = bh_1_1 / num_heads;
                            int kv_head_1_1 = head_1_1 / (num_heads / num_kv_heads);
                            int query_base_1_1 = q_block_1_1 * 64;
                            int q_valid_1_1 = seqlen_q - query_base_1_1;
                            if (q_valid_1_1 > 64) {
                                q_valid_1_1 = 64;
#endif
                            }
#if __CUDA_ARCH__ == 1030
                            if (block_sizes_mode != 0) {
                                int declared = 0;
                                if (block_sizes_mode == 1) {
                                    declared = block_sizes[cur_block];
                                } else if (block_sizes_mode == 2) {
                                    declared = block_sizes[batch_5 * k_blocks + cur_block];
#else
                            if (q_valid_1_1 < 0) {
                                q_valid_1_1 = 0;
                            }
                            int row_slot_1_1 = bh_1_1 * q_blocks + q_block_1_1;
                            int selected_count_1_1 = block_sparse_num;
                            if (has_block_nums != 0) {
                                selected_count_1_1 = block_nums[row_slot_1_1];
                            }
                            if (selected_count_1_1 < 0) {
                                selected_count_1_1 = 0;
                            }
                            if (selected_count_1_1 > capacity) {
                                selected_count_1_1 = capacity;
                            }
                            int pair_count_1_1 = (selected_count_1_1 + 1) / 2;
                            if (pair_count_1_1 < 1) {
                                pair_count_1_1 = 1;
                            }
                            int index_base_1_1 = row_slot_1_1 * capacity;
                            if (blockIdx.x + q_iter * gridDim.x >= split_from) {
                                int half_pairs_3 = (pair_count_1_1 + 1) / 2;
                                if ((blockIdx.x + q_iter * gridDim.x - split_from) % 2 == 0) {
                                    pair_count_1_1 = half_pairs_3;
#endif
                                } else {
#if __CUDA_ARCH__ == 1030
                                    declared = block_sizes[(batch_5 * num_kv_heads + kv_head_5) * k_blocks + cur_block];
                                }
                                if (declared < 0) {
                                    declared = 0;
                                }
                                if (declared < valid) {
                                    valid = declared;
#else
                                    pair_count_1_1 = pair_count_1_1 - half_pairs_3;
                                    if (pair_count_1_1 < 1) {
                                        pair_count_1_1 = 1;
                                    }
                                    index_base_1_1 = index_base_1_1 + half_pairs_3 * 2;
                                    selected_count_1_1 = selected_count_1_1 - half_pairs_3 * 2;
                                    if (selected_count_1_1 < 0) {
                                        selected_count_1_1 = 0;
                                    }
#endif
                                }
                            }
#if __CUDA_ARCH__ == 1030
                            w_i = valid;
#else
                            q_pairs = pair_count_1_1;
                            mbarrier_wait(q_full_addr + (q_stage) * 8, _phase_q_full);
                        }
                        unsigned int k_stage_2 = kv_stage;
                        mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                        mbarrier_wait(s_empty_addr + (sd_e_cur) * 8, _phase_s_empty);
                        sd_e_cur += 1;
                        if (sd_e_cur == 2) { sd_e_cur = 0; _phase_s_empty ^= 1; }
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int mma_a_lo_2 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage) * 512);
                        int mma_b_lo_2 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage_2) * 1024);
                        {
                            uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_2);
                            uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_2);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_scores + ((int)s_cur * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                            }
                        }
                        elect_commit(s_full_addr + (s_cur) * 8);
                        s_cur += 1;
                        if (s_cur == 2) { s_cur = 0; }
                        elect_commit(kv_empty_addr + (kv_stage) * 8);
                        kv_stage += 1;
                        if (kv_stage == 6) { kv_stage = 0; _phase_kv_full ^= 1; }
                        q_step = q_step + 1;
                        if (q_step == q_pairs) {
                            if (my_tiles_2 > q_iter + 1) {
                                elect_commit(q_empty_addr + (q_stage) * 8);
                            }
                            q_stage += 1;
                            if (q_stage == 2) { q_stage = 0; _phase_q_full ^= 1; }
                            q_step = 0;
                            q_iter = q_iter + 1;
#endif
                        }
                    }
#if __CUDA_ARCH__ == 1030
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
#else
                    int first_pv = ((step_1 < 2) ? 1 : 0);
                    unsigned int v_stage = kv_stage;
                    mbarrier_wait(kv_full_addr + (kv_stage) * 8, _phase_kv_full);
                    mbarrier_wait(p_full_addr + (p_cur) * 8, _phase_p_full);
                    unsigned int group_cols = p_cur % 2 * 128 + 1048576;
                    unsigned int p_cols = p_cur * 32 + 1048576;
                    p_cur += 1;
                    if (p_cur == 4) { p_cur = 0; _phase_p_full ^= 1; }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_b_lo_3 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024);
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
                    :: "r"((tmem_output_0 + ((int)group_cols))), "r"(mma_b_lo_3), "r"(tmem_probs_0 + p_cols), "r"(((first_pv) ? 0 : 1)) : "memory");
                    elect_commit(kv_empty_addr + (kv_stage) * 8);
                    kv_stage += 1;
                    if (kv_stage == 6) { kv_stage = 0; _phase_kv_full ^= 1; }
                    elect_commit(pv_done_addr + (pv_cur) * 8);
                    pv_cur += 1;
                    if (pv_cur == 2) { pv_cur = 0; }
                    if (step_1 == pair_count_4 - 1) {
                        elect_commit(o_full_addr);
#endif
                    }
#if __CUDA_ARCH__ == 1030
                    mbarrier_arrive(step_full_addr + (p_n) * 8);
                    cur_block = nxt_block;
                    p_n += 1;
                    if (p_n == (p_group * 4 + 4)) { p_n = p_group * 4; _phase_step_empty ^= 1; }
#else
#endif
                }
#if __CUDA_ARCH__ == 1030
                steps_before_2 = steps_before_2 + pair_count_5;
#else
                mbarrier_wait(tile_done_addr, _phase_tile_done_0);
                _phase_tile_done_0 ^= 1;
#endif
            }
#if __CUDA_ARCH__ == 1030
#else
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
#endif
        }
    }
    // ---- Role: load_warp ----
    if (warp == 13) {
        { // load_warp_main
            unsigned int load_stage = 0;
            unsigned int q_load_stage = 0;
#if __CUDA_ARCH__ == 1030
            int count_4 = (total_tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
            int my_tiles_4 = count_4;
#else
            int count_3 = (total_tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
            int my_tiles_3 = count_3;
#endif
            int k_iter = 0;
            int k_step = 0;
            int k_pairs = 0;
            int k_sel = 0;
            int k_base = 0;
            int k_kv_head = 0;
            int k_batch = 0;
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_kv_empty = 1;
#if __CUDA_ARCH__ == 1030
            if (k_iter < my_tiles_4) {
#else
            if (k_iter < my_tiles_3) {
                if (k_step == 0) {
                    int split_d_3 = blockIdx.x + k_iter * gridDim.x - split_from;
                    if (split_d_3 < 0) {
                        split_d_3 = 0;
                    }
                    int tile_id_3 = blockIdx.x + k_iter * gridDim.x - (split_d_3 + 1) / 2;
                    int q_block_3 = tile_id_3 % q_blocks;
                    int bh_4 = tile_id_3 / q_blocks;
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
                    if (blockIdx.x + k_iter * gridDim.x >= split_from) {
                        int half_pairs_4 = (pair_count_5 + 1) / 2;
                        if ((blockIdx.x + k_iter * gridDim.x - split_from) % 2 == 0) {
                            pair_count_5 = half_pairs_4;
                        } else {
                            pair_count_5 = pair_count_5 - half_pairs_4;
                            if (pair_count_5 < 1) {
                                pair_count_5 = 1;
                            }
                            index_base_4 = index_base_4 + half_pairs_4 * 2;
                            selected_count_4 = selected_count_4 - half_pairs_4 * 2;
                            if (selected_count_4 < 0) {
                                selected_count_4 = 0;
                            }
                        }
                    }
                    k_pairs = pair_count_5;
                    k_sel = selected_count_4;
                    k_base = index_base_4;
                    k_kv_head = kv_head_5;
                    k_batch = batch_5;
                    mbarrier_wait(q_empty_addr + (q_load_stage) * 8, _phase_q_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr + (q_load_stage) * 8, 8192);
                        tma_4d_gmem2smem(q_smem_addr + q_load_stage * 8192, (&q), 0, head_4, query_base_4, batch_5, q_full_addr + (q_load_stage) * 8);
                    }
                    q_load_stage += 1;
                    if (q_load_stage == 2) { q_load_stage = 0; _phase_q_empty ^= 1; }
                }
                int entry0 = k_step * 2;
                int block0 = 0;
                if (entry0 < k_sel) {
                    block0 = block_index[k_base + entry0];
                }
                int block1 = block0;
                if (k_sel > entry0 + 1) {
                    block1 = block_index[k_base + entry0 + 1];
                }
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384, (&k), 0, block0 * 64, k_kv_head, k_batch, kv_full_addr + (load_stage) * 8);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384 + 8192, (&k), 0, block1 * 64, k_kv_head, k_batch, kv_full_addr + (load_stage) * 8);
                }
                load_stage += 1;
                if (load_stage == 6) { load_stage = 0; _phase_kv_empty ^= 1; }
                k_step = k_step + 1;
                if (k_step == k_pairs) {
                    k_step = 0;
                    k_iter = k_iter + 1;
                }
            }
            if (k_iter < my_tiles_3) {
#endif
                if (k_step == 0) {
                    int split_d_4 = blockIdx.x + k_iter * gridDim.x - split_from;
                    if (split_d_4 < 0) {
                        split_d_4 = 0;
                    }
                    int tile_id_4 = blockIdx.x + k_iter * gridDim.x - (split_d_4 + 1) / 2;
                    int q_block_4 = tile_id_4 % q_blocks;
                    int bh_5 = tile_id_4 / q_blocks;
                    int head_5 = bh_5 % num_heads;
                    int batch_6 = bh_5 / num_heads;
                    int kv_head_6 = head_5 / (num_heads / num_kv_heads);
                    int query_base_5 = q_block_4 * 64;
                    int q_valid_6 = seqlen_q - query_base_5;
                    if (q_valid_6 > 64) {
                        q_valid_6 = 64;
                    }
                    if (q_valid_6 < 0) {
                        q_valid_6 = 0;
                    }
                    int row_slot_4 = bh_5 * q_blocks + q_block_4;
                    int selected_count_5 = block_sparse_num;
                    if (has_block_nums != 0) {
                        selected_count_5 = block_nums[row_slot_4];
                    }
                    if (selected_count_5 < 0) {
                        selected_count_5 = 0;
                    }
                    if (selected_count_5 > capacity) {
                        selected_count_5 = capacity;
                    }
                    int pair_count_6 = (selected_count_5 + 1) / 2;
                    if (pair_count_6 < 1) {
                        pair_count_6 = 1;
                    }
                    int index_base_5 = row_slot_4 * capacity;
                    if (blockIdx.x + k_iter * gridDim.x >= split_from) {
                        int half_pairs_5 = (pair_count_6 + 1) / 2;
                        if ((blockIdx.x + k_iter * gridDim.x - split_from) % 2 == 0) {
                            pair_count_6 = half_pairs_5;
                        } else {
                            pair_count_6 = pair_count_6 - half_pairs_5;
                            if (pair_count_6 < 1) {
                                pair_count_6 = 1;
                            }
                            index_base_5 = index_base_5 + half_pairs_5 * 2;
                            selected_count_5 = selected_count_5 - half_pairs_5 * 2;
                            if (selected_count_5 < 0) {
                                selected_count_5 = 0;
                            }
                        }
                    }
                    k_pairs = pair_count_6;
                    k_sel = selected_count_5;
                    k_base = index_base_5;
                    k_kv_head = kv_head_6;
                    k_batch = batch_6;
                    mbarrier_wait(q_empty_addr + (q_load_stage) * 8, _phase_q_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr + (q_load_stage) * 8, 8192);
                        tma_4d_gmem2smem(q_smem_addr + q_load_stage * 8192, (&q), 0, head_5, query_base_5, batch_6, q_full_addr + (q_load_stage) * 8);
                    }
                    q_load_stage += 1;
                    if (q_load_stage == 2) { q_load_stage = 0; _phase_q_empty ^= 1; }
                }
#if __CUDA_ARCH__ == 1030
                int entry0 = k_step * 2;
                int block0 = 0;
                if (entry0 < k_sel) {
                    block0 = block_index[k_base + entry0];
                }
                int block1 = block0;
                if (k_sel > entry0 + 1) {
                    block1 = block_index[k_base + entry0 + 1];
                }
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384, (&k), 0, block0 * 64, k_kv_head, k_batch, kv_full_addr + (load_stage) * 8);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384 + 8192, (&k), 0, block1 * 64, k_kv_head, k_batch, kv_full_addr + (load_stage) * 8);
                }
                load_stage += 1;
                if (load_stage == 8) { load_stage = 0; _phase_kv_empty ^= 1; }
                k_step = k_step + 1;
                if (k_step == k_pairs) {
                    k_step = 0;
                    k_iter = k_iter + 1;
                }
            }
            if (k_iter < my_tiles_4) {
                if (k_step == 0) {
                    int split_d_5 = blockIdx.x + k_iter * gridDim.x - split_from;
                    if (split_d_5 < 0) {
                        split_d_5 = 0;
                    }
                    int tile_id_5 = blockIdx.x + k_iter * gridDim.x - (split_d_5 + 1) / 2;
                    int q_block_5 = tile_id_5 % q_blocks;
                    int bh_6 = tile_id_5 / q_blocks;
                    int head_6 = bh_6 % num_heads;
                    int batch_7 = bh_6 / num_heads;
                    int kv_head_7 = head_6 / (num_heads / num_kv_heads);
                    int query_base_6 = q_block_5 * 64;
                    int q_valid_7 = seqlen_q - query_base_6;
                    if (q_valid_7 > 64) {
                        q_valid_7 = 64;
                    }
                    if (q_valid_7 < 0) {
                        q_valid_7 = 0;
                    }
                    int row_slot_5 = bh_6 * q_blocks + q_block_5;
                    int selected_count_6 = block_sparse_num;
                    if (has_block_nums != 0) {
                        selected_count_6 = block_nums[row_slot_5];
                    }
                    if (selected_count_6 < 0) {
                        selected_count_6 = 0;
                    }
                    if (selected_count_6 > capacity) {
                        selected_count_6 = capacity;
                    }
                    int pair_count_7 = (selected_count_6 + 1) / 2;
                    if (pair_count_7 < 1) {
                        pair_count_7 = 1;
                    }
                    int index_base_6 = row_slot_5 * capacity;
                    if (blockIdx.x + k_iter * gridDim.x >= split_from) {
                        int half_pairs_6 = (pair_count_7 + 1) / 2;
                        if ((blockIdx.x + k_iter * gridDim.x - split_from) % 2 == 0) {
                            pair_count_7 = half_pairs_6;
                        } else {
                            pair_count_7 = pair_count_7 - half_pairs_6;
                            if (pair_count_7 < 1) {
                                pair_count_7 = 1;
                            }
                            index_base_6 = index_base_6 + half_pairs_6 * 2;
                            selected_count_6 = selected_count_6 - half_pairs_6 * 2;
                            if (selected_count_6 < 0) {
                                selected_count_6 = 0;
                            }
                        }
                    }
                    k_pairs = pair_count_7;
                    k_sel = selected_count_6;
                    k_base = index_base_6;
                    k_kv_head = kv_head_7;
                    k_batch = batch_7;
                    mbarrier_wait(q_empty_addr + (q_load_stage) * 8, _phase_q_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr + (q_load_stage) * 8, 8192);
                        tma_4d_gmem2smem(q_smem_addr + q_load_stage * 8192, (&q), 0, head_6, query_base_6, batch_7, q_full_addr + (q_load_stage) * 8);
                    }
                    q_load_stage += 1;
                    if (q_load_stage == 2) { q_load_stage = 0; _phase_q_empty ^= 1; }
                }
#else
#endif
                int entry0_1 = k_step * 2;
                int block0_1 = 0;
                if (entry0_1 < k_sel) {
                    block0_1 = block_index[k_base + entry0_1];
                }
                int block1_1 = block0_1;
                if (k_sel > entry0_1 + 1) {
                    block1_1 = block_index[k_base + entry0_1 + 1];
                }
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384, (&k), 0, block0_1 * 64, k_kv_head, k_batch, kv_full_addr + (load_stage) * 8);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384 + 8192, (&k), 0, block1_1 * 64, k_kv_head, k_batch, kv_full_addr + (load_stage) * 8);
                }
                load_stage += 1;
#if __CUDA_ARCH__ == 1030
                if (load_stage == 8) { load_stage = 0; _phase_kv_empty ^= 1; }
#else
                if (load_stage == 6) { load_stage = 0; _phase_kv_empty ^= 1; }
#endif
                k_step = k_step + 1;
                if (k_step == k_pairs) {
                    k_step = 0;
                    k_iter = k_iter + 1;
                }
            }
#if __CUDA_ARCH__ == 1030
#else
            #pragma unroll 1
            for (int tile_iter_3 = 0; tile_iter_3 < my_tiles_3; tile_iter_3++) {
                int tile_idx_2 = blockIdx.x + tile_iter_3 * gridDim.x;
                int split_d_5 = tile_idx_2 - split_from;
                if (split_d_5 < 0) {
                    split_d_5 = 0;
                }
                int tile_id_5 = tile_idx_2 - (split_d_5 + 1) / 2;
                int q_block_5 = tile_id_5 % q_blocks;
                int bh_6 = tile_id_5 / q_blocks;
                int head_6 = bh_6 % num_heads;
                int batch_7 = bh_6 / num_heads;
                int kv_head_7 = head_6 / (num_heads / num_kv_heads);
                int query_base_6 = q_block_5 * 64;
                int q_valid_7 = seqlen_q - query_base_6;
                if (q_valid_7 > 64) {
                    q_valid_7 = 64;
                }
                if (q_valid_7 < 0) {
                    q_valid_7 = 0;
                }
                int row_slot_5 = bh_6 * q_blocks + q_block_5;
                int selected_count_6 = block_sparse_num;
                if (has_block_nums != 0) {
                    selected_count_6 = block_nums[row_slot_5];
                }
                if (selected_count_6 < 0) {
                    selected_count_6 = 0;
                }
                if (selected_count_6 > capacity) {
                    selected_count_6 = capacity;
                }
                int pair_count_7 = (selected_count_6 + 1) / 2;
                if (pair_count_7 < 1) {
                    pair_count_7 = 1;
                }
                int index_base_6 = row_slot_5 * capacity;
                if (tile_idx_2 >= split_from) {
                    int half_pairs_6 = (pair_count_7 + 1) / 2;
                    if ((tile_idx_2 - split_from) % 2 == 0) {
                        pair_count_7 = half_pairs_6;
                    } else {
                        pair_count_7 = pair_count_7 - half_pairs_6;
                        if (pair_count_7 < 1) {
                            pair_count_7 = 1;
                        }
                        index_base_6 = index_base_6 + half_pairs_6 * 2;
                        selected_count_6 = selected_count_6 - half_pairs_6 * 2;
                        if (selected_count_6 < 0) {
                            selected_count_6 = 0;
                        }
                    }
                }
                int lane_0 = lane;
                int k_scale_base_1 = (batch_7 * num_kv_heads + kv_head_7) * k_scale_groups;
                float acc = 0.0f;
                if (lane_0 < q_valid_7) {
                    acc = q_scale[bh_6 * seqlen_q + query_base_6 + lane_0];
                }
                if (q_valid_7 > lane_0 + 32) {
                    acc = acc + q_scale[bh_6 * seqlen_q + query_base_6 + lane_0 + 32];
                }
                int valid = 0;
                if (lane_0 < 16) {
                    int entry = lane_0 / 4;
                    int block = 0;
                    if (entry < selected_count_6) {
                        block = block_index[index_base_6 + entry];
                    } else if (entry % 2 == 1 && selected_count_6 > entry - 1) {
                        block = block_index[index_base_6 + entry - 1];
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
                                declared = block_sizes[batch_7 * k_blocks + block];
                            } else {
                                declared = block_sizes[(batch_7 * num_kv_heads + kv_head_7) * k_blocks + block];
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
                for (int step_2 = 0; step_2 < pair_count_7; step_2++) {
                    if (k_iter < my_tiles_3) {
                        if (k_step == 0) {
                            int split_d_0 = blockIdx.x + k_iter * gridDim.x - split_from;
                            if (split_d_0 < 0) {
                                split_d_0 = 0;
                            }
                            int tile_id_0 = blockIdx.x + k_iter * gridDim.x - (split_d_0 + 1) / 2;
                            int q_block_0 = tile_id_0 % q_blocks;
                            int bh_0 = tile_id_0 / q_blocks;
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
                            if (blockIdx.x + k_iter * gridDim.x >= split_from) {
                                int half_pairs_7 = (pair_count_0 + 1) / 2;
                                if ((blockIdx.x + k_iter * gridDim.x - split_from) % 2 == 0) {
                                    pair_count_0 = half_pairs_7;
                                } else {
                                    pair_count_0 = pair_count_0 - half_pairs_7;
                                    if (pair_count_0 < 1) {
                                        pair_count_0 = 1;
                                    }
                                    index_base_0 = index_base_0 + half_pairs_7 * 2;
                                    selected_count_0 = selected_count_0 - half_pairs_7 * 2;
                                    if (selected_count_0 < 0) {
                                        selected_count_0 = 0;
                                    }
                                }
                            }
                            k_pairs = pair_count_0;
                            k_sel = selected_count_0;
                            k_base = index_base_0;
                            k_kv_head = kv_head_0;
                            k_batch = batch_0;
                            mbarrier_wait(q_empty_addr + (q_load_stage) * 8, _phase_q_empty);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(q_full_addr + (q_load_stage) * 8, 8192);
                                tma_4d_gmem2smem(q_smem_addr + q_load_stage * 8192, (&q), 0, head_0, query_base_0, batch_0, q_full_addr + (q_load_stage) * 8);
                            }
                            q_load_stage += 1;
                            if (q_load_stage == 2) { q_load_stage = 0; _phase_q_empty ^= 1; }
                        }
                        int entry0_2 = k_step * 2;
                        int block0_2 = 0;
                        if (entry0_2 < k_sel) {
                            block0_2 = block_index[k_base + entry0_2];
                        }
                        int block1_2 = block0_2;
                        if (k_sel > entry0_2 + 1) {
                            block1_2 = block_index[k_base + entry0_2 + 1];
                        }
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384, (&k), 0, block0_2 * 64, k_kv_head, k_batch, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384 + 8192, (&k), 0, block1_2 * 64, k_kv_head, k_batch, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 6) { load_stage = 0; _phase_kv_empty ^= 1; }
                        k_step = k_step + 1;
                        if (k_step == k_pairs) {
                            k_step = 0;
                            k_iter = k_iter + 1;
                        }
                    }
                    int entry0_3 = step_2 * 2;
                    int block0_3 = 0;
                    if (entry0_3 < selected_count_6) {
                        block0_3 = block_index[index_base_6 + entry0_3];
                    }
                    int block1_3 = block0_3;
                    if (selected_count_6 > entry0_3 + 1) {
                        block1_3 = block_index[index_base_6 + entry0_3 + 1];
                    }
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 16384, (&v), 0, block0_3 * 64, kv_head_7, batch_7, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 16384 + 8192, (&v), 0, block1_3 * 64, kv_head_7, batch_7, kv_full_addr + (load_stage) * 8);
                    }
                    load_stage += 1;
                    if (load_stage == 6) { load_stage = 0; _phase_kv_empty ^= 1; }
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
            int steps_before_2 = 0;
            unsigned int _phase_step_empty = 1;
#endif
            #pragma unroll 1
            for (int tile_iter_4 = 0; tile_iter_4 < my_tiles_4; tile_iter_4++) {
                int tile_idx_3 = blockIdx.x + tile_iter_4 * gridDim.x;
                int split_d_6 = tile_idx_3 - split_from;
                if (split_d_6 < 0) {
                    split_d_6 = 0;
                }
                int tile_id_6 = tile_idx_3 - (split_d_6 + 1) / 2;
                int q_block_6 = tile_id_6 % q_blocks;
                int bh_7 = tile_id_6 / q_blocks;
                int head_7 = bh_7 % num_heads;
                int batch_8 = bh_7 / num_heads;
                int kv_head_8 = head_7 / (num_heads / num_kv_heads);
                int query_base_7 = q_block_6 * 64;
                int q_valid_8 = seqlen_q - query_base_7;
                if (q_valid_8 > 64) {
                    q_valid_8 = 64;
                }
                if (q_valid_8 < 0) {
                    q_valid_8 = 0;
                }
                int row_slot_6 = bh_7 * q_blocks + q_block_6;
                int selected_count_7 = block_sparse_num;
                if (has_block_nums != 0) {
                    selected_count_7 = block_nums[row_slot_6];
                }
                if (selected_count_7 < 0) {
                    selected_count_7 = 0;
                }
                if (selected_count_7 > capacity) {
                    selected_count_7 = capacity;
                }
                int pair_count_8 = (selected_count_7 + 1) / 2;
                if (pair_count_8 < 1) {
                    pair_count_8 = 1;
                }
                int index_base_7 = row_slot_6 * capacity;
                if (tile_idx_3 >= split_from) {
#if __CUDA_ARCH__ == 1030
                    int half_pairs_7 = (pair_count_8 + 1) / 2;
#else
                    int half_pairs_8 = (pair_count_8 + 1) / 2;
#endif
                    if ((tile_idx_3 - split_from) % 2 == 0) {
#if __CUDA_ARCH__ == 1030
                        pair_count_8 = half_pairs_7;
#else
                        pair_count_8 = half_pairs_8;
#endif
                    } else {
#if __CUDA_ARCH__ == 1030
                        pair_count_8 = pair_count_8 - half_pairs_7;
#else
                        pair_count_8 = pair_count_8 - half_pairs_8;
#endif
                        if (pair_count_8 < 1) {
                            pair_count_8 = 1;
                        }
#if __CUDA_ARCH__ == 1030
                        index_base_7 = index_base_7 + half_pairs_7 * 2;
                        selected_count_7 = selected_count_7 - half_pairs_7 * 2;
#else
                        index_base_7 = index_base_7 + half_pairs_8 * 2;
                        selected_count_7 = selected_count_7 - half_pairs_8 * 2;
#endif
                        if (selected_count_7 < 0) {
                            selected_count_7 = 0;
                        }
                    }
                }
#if __CUDA_ARCH__ == 1030
                int lane_0 = lane;
#else
#endif
                int k_scale_base_2 = (batch_8 * num_kv_heads + kv_head_8) * k_scale_groups;
#if __CUDA_ARCH__ == 1030
                float acc = 0.0f;
                if (lane_0 < q_valid_8) {
                    acc = q_scale[bh_7 * seqlen_q + query_base_7 + lane_0];
                }
                if (q_valid_8 > lane_0 + 32) {
                    acc = acc + q_scale[bh_7 * seqlen_q + query_base_7 + lane_0 + 32];
                }
                int valid_1 = 0;
                if (lane_0 < 16) {
                    int entry = lane_0 / 4;
                    int block = 0;
                    if (entry < selected_count_7) {
                        block = block_index[index_base_7 + entry];
                    } else if (entry % 2 == 1 && selected_count_7 > entry - 1) {
                        block = block_index[index_base_7 + entry - 1];
                    }
                    int group_idx = block * 4 + lane_0 % 4;
                    if (group_idx > k_scale_groups - 1) {
                        group_idx = k_scale_groups - 1;
                    }
                    if (group_idx < 0) {
                        group_idx = 0;
                    }
                    acc = acc + k_scale[k_scale_base_2 + group_idx];
                    if (lane_0 % 4 == 0) {
                        int valid_1_1 = seqlen_k - block * 64;
                        if (valid_1_1 > 64) {
                            valid_1_1 = 64;
                        }
                        if (valid_1_1 < 0) {
                            valid_1_1 = 0;
                        }
                        if (block_sizes_mode != 0) {
                            int declared_1 = 0;
                            if (block_sizes_mode == 1) {
                                declared_1 = block_sizes[block];
                            } else if (block_sizes_mode == 2) {
                                declared_1 = block_sizes[batch_8 * k_blocks + block];
                            } else {
                                declared_1 = block_sizes[(batch_8 * num_kv_heads + kv_head_8) * k_blocks + block];
                            }
                            if (declared_1 < 0) {
                                declared_1 = 0;
                            }
                            if (declared_1 < valid_1_1) {
                                valid_1_1 = declared_1;
                            }
                        }
                        valid_1 = valid_1_1;
#else
                int first_pair_1 = (p_group + steps_before_2) % 2;
                int e_entry0 = first_pair_1 * 2;
                int e_block = 0;
                if (p_lane < 12) {
                    if (selected_count_7 > e_entry0 + p_blk) {
                        e_block = block_index[index_base_7 + e_entry0 + p_blk];
                    } else if (p_blk == 1 && e_entry0 < selected_count_7) {
                        e_block = block_index[index_base_7 + e_entry0];
#endif
                    }
                }
#if __CUDA_ARCH__ == 1030
#else
                int cur_block = e_block;
                if (p_group == 0) {
                    float s_q_lo = 0.0f;
                    if (p_lane < q_valid_8) {
                        s_q_lo = q_scale[bh_7 * seqlen_q + query_base_7 + p_lane];
                    }
                    float s_q_hi = 0.0f;
                    if (q_valid_8 > p_lane + 32) {
                        s_q_hi = q_scale[bh_7 * seqlen_q + query_base_7 + p_lane + 32];
                    }
                    unsigned int s_stage = (unsigned int)(tile_iter_4 % 2);
                    int s_off = tile_iter_4 % 2 * 100;
                    float vs4[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(v_scale + (batch_8 * v_scale_batch_stride + kv_head_8 * 128 + p_lane * 4) + 0);
                        vs4[0 + 0] = _v4.x;
                        vs4[0 + 1] = _v4.y;
                        vs4[0 + 2] = _v4.z;
                        vs4[0 + 3] = _v4.w;
                    }
                    int _stage_meta_empty = (int)((unsigned int)tile_iter_4 % 2);
                    int _parity_meta_empty = (int)((unsigned int)tile_iter_4 / 2 % 2) ^ 1;
                    mbarrier_wait(meta_empty_addr + (_stage_meta_empty) * 8, _parity_meta_empty);
                    #pragma unroll
                    for (int vs_i = 0; vs_i < 4; vs_i++) {
                        vs_f[tile_iter_4 % 2 * 128 + p_lane * 4 + vs_i] = vs4[vs_i];
                    }
                    meta_f[s_off + p_lane] = s_q_lo;
                    meta_f[s_off + 32 + p_lane] = s_q_hi;
                    meta_i[s_off + 88] = batch_8;
                    meta_i[s_off + 88 + 1] = kv_head_8;
                    meta_i[s_off + 88 + 2] = k_scale_base_2;
                    meta_i[s_off + 88 + 3] = q_valid_8;
                    meta_i[s_off + 88 + 4] = selected_count_7;
                    meta_i[s_off + 88 + 5] = pair_count_8;
                    meta_i[s_off + 88 + 6] = index_base_7;
                    meta_i[s_off + 88 + 7] = head_7;
                    meta_i[s_off + 88 + 8] = bh_7;
                    meta_i[s_off + 88 + 9] = query_base_7;
                    int piece_1 = tile_idx_3 - split_from;
                    if (piece_1 < 0) {
                        piece_1 = -1;
                    }
                    meta_i[s_off + 88 + 10] = piece_1;
                    mbarrier_arrive(meta_full_addr + ((int)((unsigned int)tile_iter_4 % 2)) * 8);
                }
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1030
                for (int step_2 = 0; step_2 < pair_count_8; step_2++) {
                    if (k_iter < my_tiles_4) {
                        if (k_step == 0) {
                            int split_d_0 = blockIdx.x + k_iter * gridDim.x - split_from;
                            if (split_d_0 < 0) {
                                split_d_0 = 0;
                            }
                            int tile_id_0 = blockIdx.x + k_iter * gridDim.x - (split_d_0 + 1) / 2;
                            int q_block_0 = tile_id_0 % q_blocks;
                            int bh_0 = tile_id_0 / q_blocks;
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
                            if (blockIdx.x + k_iter * gridDim.x >= split_from) {
                                int half_pairs_8 = (pair_count_0 + 1) / 2;
                                if ((blockIdx.x + k_iter * gridDim.x - split_from) % 2 == 0) {
                                    pair_count_0 = half_pairs_8;
                                } else {
                                    pair_count_0 = pair_count_0 - half_pairs_8;
                                    if (pair_count_0 < 1) {
                                        pair_count_0 = 1;
                                    }
                                    index_base_0 = index_base_0 + half_pairs_8 * 2;
                                    selected_count_0 = selected_count_0 - half_pairs_8 * 2;
                                    if (selected_count_0 < 0) {
                                        selected_count_0 = 0;
                                    }
                                }
                            }
                            k_pairs = pair_count_0;
                            k_sel = selected_count_0;
                            k_base = index_base_0;
                            k_kv_head = kv_head_0;
                            k_batch = batch_0;
                            mbarrier_wait(q_empty_addr + (q_load_stage) * 8, _phase_q_empty);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(q_full_addr + (q_load_stage) * 8, 8192);
                                tma_4d_gmem2smem(q_smem_addr + q_load_stage * 8192, (&q), 0, head_0, query_base_0, batch_0, q_full_addr + (q_load_stage) * 8);
                            }
                            q_load_stage += 1;
                            if (q_load_stage == 2) { q_load_stage = 0; _phase_q_empty ^= 1; }
                        }
                        int entry0_2 = k_step * 2;
                        int block0_2 = 0;
                        if (entry0_2 < k_sel) {
                            block0_2 = block_index[k_base + entry0_2];
                        }
                        int block1_2 = block0_2;
                        if (k_sel > entry0_2 + 1) {
                            block1_2 = block_index[k_base + entry0_2 + 1];
                        }
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384, (&k), 0, block0_2 * 64, k_kv_head, k_batch, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384 + 8192, (&k), 0, block1_2 * 64, k_kv_head, k_batch, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 8) { load_stage = 0; _phase_kv_empty ^= 1; }
                        k_step = k_step + 1;
                        if (k_step == k_pairs) {
                            k_step = 0;
                            k_iter = k_iter + 1;
#else
                for (int pi = first_pair_1; pi < pair_count_8; pi += 2) {
                    int e_entry0_0 = (pi + 2) * 2;
                    int e_block_0 = 0;
                    if (p_lane < 12) {
                        if (selected_count_7 > e_entry0_0 + p_blk) {
                            e_block_0 = block_index[index_base_7 + e_entry0_0 + p_blk];
                        } else if (p_blk == 1 && e_entry0_0 < selected_count_7) {
                            e_block_0 = block_index[index_base_7 + e_entry0_0];
#endif
                        }
                    }
#if __CUDA_ARCH__ == 1030
                    int entry0_3 = step_2 * 2;
                    int block0_3 = 0;
                    if (entry0_3 < selected_count_7) {
                        block0_3 = block_index[index_base_7 + entry0_3];
#else
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
                                    declared_1 = block_sizes[batch_8 * k_blocks + cur_block];
                                } else {
                                    declared_1 = block_sizes[(batch_8 * num_kv_heads + kv_head_8) * k_blocks + cur_block];
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
#endif
                    }
#if __CUDA_ARCH__ == 1030
                    int block1_3 = block0_3;
                    if (selected_count_7 > entry0_3 + 1) {
                        block1_3 = block_index[index_base_7 + entry0_3 + 1];
#else
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
#endif
                    }
#if __CUDA_ARCH__ == 1030
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 16384, (&v), 0, block0_3 * 64, kv_head_8, batch_8, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(v_smem_addr + load_stage * 16384 + 8192, (&v), 0, block1_3 * 64, kv_head_8, batch_8, kv_full_addr + (load_stage) * 8);
                    }
                    load_stage += 1;
                    if (load_stage == 8) { load_stage = 0; _phase_kv_empty ^= 1; }
#else
                    mbarrier_arrive(step_full_addr + (p_n) * 8);
                    cur_block = nxt_block;
                    p_n += 1;
                    if (p_n == (p_group * 4 + 4)) { p_n = p_group * 4; _phase_step_empty ^= 1; }
#endif
                }
#if __CUDA_ARCH__ == 1030
                pf_sink_f[lane_0] = acc;
                pf_sink_i[lane_0] = valid_1;
#else
                steps_before_2 = steps_before_2 + pair_count_8;
#endif
            }
        }
    }

    // Cleanup
}

} // extern "C"
