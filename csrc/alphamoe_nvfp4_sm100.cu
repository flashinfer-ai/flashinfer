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

// Frozen AlphaMoE NVFP4 up -> SwiGLU -> down schedule.
//
// Loom provenance:
//   repository: Cake
//   commit: e2aa03274e40b03bbba5cdeb4615fa586ca4f369
//   IR: loom/examples/weave/alpha_moe_nvfp4_up_down.py
//   selector: alpha_moe_nvfp4_up_down
//   target used for source generation: sm_100a
//   tensor-map ABI: grid_constant
//   raw generated CUDA SHA-256:
//     c452e6d169cd03574e30f78f1c8dbf75ea154ecbdf510f930b0e6f1b6713cfb0
//
// The generated source is integrated by one mechanical prelude transform:
// its fixed-width aliases, opaque CUtensorMap alias, and Loom tensor-map
// declarations (raw lines 1-12) are replaced by the host headers and the
// layout-identical declarations below. From the generated cuda_bf16 include
// through the closing extern-C brace, the source is byte-identical. The
// TVM-FFI validation, TMA encoding, and launch binding follow that frozen body.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "tvm_ffi_utils.h"

struct __align__(128) LoomTensorMap {
  uint64_t opaque[16];
};

template <int N>
struct __align__(128) LoomTensorMapPack {
  LoomTensorMap maps[N];
};

// clang-format off
#include <cuda_bf16.h>

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 104
#define TMEM_UP_ACC_OFFSET 0
#define TMEM_DOWN_ACC_OFFSET 16
#define TMEM_UP_GATE_SF_OFFSET 40
#define TMEM_UP_UP_SF_OFFSET 56
#define TMEM_UP_X_SF_OFFSET 72
#define TMEM_DOWN_W2_SF_OFFSET 88
#define TMEM_DOWN_ACT_SF_OFFSET 96
#define NUM_UP_PIPE_STAGES 3
#define NUM_DOWN_PIPE_STAGES 3
#define NUM_SINGLE_PIPE_STAGES 1
#define SMEM_SMEM_W1_OFF 1024
#define SMEM_SMEM_W1_STAGE_BYTES 32768
#define SMEM_SMEM_W1_STRIDE 39936
#define SMEM_SMEM_X_OFF 33792
#define SMEM_SMEM_X_STAGE_BYTES 1024
#define SMEM_SMEM_X_STRIDE 39936
#define SMEM_SMEM_W1_GATE_SF_OFF 34816
#define SMEM_SMEM_W1_GATE_SF_STAGE_BYTES 2048
#define SMEM_SMEM_W1_GATE_SF_STRIDE 39936
#define SMEM_SMEM_W1_UP_SF_OFF 36864
#define SMEM_SMEM_W1_UP_SF_STAGE_BYTES 2048
#define SMEM_SMEM_W1_UP_SF_STRIDE 39936
#define SMEM_SMEM_X_SF_OFF 38912
#define SMEM_SMEM_X_SF_STAGE_BYTES 2048
#define SMEM_SMEM_X_SF_STRIDE 39936
#define SMEM_SMEM_W2_OFF 120832
#define SMEM_SMEM_W2_STAGE_BYTES 8192
#define SMEM_SMEM_W2_STRIDE 9216
#define SMEM_SMEM_W2_SF_OFF 129024
#define SMEM_SMEM_W2_SF_STAGE_BYTES 1024
#define SMEM_SMEM_W2_SF_STRIDE 9216
#define SMEM_SMEM_ACT_OFF 148480
#define SMEM_SMEM_ACT_STAGE_BYTES 512
#define SMEM_SMEM_ACT_STRIDE 512
#define SMEM_SMEM_ACT_SCALE_OFF 148992
#define SMEM_SMEM_ACT_SCALE_STAGE_BYTES 256
#define SMEM_SMEM_ACT_SCALE_STRIDE 256
#define SMEM_SMEM_ACT_SF_CP_OFF 149248
#define SMEM_SMEM_ACT_SF_CP_STAGE_BYTES 1024
#define SMEM_SMEM_ACT_SF_CP_STRIDE 1024
#define SMEM_SMEM_OUT_OFF 150272
#define SMEM_SMEM_OUT_STAGE_BYTES 2048
#define SMEM_SMEM_OUT_STRIDE 2048
#define SMEM_TOTAL 152320
#define THREADS 192

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
        :: "r"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ void mma_ss_step(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::mxf4nvf4 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
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


__device__ __forceinline__ void mbarrier_init_pred(int mbar_addr, uint32_t count, uint32_t pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
        "}\n" :: "r"(mbar_addr), "r"(count), "r"(pred));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo256(int addr) {
    const int SBO = 256;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo512(int addr) {
    const int SBO = 512;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
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
    // Canonical .shared::cta form for non-multicast gather4, matching
    // trtllm-gen / cuda_ptx and the PTX ISA qualifier order
    // (dim.dst.src.load_mode.completion_mechanism). Per the PTX grammar,
    // .shared::cluster is reserved for the multicast variant (ctaMask).
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_ld_x8_wait(float* dst, int addr) {
    tmem_ld_x8(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_alpha_moe_nvfp4_up_down(uint8_t* __restrict__ x_scale, uint8_t* __restrict__ w1_scale, uint8_t* __restrict__ w2_scale, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, float* __restrict__ topk_weights, __nv_bfloat16* __restrict__ out, int M, int K, int top_k, int route_block_m, float scaling_factor, __grid_constant__ LoomTensorMapPack<3> const _loom_tma_params)
{
    uint64_t _loom_tma_param_base;
    asm volatile("mov.b64 %0, %1;" : "=l"(_loom_tma_param_base) : "l"((uint64_t)(&_loom_tma_params)));

    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* smem_w1 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_w1_addr = smem + 1024;
    uint8_t* smem_x = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_x_addr = smem + 33792;
    uint8_t* smem_w1_gate_sf = reinterpret_cast<uint8_t*>(smem_raw + 34816);
    const int smem_w1_gate_sf_addr = smem + 34816;
    uint8_t* smem_w1_up_sf = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int smem_w1_up_sf_addr = smem + 36864;
    uint8_t* smem_x_sf = reinterpret_cast<uint8_t*>(smem_raw + 38912);
    const int smem_x_sf_addr = smem + 38912;
    uint8_t* smem_w2 = reinterpret_cast<uint8_t*>(smem_raw + 120832);
    const int smem_w2_addr = smem + 120832;
    uint8_t* smem_w2_sf = reinterpret_cast<uint8_t*>(smem_raw + 129024);
    const int smem_w2_sf_addr = smem + 129024;
    uint8_t* smem_act = reinterpret_cast<uint8_t*>(smem_raw + 148480);
    const int smem_act_addr = smem + 148480;
    float* smem_act_scale = reinterpret_cast<float*>(smem_raw + 148992);
    const int smem_act_scale_addr = smem + 148992;
    uint8_t* smem_act_sf_cp = reinterpret_cast<uint8_t*>(smem_raw + 149248);
    const int smem_act_sf_cp_addr = smem + 149248;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 150272);
    const int smem_out_addr = smem + 150272;

    // Mbarrier init (6 groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // --- pipeline 'up_pipe' ---
        // up_full: 3 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        mbarrier_init_pred(smem + 8, 1, leader);
        mbarrier_init_pred(smem + 16, 1, leader);
        // up_free: 3 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        mbarrier_init_pred(smem + 32, 1, leader);
        mbarrier_init_pred(smem + 40, 1, leader);
        // --- pipeline 'single_pipe' ---
        // up_ready: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 48, 1, leader);
        // --- pipeline 'down_pipe' ---
        // down_full: 3 barriers, init_count=1
        mbarrier_init_pred(smem + 56, 1, leader);
        mbarrier_init_pred(smem + 64, 1, leader);
        mbarrier_init_pred(smem + 72, 1, leader);
        // down_ready: 3 barriers, init_count=1
        mbarrier_init_pred(smem + 80, 1, leader);
        mbarrier_init_pred(smem + 88, 1, leader);
        mbarrier_init_pred(smem + 96, 1, leader);
        // down_free: 3 barriers, init_count=4
        mbarrier_init_pred(smem + 104, 4, leader);
        mbarrier_init_pred(smem + 112, 4, leader);
        mbarrier_init_pred(smem + 120, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (128 columns, 104 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
    if (warp == 0) {
        int _tmem_hold = smem + 128;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define up_full_addr (mbar_base + 0)
    #define up_free_addr (mbar_base + 24)
    #define up_ready_addr (mbar_base + 48)
    #define down_full_addr (mbar_base + 56)
    #define down_ready_addr (mbar_base + 80)
    #define down_free_addr (mbar_base + 104)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_up_acc = taddr;
    const int tmem_down_acc = taddr + 16;
    const int tmem_up_gate_sf = taddr + 40;
    const int tmem_up_up_sf = taddr + 56;
    const int tmem_up_x_sf = taddr + 72;
    const int tmem_down_w2_sf = taddr + 88;
    const int tmem_down_act_sf = taddr + 96;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int route_work = blockIdx.x;
            int intermediate_block = blockIdx.y;
            int intermediate_blocks = gridDim.y;
            int subtiles_per_route_block = route_block_m / 8;
            int route_block = route_work / subtiles_per_route_block;
            int route_subtile = route_work % subtiles_per_route_block;
            int route_base = route_block * route_block_m + route_subtile * 8;
            bool route_active = route_base < num_tokens_post_padded[0];
            int expert = 0;
            if (route_active) {
                expert = expert_ids[route_block];
            }
            int sf_cols = K / 16;
            unsigned int up_stage = 0;
            unsigned int _phase_up_free = 1;
            #pragma unroll 1
            for (int kb = 0; kb < K / 256; kb++) {
                mbarrier_wait(up_free_addr + (up_stage) * 8, _phase_up_free);
                int gate_sf_base = smem_w1_gate_sf_addr + up_stage * 39936;
                int up_sf_base = smem_w1_up_sf_addr + up_stage * 39936;
                int x_sf_base = smem_x_sf_addr + up_stage * 39936;
                int sf_row = lane;
                int sf_c = lane / 8;
                int sf_d = lane % 8;
                int sf_g = 0;
                int sf_dst = (sf_c * 4 * 8 + sf_d) * 16 + sf_g * 4;
                int gate_idx = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row) * sf_cols + kb * 16;
                int up_idx = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row) * sf_cols + kb * 16;
                unsigned int gate0 = w1_scale[gate_idx];
                unsigned int gate1 = w1_scale[gate_idx + 1];
                unsigned int gate2 = w1_scale[gate_idx + 2];
                unsigned int gate3 = w1_scale[gate_idx + 3];
                unsigned int up0 = w1_scale[up_idx];
                unsigned int up1 = w1_scale[up_idx + 1];
                unsigned int up2 = w1_scale[up_idx + 2];
                unsigned int up3 = w1_scale[up_idx + 3];
                unsigned int gate_word = gate0 | gate1 << 8 | gate2 << 16 | gate3 << 24;
                unsigned int up_word = up0 | up1 << 8 | up2 << 16 | up3 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst), "r"(gate_word));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst), "r"(up_word));
                unsigned int x0 = 0;
                unsigned int x1 = 0;
                unsigned int x2 = 0;
                unsigned int x3 = 0;
                if (sf_row < 8) {
                    int sf_pair = M * top_k;
                    if (route_active) {
                        sf_pair = sorted_token_ids[route_base + sf_row];
                    }
                    int _min_0 = ((sf_pair / top_k) < (M - 1) ? (sf_pair / top_k) : (M - 1));
                    int sf_token = _min_0;
                    int x_idx = sf_token * sf_cols + kb * 16;
                    x0 = x_scale[x_idx];
                    x1 = x_scale[x_idx + 1];
                    x2 = x_scale[x_idx + 2];
                    x3 = x_scale[x_idx + 3];
                }
                unsigned int x_word = x0 | x1 << 8 | x2 << 16 | x3 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst), "r"(x_word));
                int sf_dst_0 = ((sf_c * 4 + 1) * 8 + sf_d) * 16 + sf_g * 4;
                int gate_idx_1 = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row) * sf_cols + kb * 16 + 4;
                int up_idx_2 = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row) * sf_cols + kb * 16 + 4;
                unsigned int gate0_3 = w1_scale[gate_idx_1];
                unsigned int gate1_4 = w1_scale[gate_idx_1 + 1];
                unsigned int gate2_5 = w1_scale[gate_idx_1 + 2];
                unsigned int gate3_6 = w1_scale[gate_idx_1 + 3];
                unsigned int up0_7 = w1_scale[up_idx_2];
                unsigned int up1_8 = w1_scale[up_idx_2 + 1];
                unsigned int up2_9 = w1_scale[up_idx_2 + 2];
                unsigned int up3_10 = w1_scale[up_idx_2 + 3];
                unsigned int gate_word_11 = gate0_3 | gate1_4 << 8 | gate2_5 << 16 | gate3_6 << 24;
                unsigned int up_word_12 = up0_7 | up1_8 << 8 | up2_9 << 16 | up3_10 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst_0), "r"(gate_word_11));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst_0), "r"(up_word_12));
                unsigned int x0_13 = 0;
                unsigned int x1_14 = 0;
                unsigned int x2_15 = 0;
                unsigned int x3_16 = 0;
                if (sf_row < 8) {
                    int sf_pair_1 = M * top_k;
                    if (route_active) {
                        sf_pair_1 = sorted_token_ids[route_base + sf_row];
                    }
                    int _min_1 = ((sf_pair_1 / top_k) < (M - 1) ? (sf_pair_1 / top_k) : (M - 1));
                    int sf_token_1 = _min_1;
                    int x_idx_1 = sf_token_1 * sf_cols + kb * 16 + 4;
                    x0_13 = x_scale[x_idx_1];
                    x1_14 = x_scale[x_idx_1 + 1];
                    x2_15 = x_scale[x_idx_1 + 2];
                    x3_16 = x_scale[x_idx_1 + 3];
                }
                unsigned int x_word_17 = x0_13 | x1_14 << 8 | x2_15 << 16 | x3_16 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_0), "r"(x_word_17));
                int sf_dst_18 = ((sf_c * 4 + 2) * 8 + sf_d) * 16 + sf_g * 4;
                int gate_idx_19 = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row) * sf_cols + kb * 16 + 8;
                int up_idx_20 = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row) * sf_cols + kb * 16 + 8;
                unsigned int gate0_21 = w1_scale[gate_idx_19];
                unsigned int gate1_22 = w1_scale[gate_idx_19 + 1];
                unsigned int gate2_23 = w1_scale[gate_idx_19 + 2];
                unsigned int gate3_24 = w1_scale[gate_idx_19 + 3];
                unsigned int up0_25 = w1_scale[up_idx_20];
                unsigned int up1_26 = w1_scale[up_idx_20 + 1];
                unsigned int up2_27 = w1_scale[up_idx_20 + 2];
                unsigned int up3_28 = w1_scale[up_idx_20 + 3];
                unsigned int gate_word_29 = gate0_21 | gate1_22 << 8 | gate2_23 << 16 | gate3_24 << 24;
                unsigned int up_word_30 = up0_25 | up1_26 << 8 | up2_27 << 16 | up3_28 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst_18), "r"(gate_word_29));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst_18), "r"(up_word_30));
                unsigned int x0_31 = 0;
                unsigned int x1_32 = 0;
                unsigned int x2_33 = 0;
                unsigned int x3_34 = 0;
                if (sf_row < 8) {
                    int sf_pair_2 = M * top_k;
                    if (route_active) {
                        sf_pair_2 = sorted_token_ids[route_base + sf_row];
                    }
                    int _min_2 = ((sf_pair_2 / top_k) < (M - 1) ? (sf_pair_2 / top_k) : (M - 1));
                    int sf_token_2 = _min_2;
                    int x_idx_2 = sf_token_2 * sf_cols + kb * 16 + 8;
                    x0_31 = x_scale[x_idx_2];
                    x1_32 = x_scale[x_idx_2 + 1];
                    x2_33 = x_scale[x_idx_2 + 2];
                    x3_34 = x_scale[x_idx_2 + 3];
                }
                unsigned int x_word_35 = x0_31 | x1_32 << 8 | x2_33 << 16 | x3_34 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_18), "r"(x_word_35));
                int sf_dst_36 = ((sf_c * 4 + 3) * 8 + sf_d) * 16 + sf_g * 4;
                int gate_idx_37 = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row) * sf_cols + kb * 16 + 12;
                int up_idx_38 = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row) * sf_cols + kb * 16 + 12;
                unsigned int gate0_39 = w1_scale[gate_idx_37];
                unsigned int gate1_40 = w1_scale[gate_idx_37 + 1];
                unsigned int gate2_41 = w1_scale[gate_idx_37 + 2];
                unsigned int gate3_42 = w1_scale[gate_idx_37 + 3];
                unsigned int up0_43 = w1_scale[up_idx_38];
                unsigned int up1_44 = w1_scale[up_idx_38 + 1];
                unsigned int up2_45 = w1_scale[up_idx_38 + 2];
                unsigned int up3_46 = w1_scale[up_idx_38 + 3];
                unsigned int gate_word_47 = gate0_39 | gate1_40 << 8 | gate2_41 << 16 | gate3_42 << 24;
                unsigned int up_word_48 = up0_43 | up1_44 << 8 | up2_45 << 16 | up3_46 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst_36), "r"(gate_word_47));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst_36), "r"(up_word_48));
                unsigned int x0_49 = 0;
                unsigned int x1_50 = 0;
                unsigned int x2_51 = 0;
                unsigned int x3_52 = 0;
                if (sf_row < 8) {
                    int sf_pair_3 = M * top_k;
                    if (route_active) {
                        sf_pair_3 = sorted_token_ids[route_base + sf_row];
                    }
                    int _min_3 = ((sf_pair_3 / top_k) < (M - 1) ? (sf_pair_3 / top_k) : (M - 1));
                    int sf_token_3 = _min_3;
                    int x_idx_3 = sf_token_3 * sf_cols + kb * 16 + 12;
                    x0_49 = x_scale[x_idx_3];
                    x1_50 = x_scale[x_idx_3 + 1];
                    x2_51 = x_scale[x_idx_3 + 2];
                    x3_52 = x_scale[x_idx_3 + 3];
                }
                unsigned int x_word_53 = x0_49 | x1_50 << 8 | x2_51 << 16 | x3_52 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_36), "r"(x_word_53));
                int sf_row_54 = 32 + lane;
                int sf_c_55 = lane / 8;
                int sf_d_56 = lane % 8;
                int sf_g_57 = 1;
                int sf_dst_58 = (sf_c_55 * 4 * 8 + sf_d_56) * 16 + sf_g_57 * 4;
                int gate_idx_59 = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row_54) * sf_cols + kb * 16;
                int up_idx_60 = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row_54) * sf_cols + kb * 16;
                unsigned int gate0_61 = w1_scale[gate_idx_59];
                unsigned int gate1_62 = w1_scale[gate_idx_59 + 1];
                unsigned int gate2_63 = w1_scale[gate_idx_59 + 2];
                unsigned int gate3_64 = w1_scale[gate_idx_59 + 3];
                unsigned int up0_65 = w1_scale[up_idx_60];
                unsigned int up1_66 = w1_scale[up_idx_60 + 1];
                unsigned int up2_67 = w1_scale[up_idx_60 + 2];
                unsigned int up3_68 = w1_scale[up_idx_60 + 3];
                unsigned int gate_word_69 = gate0_61 | gate1_62 << 8 | gate2_63 << 16 | gate3_64 << 24;
                unsigned int up_word_70 = up0_65 | up1_66 << 8 | up2_67 << 16 | up3_68 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst_58), "r"(gate_word_69));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst_58), "r"(up_word_70));
                unsigned int x0_71 = 0;
                unsigned int x1_72 = 0;
                unsigned int x2_73 = 0;
                unsigned int x3_74 = 0;
                if (sf_row_54 < 8) {
                    int sf_pair_4 = M * top_k;
                    if (route_active) {
                        sf_pair_4 = sorted_token_ids[route_base + sf_row_54];
                    }
                    int _min_4 = ((sf_pair_4 / top_k) < (M - 1) ? (sf_pair_4 / top_k) : (M - 1));
                    int sf_token_4 = _min_4;
                    int x_idx_4 = sf_token_4 * sf_cols + kb * 16;
                    x0_71 = x_scale[x_idx_4];
                    x1_72 = x_scale[x_idx_4 + 1];
                    x2_73 = x_scale[x_idx_4 + 2];
                    x3_74 = x_scale[x_idx_4 + 3];
                }
                unsigned int x_word_75 = x0_71 | x1_72 << 8 | x2_73 << 16 | x3_74 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_58), "r"(x_word_75));
                int sf_dst_76 = ((sf_c_55 * 4 + 1) * 8 + sf_d_56) * 16 + sf_g_57 * 4;
                int gate_idx_77 = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row_54) * sf_cols + kb * 16 + 4;
                int up_idx_78 = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row_54) * sf_cols + kb * 16 + 4;
                unsigned int gate0_79 = w1_scale[gate_idx_77];
                unsigned int gate1_80 = w1_scale[gate_idx_77 + 1];
                unsigned int gate2_81 = w1_scale[gate_idx_77 + 2];
                unsigned int gate3_82 = w1_scale[gate_idx_77 + 3];
                unsigned int up0_83 = w1_scale[up_idx_78];
                unsigned int up1_84 = w1_scale[up_idx_78 + 1];
                unsigned int up2_85 = w1_scale[up_idx_78 + 2];
                unsigned int up3_86 = w1_scale[up_idx_78 + 3];
                unsigned int gate_word_87 = gate0_79 | gate1_80 << 8 | gate2_81 << 16 | gate3_82 << 24;
                unsigned int up_word_88 = up0_83 | up1_84 << 8 | up2_85 << 16 | up3_86 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst_76), "r"(gate_word_87));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst_76), "r"(up_word_88));
                unsigned int x0_89 = 0;
                unsigned int x1_90 = 0;
                unsigned int x2_91 = 0;
                unsigned int x3_92 = 0;
                if (sf_row_54 < 8) {
                    int sf_pair_5 = M * top_k;
                    if (route_active) {
                        sf_pair_5 = sorted_token_ids[route_base + sf_row_54];
                    }
                    int _min_5 = ((sf_pair_5 / top_k) < (M - 1) ? (sf_pair_5 / top_k) : (M - 1));
                    int sf_token_5 = _min_5;
                    int x_idx_5 = sf_token_5 * sf_cols + kb * 16 + 4;
                    x0_89 = x_scale[x_idx_5];
                    x1_90 = x_scale[x_idx_5 + 1];
                    x2_91 = x_scale[x_idx_5 + 2];
                    x3_92 = x_scale[x_idx_5 + 3];
                }
                unsigned int x_word_93 = x0_89 | x1_90 << 8 | x2_91 << 16 | x3_92 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_76), "r"(x_word_93));
                int sf_dst_94 = ((sf_c_55 * 4 + 2) * 8 + sf_d_56) * 16 + sf_g_57 * 4;
                int gate_idx_95 = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row_54) * sf_cols + kb * 16 + 8;
                int up_idx_96 = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row_54) * sf_cols + kb * 16 + 8;
                unsigned int gate0_97 = w1_scale[gate_idx_95];
                unsigned int gate1_98 = w1_scale[gate_idx_95 + 1];
                unsigned int gate2_99 = w1_scale[gate_idx_95 + 2];
                unsigned int gate3_100 = w1_scale[gate_idx_95 + 3];
                unsigned int up0_101 = w1_scale[up_idx_96];
                unsigned int up1_102 = w1_scale[up_idx_96 + 1];
                unsigned int up2_103 = w1_scale[up_idx_96 + 2];
                unsigned int up3_104 = w1_scale[up_idx_96 + 3];
                unsigned int gate_word_105 = gate0_97 | gate1_98 << 8 | gate2_99 << 16 | gate3_100 << 24;
                unsigned int up_word_106 = up0_101 | up1_102 << 8 | up2_103 << 16 | up3_104 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst_94), "r"(gate_word_105));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst_94), "r"(up_word_106));
                unsigned int x0_107 = 0;
                unsigned int x1_108 = 0;
                unsigned int x2_109 = 0;
                unsigned int x3_110 = 0;
                if (sf_row_54 < 8) {
                    int sf_pair_6 = M * top_k;
                    if (route_active) {
                        sf_pair_6 = sorted_token_ids[route_base + sf_row_54];
                    }
                    int _min_6 = ((sf_pair_6 / top_k) < (M - 1) ? (sf_pair_6 / top_k) : (M - 1));
                    int sf_token_6 = _min_6;
                    int x_idx_6 = sf_token_6 * sf_cols + kb * 16 + 8;
                    x0_107 = x_scale[x_idx_6];
                    x1_108 = x_scale[x_idx_6 + 1];
                    x2_109 = x_scale[x_idx_6 + 2];
                    x3_110 = x_scale[x_idx_6 + 3];
                }
                unsigned int x_word_111 = x0_107 | x1_108 << 8 | x2_109 << 16 | x3_110 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_94), "r"(x_word_111));
                int sf_dst_112 = ((sf_c_55 * 4 + 3) * 8 + sf_d_56) * 16 + sf_g_57 * 4;
                int gate_idx_113 = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row_54) * sf_cols + kb * 16 + 12;
                int up_idx_114 = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row_54) * sf_cols + kb * 16 + 12;
                unsigned int gate0_115 = w1_scale[gate_idx_113];
                unsigned int gate1_116 = w1_scale[gate_idx_113 + 1];
                unsigned int gate2_117 = w1_scale[gate_idx_113 + 2];
                unsigned int gate3_118 = w1_scale[gate_idx_113 + 3];
                unsigned int up0_119 = w1_scale[up_idx_114];
                unsigned int up1_120 = w1_scale[up_idx_114 + 1];
                unsigned int up2_121 = w1_scale[up_idx_114 + 2];
                unsigned int up3_122 = w1_scale[up_idx_114 + 3];
                unsigned int gate_word_123 = gate0_115 | gate1_116 << 8 | gate2_117 << 16 | gate3_118 << 24;
                unsigned int up_word_124 = up0_119 | up1_120 << 8 | up2_121 << 16 | up3_122 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst_112), "r"(gate_word_123));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst_112), "r"(up_word_124));
                unsigned int x0_125 = 0;
                unsigned int x1_126 = 0;
                unsigned int x2_127 = 0;
                unsigned int x3_128 = 0;
                if (sf_row_54 < 8) {
                    int sf_pair_7 = M * top_k;
                    if (route_active) {
                        sf_pair_7 = sorted_token_ids[route_base + sf_row_54];
                    }
                    int _min_7 = ((sf_pair_7 / top_k) < (M - 1) ? (sf_pair_7 / top_k) : (M - 1));
                    int sf_token_7 = _min_7;
                    int x_idx_7 = sf_token_7 * sf_cols + kb * 16 + 12;
                    x0_125 = x_scale[x_idx_7];
                    x1_126 = x_scale[x_idx_7 + 1];
                    x2_127 = x_scale[x_idx_7 + 2];
                    x3_128 = x_scale[x_idx_7 + 3];
                }
                unsigned int x_word_129 = x0_125 | x1_126 << 8 | x2_127 << 16 | x3_128 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_112), "r"(x_word_129));
                int sf_row_130 = 64 + lane;
                int sf_c_131 = lane / 8;
                int sf_d_132 = lane % 8;
                int sf_g_133 = 2;
                int sf_dst_134 = (sf_c_131 * 4 * 8 + sf_d_132) * 16 + sf_g_133 * 4;
                int gate_idx_135 = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row_130) * sf_cols + kb * 16;
                int up_idx_136 = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row_130) * sf_cols + kb * 16;
                unsigned int gate0_137 = w1_scale[gate_idx_135];
                unsigned int gate1_138 = w1_scale[gate_idx_135 + 1];
                unsigned int gate2_139 = w1_scale[gate_idx_135 + 2];
                unsigned int gate3_140 = w1_scale[gate_idx_135 + 3];
                unsigned int up0_141 = w1_scale[up_idx_136];
                unsigned int up1_142 = w1_scale[up_idx_136 + 1];
                unsigned int up2_143 = w1_scale[up_idx_136 + 2];
                unsigned int up3_144 = w1_scale[up_idx_136 + 3];
                unsigned int gate_word_145 = gate0_137 | gate1_138 << 8 | gate2_139 << 16 | gate3_140 << 24;
                unsigned int up_word_146 = up0_141 | up1_142 << 8 | up2_143 << 16 | up3_144 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst_134), "r"(gate_word_145));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst_134), "r"(up_word_146));
                unsigned int x0_147 = 0;
                unsigned int x1_148 = 0;
                unsigned int x2_149 = 0;
                unsigned int x3_150 = 0;
                if (sf_row_130 < 8) {
                    int sf_pair_8 = M * top_k;
                    if (route_active) {
                        sf_pair_8 = sorted_token_ids[route_base + sf_row_130];
                    }
                    int _min_8 = ((sf_pair_8 / top_k) < (M - 1) ? (sf_pair_8 / top_k) : (M - 1));
                    int sf_token_8 = _min_8;
                    int x_idx_8 = sf_token_8 * sf_cols + kb * 16;
                    x0_147 = x_scale[x_idx_8];
                    x1_148 = x_scale[x_idx_8 + 1];
                    x2_149 = x_scale[x_idx_8 + 2];
                    x3_150 = x_scale[x_idx_8 + 3];
                }
                unsigned int x_word_151 = x0_147 | x1_148 << 8 | x2_149 << 16 | x3_150 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_134), "r"(x_word_151));
                int sf_dst_152 = ((sf_c_131 * 4 + 1) * 8 + sf_d_132) * 16 + sf_g_133 * 4;
                int gate_idx_153 = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row_130) * sf_cols + kb * 16 + 4;
                int up_idx_154 = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row_130) * sf_cols + kb * 16 + 4;
                unsigned int gate0_155 = w1_scale[gate_idx_153];
                unsigned int gate1_156 = w1_scale[gate_idx_153 + 1];
                unsigned int gate2_157 = w1_scale[gate_idx_153 + 2];
                unsigned int gate3_158 = w1_scale[gate_idx_153 + 3];
                unsigned int up0_159 = w1_scale[up_idx_154];
                unsigned int up1_160 = w1_scale[up_idx_154 + 1];
                unsigned int up2_161 = w1_scale[up_idx_154 + 2];
                unsigned int up3_162 = w1_scale[up_idx_154 + 3];
                unsigned int gate_word_163 = gate0_155 | gate1_156 << 8 | gate2_157 << 16 | gate3_158 << 24;
                unsigned int up_word_164 = up0_159 | up1_160 << 8 | up2_161 << 16 | up3_162 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst_152), "r"(gate_word_163));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst_152), "r"(up_word_164));
                unsigned int x0_165 = 0;
                unsigned int x1_166 = 0;
                unsigned int x2_167 = 0;
                unsigned int x3_168 = 0;
                if (sf_row_130 < 8) {
                    int sf_pair_9 = M * top_k;
                    if (route_active) {
                        sf_pair_9 = sorted_token_ids[route_base + sf_row_130];
                    }
                    int _min_9 = ((sf_pair_9 / top_k) < (M - 1) ? (sf_pair_9 / top_k) : (M - 1));
                    int sf_token_9 = _min_9;
                    int x_idx_9 = sf_token_9 * sf_cols + kb * 16 + 4;
                    x0_165 = x_scale[x_idx_9];
                    x1_166 = x_scale[x_idx_9 + 1];
                    x2_167 = x_scale[x_idx_9 + 2];
                    x3_168 = x_scale[x_idx_9 + 3];
                }
                unsigned int x_word_169 = x0_165 | x1_166 << 8 | x2_167 << 16 | x3_168 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_152), "r"(x_word_169));
                int sf_dst_170 = ((sf_c_131 * 4 + 2) * 8 + sf_d_132) * 16 + sf_g_133 * 4;
                int gate_idx_171 = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row_130) * sf_cols + kb * 16 + 8;
                int up_idx_172 = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row_130) * sf_cols + kb * 16 + 8;
                unsigned int gate0_173 = w1_scale[gate_idx_171];
                unsigned int gate1_174 = w1_scale[gate_idx_171 + 1];
                unsigned int gate2_175 = w1_scale[gate_idx_171 + 2];
                unsigned int gate3_176 = w1_scale[gate_idx_171 + 3];
                unsigned int up0_177 = w1_scale[up_idx_172];
                unsigned int up1_178 = w1_scale[up_idx_172 + 1];
                unsigned int up2_179 = w1_scale[up_idx_172 + 2];
                unsigned int up3_180 = w1_scale[up_idx_172 + 3];
                unsigned int gate_word_181 = gate0_173 | gate1_174 << 8 | gate2_175 << 16 | gate3_176 << 24;
                unsigned int up_word_182 = up0_177 | up1_178 << 8 | up2_179 << 16 | up3_180 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst_170), "r"(gate_word_181));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst_170), "r"(up_word_182));
                unsigned int x0_183 = 0;
                unsigned int x1_184 = 0;
                unsigned int x2_185 = 0;
                unsigned int x3_186 = 0;
                if (sf_row_130 < 8) {
                    int sf_pair_10 = M * top_k;
                    if (route_active) {
                        sf_pair_10 = sorted_token_ids[route_base + sf_row_130];
                    }
                    int _min_10 = ((sf_pair_10 / top_k) < (M - 1) ? (sf_pair_10 / top_k) : (M - 1));
                    int sf_token_10 = _min_10;
                    int x_idx_10 = sf_token_10 * sf_cols + kb * 16 + 8;
                    x0_183 = x_scale[x_idx_10];
                    x1_184 = x_scale[x_idx_10 + 1];
                    x2_185 = x_scale[x_idx_10 + 2];
                    x3_186 = x_scale[x_idx_10 + 3];
                }
                unsigned int x_word_187 = x0_183 | x1_184 << 8 | x2_185 << 16 | x3_186 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_170), "r"(x_word_187));
                int sf_dst_188 = ((sf_c_131 * 4 + 3) * 8 + sf_d_132) * 16 + sf_g_133 * 4;
                int gate_idx_189 = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row_130) * sf_cols + kb * 16 + 12;
                int up_idx_190 = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row_130) * sf_cols + kb * 16 + 12;
                unsigned int gate0_191 = w1_scale[gate_idx_189];
                unsigned int gate1_192 = w1_scale[gate_idx_189 + 1];
                unsigned int gate2_193 = w1_scale[gate_idx_189 + 2];
                unsigned int gate3_194 = w1_scale[gate_idx_189 + 3];
                unsigned int up0_195 = w1_scale[up_idx_190];
                unsigned int up1_196 = w1_scale[up_idx_190 + 1];
                unsigned int up2_197 = w1_scale[up_idx_190 + 2];
                unsigned int up3_198 = w1_scale[up_idx_190 + 3];
                unsigned int gate_word_199 = gate0_191 | gate1_192 << 8 | gate2_193 << 16 | gate3_194 << 24;
                unsigned int up_word_200 = up0_195 | up1_196 << 8 | up2_197 << 16 | up3_198 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst_188), "r"(gate_word_199));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst_188), "r"(up_word_200));
                unsigned int x0_201 = 0;
                unsigned int x1_202 = 0;
                unsigned int x2_203 = 0;
                unsigned int x3_204 = 0;
                if (sf_row_130 < 8) {
                    int sf_pair_11 = M * top_k;
                    if (route_active) {
                        sf_pair_11 = sorted_token_ids[route_base + sf_row_130];
                    }
                    int _min_11 = ((sf_pair_11 / top_k) < (M - 1) ? (sf_pair_11 / top_k) : (M - 1));
                    int sf_token_11 = _min_11;
                    int x_idx_11 = sf_token_11 * sf_cols + kb * 16 + 12;
                    x0_201 = x_scale[x_idx_11];
                    x1_202 = x_scale[x_idx_11 + 1];
                    x2_203 = x_scale[x_idx_11 + 2];
                    x3_204 = x_scale[x_idx_11 + 3];
                }
                unsigned int x_word_205 = x0_201 | x1_202 << 8 | x2_203 << 16 | x3_204 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_188), "r"(x_word_205));
                int sf_row_206 = 96 + lane;
                int sf_c_207 = lane / 8;
                int sf_d_208 = lane % 8;
                int sf_g_209 = 3;
                int sf_dst_210 = (sf_c_207 * 4 * 8 + sf_d_208) * 16 + sf_g_209 * 4;
                int gate_idx_211 = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row_206) * sf_cols + kb * 16;
                int up_idx_212 = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row_206) * sf_cols + kb * 16;
                unsigned int gate0_213 = w1_scale[gate_idx_211];
                unsigned int gate1_214 = w1_scale[gate_idx_211 + 1];
                unsigned int gate2_215 = w1_scale[gate_idx_211 + 2];
                unsigned int gate3_216 = w1_scale[gate_idx_211 + 3];
                unsigned int up0_217 = w1_scale[up_idx_212];
                unsigned int up1_218 = w1_scale[up_idx_212 + 1];
                unsigned int up2_219 = w1_scale[up_idx_212 + 2];
                unsigned int up3_220 = w1_scale[up_idx_212 + 3];
                unsigned int gate_word_221 = gate0_213 | gate1_214 << 8 | gate2_215 << 16 | gate3_216 << 24;
                unsigned int up_word_222 = up0_217 | up1_218 << 8 | up2_219 << 16 | up3_220 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst_210), "r"(gate_word_221));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst_210), "r"(up_word_222));
                unsigned int x0_223 = 0;
                unsigned int x1_224 = 0;
                unsigned int x2_225 = 0;
                unsigned int x3_226 = 0;
                if (sf_row_206 < 8) {
                    int sf_pair_12 = M * top_k;
                    if (route_active) {
                        sf_pair_12 = sorted_token_ids[route_base + sf_row_206];
                    }
                    int _min_12 = ((sf_pair_12 / top_k) < (M - 1) ? (sf_pair_12 / top_k) : (M - 1));
                    int sf_token_12 = _min_12;
                    int x_idx_12 = sf_token_12 * sf_cols + kb * 16;
                    x0_223 = x_scale[x_idx_12];
                    x1_224 = x_scale[x_idx_12 + 1];
                    x2_225 = x_scale[x_idx_12 + 2];
                    x3_226 = x_scale[x_idx_12 + 3];
                }
                unsigned int x_word_227 = x0_223 | x1_224 << 8 | x2_225 << 16 | x3_226 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_210), "r"(x_word_227));
                int sf_dst_228 = ((sf_c_207 * 4 + 1) * 8 + sf_d_208) * 16 + sf_g_209 * 4;
                int gate_idx_229 = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row_206) * sf_cols + kb * 16 + 4;
                int up_idx_230 = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row_206) * sf_cols + kb * 16 + 4;
                unsigned int gate0_231 = w1_scale[gate_idx_229];
                unsigned int gate1_232 = w1_scale[gate_idx_229 + 1];
                unsigned int gate2_233 = w1_scale[gate_idx_229 + 2];
                unsigned int gate3_234 = w1_scale[gate_idx_229 + 3];
                unsigned int up0_235 = w1_scale[up_idx_230];
                unsigned int up1_236 = w1_scale[up_idx_230 + 1];
                unsigned int up2_237 = w1_scale[up_idx_230 + 2];
                unsigned int up3_238 = w1_scale[up_idx_230 + 3];
                unsigned int gate_word_239 = gate0_231 | gate1_232 << 8 | gate2_233 << 16 | gate3_234 << 24;
                unsigned int up_word_240 = up0_235 | up1_236 << 8 | up2_237 << 16 | up3_238 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst_228), "r"(gate_word_239));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst_228), "r"(up_word_240));
                unsigned int x0_241 = 0;
                unsigned int x1_242 = 0;
                unsigned int x2_243 = 0;
                unsigned int x3_244 = 0;
                if (sf_row_206 < 8) {
                    int sf_pair_13 = M * top_k;
                    if (route_active) {
                        sf_pair_13 = sorted_token_ids[route_base + sf_row_206];
                    }
                    int _min_13 = ((sf_pair_13 / top_k) < (M - 1) ? (sf_pair_13 / top_k) : (M - 1));
                    int sf_token_13 = _min_13;
                    int x_idx_13 = sf_token_13 * sf_cols + kb * 16 + 4;
                    x0_241 = x_scale[x_idx_13];
                    x1_242 = x_scale[x_idx_13 + 1];
                    x2_243 = x_scale[x_idx_13 + 2];
                    x3_244 = x_scale[x_idx_13 + 3];
                }
                unsigned int x_word_245 = x0_241 | x1_242 << 8 | x2_243 << 16 | x3_244 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_228), "r"(x_word_245));
                int sf_dst_246 = ((sf_c_207 * 4 + 2) * 8 + sf_d_208) * 16 + sf_g_209 * 4;
                int gate_idx_247 = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row_206) * sf_cols + kb * 16 + 8;
                int up_idx_248 = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row_206) * sf_cols + kb * 16 + 8;
                unsigned int gate0_249 = w1_scale[gate_idx_247];
                unsigned int gate1_250 = w1_scale[gate_idx_247 + 1];
                unsigned int gate2_251 = w1_scale[gate_idx_247 + 2];
                unsigned int gate3_252 = w1_scale[gate_idx_247 + 3];
                unsigned int up0_253 = w1_scale[up_idx_248];
                unsigned int up1_254 = w1_scale[up_idx_248 + 1];
                unsigned int up2_255 = w1_scale[up_idx_248 + 2];
                unsigned int up3_256 = w1_scale[up_idx_248 + 3];
                unsigned int gate_word_257 = gate0_249 | gate1_250 << 8 | gate2_251 << 16 | gate3_252 << 24;
                unsigned int up_word_258 = up0_253 | up1_254 << 8 | up2_255 << 16 | up3_256 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst_246), "r"(gate_word_257));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst_246), "r"(up_word_258));
                unsigned int x0_259 = 0;
                unsigned int x1_260 = 0;
                unsigned int x2_261 = 0;
                unsigned int x3_262 = 0;
                if (sf_row_206 < 8) {
                    int sf_pair_14 = M * top_k;
                    if (route_active) {
                        sf_pair_14 = sorted_token_ids[route_base + sf_row_206];
                    }
                    int _min_14 = ((sf_pair_14 / top_k) < (M - 1) ? (sf_pair_14 / top_k) : (M - 1));
                    int sf_token_14 = _min_14;
                    int x_idx_14 = sf_token_14 * sf_cols + kb * 16 + 8;
                    x0_259 = x_scale[x_idx_14];
                    x1_260 = x_scale[x_idx_14 + 1];
                    x2_261 = x_scale[x_idx_14 + 2];
                    x3_262 = x_scale[x_idx_14 + 3];
                }
                unsigned int x_word_263 = x0_259 | x1_260 << 8 | x2_261 << 16 | x3_262 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_246), "r"(x_word_263));
                int sf_dst_264 = ((sf_c_207 * 4 + 3) * 8 + sf_d_208) * 16 + sf_g_209 * 4;
                int gate_idx_265 = (expert * intermediate_blocks * 256 + intermediate_block * 128 + sf_row_206) * sf_cols + kb * 16 + 12;
                int up_idx_266 = (expert * intermediate_blocks * 256 + intermediate_blocks * 128 + intermediate_block * 128 + sf_row_206) * sf_cols + kb * 16 + 12;
                unsigned int gate0_267 = w1_scale[gate_idx_265];
                unsigned int gate1_268 = w1_scale[gate_idx_265 + 1];
                unsigned int gate2_269 = w1_scale[gate_idx_265 + 2];
                unsigned int gate3_270 = w1_scale[gate_idx_265 + 3];
                unsigned int up0_271 = w1_scale[up_idx_266];
                unsigned int up1_272 = w1_scale[up_idx_266 + 1];
                unsigned int up2_273 = w1_scale[up_idx_266 + 2];
                unsigned int up3_274 = w1_scale[up_idx_266 + 3];
                unsigned int gate_word_275 = gate0_267 | gate1_268 << 8 | gate2_269 << 16 | gate3_270 << 24;
                unsigned int up_word_276 = up0_271 | up1_272 << 8 | up2_273 << 16 | up3_274 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(gate_sf_base + sf_dst_264), "r"(gate_word_275));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(up_sf_base + sf_dst_264), "r"(up_word_276));
                unsigned int x0_277 = 0;
                unsigned int x1_278 = 0;
                unsigned int x2_279 = 0;
                unsigned int x3_280 = 0;
                if (sf_row_206 < 8) {
                    int sf_pair_15 = M * top_k;
                    if (route_active) {
                        sf_pair_15 = sorted_token_ids[route_base + sf_row_206];
                    }
                    int _min_15 = ((sf_pair_15 / top_k) < (M - 1) ? (sf_pair_15 / top_k) : (M - 1));
                    int sf_token_15 = _min_15;
                    int x_idx_15 = sf_token_15 * sf_cols + kb * 16 + 12;
                    x0_277 = x_scale[x_idx_15];
                    x1_278 = x_scale[x_idx_15 + 1];
                    x2_279 = x_scale[x_idx_15 + 2];
                    x3_280 = x_scale[x_idx_15 + 3];
                }
                unsigned int x_word_281 = x0_277 | x1_278 << 8 | x2_279 << 16 | x3_280 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_264), "r"(x_word_281));
                if (elect_sync()) {
                    int pair0 = M * top_k;
                    int pair1 = M * top_k;
                    int pair2 = M * top_k;
                    int pair3 = M * top_k;
                    int pair4 = M * top_k;
                    int pair5 = M * top_k;
                    int pair6 = M * top_k;
                    int pair7 = M * top_k;
                    if (route_active) {
                        pair0 = sorted_token_ids[route_base];
                        pair1 = sorted_token_ids[route_base + 1];
                        pair2 = sorted_token_ids[route_base + 2];
                        pair3 = sorted_token_ids[route_base + 3];
                        pair4 = sorted_token_ids[route_base + 4];
                        pair5 = sorted_token_ids[route_base + 5];
                        pair6 = sorted_token_ids[route_base + 6];
                        pair7 = sorted_token_ids[route_base + 7];
                    }
                    int _min_16 = ((pair0 / top_k) < (M - 1) ? (pair0 / top_k) : (M - 1));
                    int row0 = _min_16;
                    int _min_17 = ((pair1 / top_k) < (M - 1) ? (pair1 / top_k) : (M - 1));
                    int row1 = _min_17;
                    int _min_18 = ((pair2 / top_k) < (M - 1) ? (pair2 / top_k) : (M - 1));
                    int row2 = _min_18;
                    int _min_19 = ((pair3 / top_k) < (M - 1) ? (pair3 / top_k) : (M - 1));
                    int row3 = _min_19;
                    int _min_20 = ((pair4 / top_k) < (M - 1) ? (pair4 / top_k) : (M - 1));
                    int row4 = _min_20;
                    int _min_21 = ((pair5 / top_k) < (M - 1) ? (pair5 / top_k) : (M - 1));
                    int row5 = _min_21;
                    int _min_22 = ((pair6 / top_k) < (M - 1) ? (pair6 / top_k) : (M - 1));
                    int row6 = _min_22;
                    int _min_23 = ((pair7 / top_k) < (M - 1) ? (pair7 / top_k) : (M - 1));
                    int row7 = _min_23;
                    mbarrier_arrive_expect_tx(up_full_addr + (up_stage) * 8, 33792);
                    tma_gather4_gmem2smem(smem_x_addr + up_stage * 39936, ((const void*)(_loom_tma_param_base + 0)), kb * 128, row0, row1, row2, row3, up_full_addr + (up_stage) * 8);
                    tma_gather4_gmem2smem(smem_x_addr + up_stage * 39936 + 512, ((const void*)(_loom_tma_param_base + 0)), kb * 128, row4, row5, row6, row7, up_full_addr + (up_stage) * 8);
                    tma_4d_gmem2smem(smem_w1_addr + up_stage * 39936, ((const void*)(_loom_tma_param_base + 128)), 0, intermediate_block * 128, kb, expert, up_full_addr + (up_stage) * 8);
                    tma_4d_gmem2smem(smem_w1_addr + up_stage * 39936 + 16384, ((const void*)(_loom_tma_param_base + 128)), 0, intermediate_blocks * 128 + intermediate_block * 128, kb, expert, up_full_addr + (up_stage) * 8);
                }
                up_stage += 1;
                if (up_stage == 3) { up_stage = 0; _phase_up_free ^= 1; }
            }
            asm volatile("barrier.sync 14, 192;" ::: "memory");
            unsigned int down_stage = 0;
            int w2_sf_cols = intermediate_blocks * 8;
            unsigned int _phase_down_free = 1;
            #pragma unroll 1
            for (int ob = 0; ob < K / 128; ob++) {
                mbarrier_wait(down_free_addr + (down_stage) * 8, _phase_down_free);
                int w2_sf_base = smem_w2_sf_addr + down_stage * 9216;
                int sf_row_down = lane;
                int sf_c_down = lane / 8;
                int sf_d_down = lane % 8;
                int sf_g_down = 0;
                int sf_dst_down = (sf_c_down * 2 * 8 + sf_d_down) * 16 + sf_g_down * 4;
                int w2_idx = (expert * K + ob * 128 + sf_row_down) * w2_sf_cols + intermediate_block * 8;
                unsigned int w20 = w2_scale[w2_idx];
                unsigned int w21 = w2_scale[w2_idx + 1];
                unsigned int w22 = w2_scale[w2_idx + 2];
                unsigned int w23 = w2_scale[w2_idx + 3];
                unsigned int w2_word = w20 | w21 << 8 | w22 << 16 | w23 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(w2_sf_base + sf_dst_down), "r"(w2_word));
                int sf_dst_down_0 = ((sf_c_down * 2 + 1) * 8 + sf_d_down) * 16 + sf_g_down * 4;
                int w2_idx_1 = (expert * K + ob * 128 + sf_row_down) * w2_sf_cols + intermediate_block * 8 + 4;
                unsigned int w20_2 = w2_scale[w2_idx_1];
                unsigned int w21_3 = w2_scale[w2_idx_1 + 1];
                unsigned int w22_4 = w2_scale[w2_idx_1 + 2];
                unsigned int w23_5 = w2_scale[w2_idx_1 + 3];
                unsigned int w2_word_6 = w20_2 | w21_3 << 8 | w22_4 << 16 | w23_5 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(w2_sf_base + sf_dst_down_0), "r"(w2_word_6));
                int sf_row_down_7 = 32 + lane;
                int sf_c_down_8 = lane / 8;
                int sf_d_down_9 = lane % 8;
                int sf_g_down_10 = 1;
                int sf_dst_down_11 = (sf_c_down_8 * 2 * 8 + sf_d_down_9) * 16 + sf_g_down_10 * 4;
                int w2_idx_12 = (expert * K + ob * 128 + sf_row_down_7) * w2_sf_cols + intermediate_block * 8;
                unsigned int w20_13 = w2_scale[w2_idx_12];
                unsigned int w21_14 = w2_scale[w2_idx_12 + 1];
                unsigned int w22_15 = w2_scale[w2_idx_12 + 2];
                unsigned int w23_16 = w2_scale[w2_idx_12 + 3];
                unsigned int w2_word_17 = w20_13 | w21_14 << 8 | w22_15 << 16 | w23_16 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(w2_sf_base + sf_dst_down_11), "r"(w2_word_17));
                int sf_dst_down_18 = ((sf_c_down_8 * 2 + 1) * 8 + sf_d_down_9) * 16 + sf_g_down_10 * 4;
                int w2_idx_19 = (expert * K + ob * 128 + sf_row_down_7) * w2_sf_cols + intermediate_block * 8 + 4;
                unsigned int w20_20 = w2_scale[w2_idx_19];
                unsigned int w21_21 = w2_scale[w2_idx_19 + 1];
                unsigned int w22_22 = w2_scale[w2_idx_19 + 2];
                unsigned int w23_23 = w2_scale[w2_idx_19 + 3];
                unsigned int w2_word_24 = w20_20 | w21_21 << 8 | w22_22 << 16 | w23_23 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(w2_sf_base + sf_dst_down_18), "r"(w2_word_24));
                int sf_row_down_25 = 64 + lane;
                int sf_c_down_26 = lane / 8;
                int sf_d_down_27 = lane % 8;
                int sf_g_down_28 = 2;
                int sf_dst_down_29 = (sf_c_down_26 * 2 * 8 + sf_d_down_27) * 16 + sf_g_down_28 * 4;
                int w2_idx_30 = (expert * K + ob * 128 + sf_row_down_25) * w2_sf_cols + intermediate_block * 8;
                unsigned int w20_31 = w2_scale[w2_idx_30];
                unsigned int w21_32 = w2_scale[w2_idx_30 + 1];
                unsigned int w22_33 = w2_scale[w2_idx_30 + 2];
                unsigned int w23_34 = w2_scale[w2_idx_30 + 3];
                unsigned int w2_word_35 = w20_31 | w21_32 << 8 | w22_33 << 16 | w23_34 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(w2_sf_base + sf_dst_down_29), "r"(w2_word_35));
                int sf_dst_down_36 = ((sf_c_down_26 * 2 + 1) * 8 + sf_d_down_27) * 16 + sf_g_down_28 * 4;
                int w2_idx_37 = (expert * K + ob * 128 + sf_row_down_25) * w2_sf_cols + intermediate_block * 8 + 4;
                unsigned int w20_38 = w2_scale[w2_idx_37];
                unsigned int w21_39 = w2_scale[w2_idx_37 + 1];
                unsigned int w22_40 = w2_scale[w2_idx_37 + 2];
                unsigned int w23_41 = w2_scale[w2_idx_37 + 3];
                unsigned int w2_word_42 = w20_38 | w21_39 << 8 | w22_40 << 16 | w23_41 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(w2_sf_base + sf_dst_down_36), "r"(w2_word_42));
                int sf_row_down_43 = 96 + lane;
                int sf_c_down_44 = lane / 8;
                int sf_d_down_45 = lane % 8;
                int sf_g_down_46 = 3;
                int sf_dst_down_47 = (sf_c_down_44 * 2 * 8 + sf_d_down_45) * 16 + sf_g_down_46 * 4;
                int w2_idx_48 = (expert * K + ob * 128 + sf_row_down_43) * w2_sf_cols + intermediate_block * 8;
                unsigned int w20_49 = w2_scale[w2_idx_48];
                unsigned int w21_50 = w2_scale[w2_idx_48 + 1];
                unsigned int w22_51 = w2_scale[w2_idx_48 + 2];
                unsigned int w23_52 = w2_scale[w2_idx_48 + 3];
                unsigned int w2_word_53 = w20_49 | w21_50 << 8 | w22_51 << 16 | w23_52 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(w2_sf_base + sf_dst_down_47), "r"(w2_word_53));
                int sf_dst_down_54 = ((sf_c_down_44 * 2 + 1) * 8 + sf_d_down_45) * 16 + sf_g_down_46 * 4;
                int w2_idx_55 = (expert * K + ob * 128 + sf_row_down_43) * w2_sf_cols + intermediate_block * 8 + 4;
                unsigned int w20_56 = w2_scale[w2_idx_55];
                unsigned int w21_57 = w2_scale[w2_idx_55 + 1];
                unsigned int w22_58 = w2_scale[w2_idx_55 + 2];
                unsigned int w23_59 = w2_scale[w2_idx_55 + 3];
                unsigned int w2_word_60 = w20_56 | w21_57 << 8 | w22_58 << 16 | w23_59 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(w2_sf_base + sf_dst_down_54), "r"(w2_word_60));
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(down_full_addr + (down_stage) * 8, 8192);
                    tma_3d_gmem2smem(smem_w2_addr + down_stage * 9216, ((const void*)(_loom_tma_param_base + 256)), intermediate_block * 64, ob * 128, expert, down_full_addr + (down_stage) * 8);
                }
                down_stage += 1;
                if (down_stage == 3) { down_stage = 0; _phase_down_free ^= 1; }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            unsigned int up_stage_mma = 0;
            unsigned int _phase_up_full = 0;
            #pragma unroll 1
            for (int kb_mma = 0; kb_mma < K / 256; kb_mma++) {
                mbarrier_wait(up_full_addr + (up_stage_mma) * 8, _phase_up_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_up_gate_sf, make_sf_cp_desc_sbo512(smem_w1_gate_sf_addr + up_stage_mma * 39936));
                    tcgen05_cp_32x128b_warpx4((tmem_up_gate_sf + 4), make_sf_cp_desc_sbo512((smem_w1_gate_sf_addr + up_stage_mma * 39936 + 128)));
                    tcgen05_cp_32x128b_warpx4((tmem_up_gate_sf + 8), make_sf_cp_desc_sbo512((smem_w1_gate_sf_addr + up_stage_mma * 39936 + 256)));
                    tcgen05_cp_32x128b_warpx4((tmem_up_gate_sf + 12), make_sf_cp_desc_sbo512((smem_w1_gate_sf_addr + up_stage_mma * 39936 + 384)));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_up_up_sf, make_sf_cp_desc_sbo512(smem_w1_up_sf_addr + up_stage_mma * 39936));
                    tcgen05_cp_32x128b_warpx4((tmem_up_up_sf + 4), make_sf_cp_desc_sbo512((smem_w1_up_sf_addr + up_stage_mma * 39936 + 128)));
                    tcgen05_cp_32x128b_warpx4((tmem_up_up_sf + 8), make_sf_cp_desc_sbo512((smem_w1_up_sf_addr + up_stage_mma * 39936 + 256)));
                    tcgen05_cp_32x128b_warpx4((tmem_up_up_sf + 12), make_sf_cp_desc_sbo512((smem_w1_up_sf_addr + up_stage_mma * 39936 + 384)));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_up_x_sf, make_sf_cp_desc_sbo512(smem_x_sf_addr + up_stage_mma * 39936));
                    tcgen05_cp_32x128b_warpx4((tmem_up_x_sf + 4), make_sf_cp_desc_sbo512((smem_x_sf_addr + up_stage_mma * 39936 + 128)));
                    tcgen05_cp_32x128b_warpx4((tmem_up_x_sf + 8), make_sf_cp_desc_sbo512((smem_x_sf_addr + up_stage_mma * 39936 + 256)));
                    tcgen05_cp_32x128b_warpx4((tmem_up_x_sf + 12), make_sf_cp_desc_sbo512((smem_x_sf_addr + up_stage_mma * 39936 + 384)));
                }
                int init_up = ((kb_mma == 0) ? 1 : 0);
                int _mma_a_lo_0 = make_warp_uniform((((smem_w1_addr) >> 4) & 0x3FFF) + (up_stage_mma) * 2496);
                int _mma_b_lo_0 = make_warp_uniform((((smem_x_addr) >> 4) & 0x3FFF) + (up_stage_mma) * 2496);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs(tmem_up_acc, a_desc + 0, b_desc + 0,
                            0x8020480U, tmem_up_gate_sf + 0, tmem_up_x_sf + 0, ((init_up) ? 0 : 1));
                        tcgen05_mma_mxf4nvf4_bs(tmem_up_acc, a_desc + 2, b_desc + 2,
                            0x8020480U, tmem_up_gate_sf + 4, tmem_up_x_sf + 4, 1);
                        tcgen05_mma_mxf4nvf4_bs(tmem_up_acc, a_desc + 4, b_desc + 4,
                            0x8020480U, tmem_up_gate_sf + 8, tmem_up_x_sf + 8, 1);
                        tcgen05_mma_mxf4nvf4_bs(tmem_up_acc, a_desc + 6, b_desc + 6,
                            0x8020480U, tmem_up_gate_sf + 12, tmem_up_x_sf + 12, 1);
                    }
                }
                int _mma_a_lo_1 = make_warp_uniform((((smem_w1_addr + 16384) >> 4) & 0x3FFF) + (up_stage_mma) * 2496);
                int _mma_b_lo_1 = make_warp_uniform((((smem_x_addr) >> 4) & 0x3FFF) + (up_stage_mma) * 2496);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (8)), a_desc + 0, b_desc + 0,
                            0x8020480U, tmem_up_up_sf + 0, tmem_up_x_sf + 0, ((init_up) ? 0 : 1));
                        tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (8)), a_desc + 2, b_desc + 2,
                            0x8020480U, tmem_up_up_sf + 4, tmem_up_x_sf + 4, 1);
                        tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (8)), a_desc + 4, b_desc + 4,
                            0x8020480U, tmem_up_up_sf + 8, tmem_up_x_sf + 8, 1);
                        tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (8)), a_desc + 6, b_desc + 6,
                            0x8020480U, tmem_up_up_sf + 12, tmem_up_x_sf + 12, 1);
                    }
                }
                elect_commit(up_free_addr + (up_stage_mma) * 8);
                up_stage_mma += 1;
                if (up_stage_mma == 3) { up_stage_mma = 0; _phase_up_full ^= 1; }
            }
            elect_commit(up_ready_addr);
            asm volatile("barrier.sync 14, 192;" ::: "memory");
            unsigned int down_stage_mma = 0;
            unsigned int _phase_down_full = 0;
            #pragma unroll 1
            for (int _ob_mma = 0; _ob_mma < K / 128; _ob_mma++) {
                mbarrier_wait(down_full_addr + (down_stage_mma) * 8, _phase_down_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_down_w2_sf, make_sf_cp_desc_sbo256(smem_w2_sf_addr + down_stage_mma * 9216));
                    tcgen05_cp_32x128b_warpx4((tmem_down_w2_sf + 4), make_sf_cp_desc_sbo256((smem_w2_sf_addr + down_stage_mma * 9216 + 128)));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_down_act_sf, make_sf_cp_desc_sbo256(smem_act_sf_cp_addr));
                    tcgen05_cp_32x128b_warpx4((tmem_down_act_sf + 4), make_sf_cp_desc_sbo256((smem_act_sf_cp_addr + 128)));
                }
                int _mma_a_lo_2 = make_warp_uniform((((smem_w2_addr) >> 4) & 0x3FFF) + (down_stage_mma) * 576);
                int _mma_b_lo_2 = make_warp_uniform(((smem_act_addr) >> 4) & 0x3FFF);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)_mma_a_lo_2) | ((uint64_t)0x80004020 << 32);
                        uint64_t b_desc = ((uint64_t)_mma_b_lo_2) | ((uint64_t)0x80004020 << 32);

                        tcgen05_mma_mxf4nvf4_bs((tmem_down_acc + (down_stage_mma * 8)), a_desc + 0, b_desc + 0,
                            0x8020480U, tmem_down_w2_sf + 0, tmem_down_act_sf + 0, 0);
                        tcgen05_mma_mxf4nvf4_bs((tmem_down_acc + (down_stage_mma * 8)), a_desc + 2, b_desc + 2,
                            0x8020480U, tmem_down_w2_sf + 4, tmem_down_act_sf + 4, 1);
                    }
                }
                elect_commit(down_ready_addr + (down_stage_mma) * 8);
                down_stage_mma += 1;
                if (down_stage_mma == 3) { down_stage_mma = 0; _phase_down_full ^= 1; }
            }
        }
    // ---- Role: consumer ----
    } else if (warp >= 2 && warp <= 5) {
        { // consumer_main
            int route_work_c = blockIdx.x;
            int subtiles_per_route_block_c = route_block_m / 8;
            int route_block_c = route_work_c / subtiles_per_route_block_c;
            int route_subtile_c = route_work_c % subtiles_per_route_block_c;
            int route_base_c = route_block_c * route_block_m + route_subtile_c * 8;
            bool route_active_c = route_base_c < num_tokens_post_padded[0];
            const int consumer_warp = warp % 4;
            const int physical_feature = consumer_warp * 32 + lane;
            int lane_pair = M * top_k;
            float lane_route_weight = 0.0f;
            if (lane < 8 && route_active_c) {
                lane_pair = sorted_token_ids[route_base_c + lane];
                if (lane_pair < M * top_k) {
                    lane_route_weight = topk_weights[lane_pair];
                }
            }
            int route_pairs[8];
            float route_weights[8];
            #pragma unroll
            for (int token_slot_route = 0; token_slot_route < 8; token_slot_route++) {
                int _shfl_0 = __shfl_sync(0xFFFFFFFF, lane_pair, token_slot_route);
                route_pairs[token_slot_route] = _shfl_0;
                float _shfl_1 = __shfl_sync(0xFFFFFFFF, lane_route_weight, token_slot_route);
                route_weights[token_slot_route] = _shfl_1;
            }
            unsigned int _phase_up_ready_0 = 0;
            mbarrier_wait(up_ready_addr, _phase_up_ready_0);
            _phase_up_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int gate_addr = taddr + (unsigned int)(physical_feature << 16);
            int up_addr = gate_addr + 8;
            float _tmem_load_0[8];
            tmem_ld_x8(&_tmem_load_0[0], gate_addr);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            float _tmem_load_1[8];
            tmem_ld_x8(&_tmem_load_1[0], up_addr);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            float act[8];
            #pragma unroll
            for (int token_slot_act = 0; token_slot_act < 8; token_slot_act++) {
                float gate = _tmem_load_0[token_slot_act];
                float up = _tmem_load_1[token_slot_act];
                float _expf_0 = __expf(-gate);
                act[token_slot_act] = gate * (1.0f / (1.0f + _expf_0)) * up;
                float _fabs_0 = fabsf(act[token_slot_act]);
                float group_max = _fabs_0;
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, group_max, 1);
                float _max_0 = max_noftz(group_max, _shfl_xor_0);
                group_max = _max_0;
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, group_max, 2);
                float _max_1 = max_noftz(group_max, _shfl_xor_1);
                group_max = _max_1;
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, group_max, 4);
                float _max_2 = max_noftz(group_max, _shfl_xor_2);
                group_max = _max_2;
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, group_max, 8);
                float _max_3 = max_noftz(group_max, _shfl_xor_3);
                group_max = _max_3;
                if (lane % 16 == 0) {
                    float _max_4 = max_noftz(group_max * 0.16666666666666666f, 1e-08f);
                    float scale_value = _max_4;
                    smem_act_scale[token_slot_act * 8 + physical_feature / 16] = scale_value;
                }
            }
            asm volatile("barrier.sync 15, 128;" ::: "memory");
            int feature_group_lane = lane - lane % 16;
            #pragma unroll
            for (int token_slot_quant = 0; token_slot_quant < 8; token_slot_quant++) {
                float act_scale = smem_act_scale[token_slot_quant * 8 + physical_feature / 16];
                float rounded_act_scale = 0.0f;
                if (lane % 16 == 0) {
                    float scale_pack_src[4];
                    scale_pack_src[0] = act_scale;
                    scale_pack_src[1] = 0.0f;
                    scale_pack_src[2] = 0.0f;
                    scale_pack_src[3] = 0.0f;
                    unsigned int scale_pack_dst[1];
                    {
                        unsigned short _lo, _hi;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(scale_pack_src[1]), "f"(scale_pack_src[0]));
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(scale_pack_src[3]), "f"(scale_pack_src[2]));
                        scale_pack_dst[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                    }
                    unsigned int scale_code = scale_pack_dst[0] & 127;
                    unsigned int scale_exp = scale_code >> 3 & 15;
                    unsigned int scale_mant = scale_code & 7;
                    if (scale_exp == 0) {
                        rounded_act_scale = (float)scale_mant * 0.001953125f;
                    } else {
                        float _exp2_0 = approx_exp2((float)scale_exp - 7.0f);
                        rounded_act_scale = _exp2_0 * (1.0f + (float)scale_mant * 0.125f);
                    }
                }
                float _shfl_2 = __shfl_sync(0xFFFFFFFF, rounded_act_scale, feature_group_lane);
                rounded_act_scale = _shfl_2;
                float safe_act_scale = ((rounded_act_scale == 0.0f) ? 1.0f : rounded_act_scale);
                float fp4_lo[8];
                float fp4_hi[8];
                #pragma unroll
                for (int fp4_lane = 0; fp4_lane < 8; fp4_lane++) {
                    float _shfl_3 = __shfl_sync(0xFFFFFFFF, act[token_slot_quant], feature_group_lane + fp4_lane);
                    fp4_lo[fp4_lane] = _shfl_3 / safe_act_scale;
                    float _shfl_4 = __shfl_sync(0xFFFFFFFF, act[token_slot_quant], feature_group_lane + 8 + fp4_lane);
                    fp4_hi[fp4_lane] = _shfl_4 / safe_act_scale;
                }
                if (lane % 16 == 0) {
                    unsigned int packed_lo[1];
                    unsigned int packed_hi[1];
                    {
                        unsigned short b0, b1, b2, b3;
                        {
                            asm(" { .reg .b8 __t; \n"     " cvt.rn.satfinite.e2m1x2.f32 __t, %1, %2; \n"     " mov.b16 %0, {__t, 0}; \n"     " } \n"     : "=h"(b0) : "f"(fp4_lo[1]), "f"(fp4_lo[0]));
                        }
                        {
                            asm(" { .reg .b8 __t; \n"     " cvt.rn.satfinite.e2m1x2.f32 __t, %1, %2; \n"     " mov.b16 %0, {__t, 0}; \n"     " } \n"     : "=h"(b1) : "f"(fp4_lo[3]), "f"(fp4_lo[2]));
                        }
                        {
                            asm(" { .reg .b8 __t; \n"     " cvt.rn.satfinite.e2m1x2.f32 __t, %1, %2; \n"     " mov.b16 %0, {__t, 0}; \n"     " } \n"     : "=h"(b2) : "f"(fp4_lo[5]), "f"(fp4_lo[4]));
                        }
                        {
                            asm(" { .reg .b8 __t; \n"     " cvt.rn.satfinite.e2m1x2.f32 __t, %1, %2; \n"     " mov.b16 %0, {__t, 0}; \n"     " } \n"     : "=h"(b3) : "f"(fp4_lo[7]), "f"(fp4_lo[6]));
                        }
                        packed_lo[0] = ((unsigned)b0 & 0xFFu) | (((unsigned)b1 & 0xFFu) << 8) | (((unsigned)b2 & 0xFFu) << 16) | (((unsigned)b3 & 0xFFu) << 24);
                    }
                    {
                        unsigned short b0, b1, b2, b3;
                        {
                            asm(" { .reg .b8 __t; \n"     " cvt.rn.satfinite.e2m1x2.f32 __t, %1, %2; \n"     " mov.b16 %0, {__t, 0}; \n"     " } \n"     : "=h"(b0) : "f"(fp4_hi[1]), "f"(fp4_hi[0]));
                        }
                        {
                            asm(" { .reg .b8 __t; \n"     " cvt.rn.satfinite.e2m1x2.f32 __t, %1, %2; \n"     " mov.b16 %0, {__t, 0}; \n"     " } \n"     : "=h"(b1) : "f"(fp4_hi[3]), "f"(fp4_hi[2]));
                        }
                        {
                            asm(" { .reg .b8 __t; \n"     " cvt.rn.satfinite.e2m1x2.f32 __t, %1, %2; \n"     " mov.b16 %0, {__t, 0}; \n"     " } \n"     : "=h"(b2) : "f"(fp4_hi[5]), "f"(fp4_hi[4]));
                        }
                        {
                            asm(" { .reg .b8 __t; \n"     " cvt.rn.satfinite.e2m1x2.f32 __t, %1, %2; \n"     " mov.b16 %0, {__t, 0}; \n"     " } \n"     : "=h"(b3) : "f"(fp4_hi[7]), "f"(fp4_hi[6]));
                        }
                        packed_hi[0] = ((unsigned)b0 & 0xFFu) | (((unsigned)b1 & 0xFFu) << 8) | (((unsigned)b2 & 0xFFu) << 16) | (((unsigned)b3 & 0xFFu) << 24);
                    }
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((smem_act_addr + (unsigned int)(token_slot_quant * 64 + physical_feature / 2 ^ (token_slot_quant * 64 + physical_feature / 2 >> 7 & 3) << 4))), "r"(packed_lo[0]), "r"(packed_hi[0]) : "memory");
                }
            }
            int sf_c_act = physical_feature % 32 / 8;
            int sf_d_act = physical_feature % 8;
            int sf_g_act = physical_feature / 32;
            float act_sf_values[4];
            act_sf_values[0] = 0.0f;
            if (physical_feature < 8) {
                act_sf_values[0] = smem_act_scale[physical_feature * 8];
            }
            act_sf_values[1] = 0.0f;
            if (physical_feature < 8) {
                act_sf_values[1] = smem_act_scale[physical_feature * 8 + 1];
            }
            act_sf_values[2] = 0.0f;
            if (physical_feature < 8) {
                act_sf_values[2] = smem_act_scale[physical_feature * 8 + 2];
            }
            act_sf_values[3] = 0.0f;
            if (physical_feature < 8) {
                act_sf_values[3] = smem_act_scale[physical_feature * 8 + 3];
            }
            unsigned int packed_act_sf[1];
            {
                unsigned short _lo, _hi;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(act_sf_values[1]), "f"(act_sf_values[0]));
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(act_sf_values[3]), "f"(act_sf_values[2]));
                packed_act_sf[0] = (unsigned)_lo | ((unsigned)_hi << 16);
            }
            int act_sf_dst = (sf_c_act * 2 * 8 + sf_d_act) * 16 + sf_g_act * 4;
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_act_sf_cp_addr + (unsigned int)act_sf_dst), "r"((packed_act_sf[0])));
            float act_sf_values_0[4];
            act_sf_values_0[0] = 0.0f;
            if (physical_feature < 8) {
                act_sf_values_0[0] = smem_act_scale[physical_feature * 8 + 4];
            }
            act_sf_values_0[1] = 0.0f;
            if (physical_feature < 8) {
                act_sf_values_0[1] = smem_act_scale[physical_feature * 8 + 4 + 1];
            }
            act_sf_values_0[2] = 0.0f;
            if (physical_feature < 8) {
                act_sf_values_0[2] = smem_act_scale[physical_feature * 8 + 4 + 2];
            }
            act_sf_values_0[3] = 0.0f;
            if (physical_feature < 8) {
                act_sf_values_0[3] = smem_act_scale[physical_feature * 8 + 4 + 3];
            }
            unsigned int packed_act_sf_1[1];
            {
                unsigned short _lo, _hi;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(act_sf_values_0[1]), "f"(act_sf_values_0[0]));
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(act_sf_values_0[3]), "f"(act_sf_values_0[2]));
                packed_act_sf_1[0] = (unsigned)_lo | ((unsigned)_hi << 16);
            }
            int act_sf_dst_2 = ((sf_c_act * 2 + 1) * 8 + sf_d_act) * 16 + sf_g_act * 4;
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_act_sf_cp_addr + (unsigned int)act_sf_dst_2), "r"((packed_act_sf_1[0])));
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 15, 128;" ::: "memory");
            asm volatile("barrier.sync 14, 192;" ::: "memory");
            unsigned int down_stage_c = 0;
            unsigned int _phase_down_ready = 0;
            #pragma unroll 1
            for (int ob_c = 0; ob_c < K / 128; ob_c++) {
                mbarrier_wait(down_ready_addr + (down_stage_c) * 8, _phase_down_ready);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int down_addr = taddr + 16 + (unsigned int)(physical_feature << 16) + down_stage_c * 8;
                float _tmem_load_2[8];
                tmem_ld_x8(&_tmem_load_2[0], down_addr);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int token_slot_out = 0; token_slot_out < 8; token_slot_out++) {
                    int pair_out = route_pairs[token_slot_out];
                    float route_weight = 0.0f;
                    if (pair_out < M * top_k) {
                        route_weight = route_weights[token_slot_out];
                    }
                    smem_out[token_slot_out * 128 + physical_feature] = _tmem_load_2[token_slot_out] * route_weight * scaling_factor;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 15, 128;" ::: "memory");
                if (warp == 2) {
                    if (elect_sync()) {
                        #pragma unroll
                        for (int token_slot_reduce = 0; token_slot_reduce < 8; token_slot_reduce++) {
                            int pair_reduce = route_pairs[token_slot_reduce];
                            if (pair_reduce < M * top_k) {
                                int token_reduce = pair_reduce / top_k;
                                {
                                    void* _cpred_dst_0 = reinterpret_cast<void*>(out + (token_reduce * K + ob_c * 128));
                                    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16"
                                        " [%0], [%1], %2;"
                                        :: "l"(_cpred_dst_0), "r"(smem_out_addr + (unsigned int)(token_slot_reduce * 128 * 2)), "r"((uint32_t)(256))
                                        : "memory");
                                }
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group 0;");
                    }
                }
                asm volatile("barrier.sync 15, 128;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(down_free_addr + (down_stage_c) * 8);
                }
                down_stage_c += 1;
                if (down_stage_c == 3) { down_stage_c = 0; _phase_down_ready ^= 1; }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"


namespace {
constexpr int kAlphaMoeNvfp4GeneratedThreads = THREADS;
constexpr int kAlphaMoeNvfp4GeneratedSmemTotal = SMEM_TOTAL;
}  // namespace

#undef LOOM_INF
#undef TMEM_NCOLS
#undef TMEM_UP_ACC_OFFSET
#undef TMEM_DOWN_ACC_OFFSET
#undef TMEM_UP_GATE_SF_OFFSET
#undef TMEM_UP_UP_SF_OFFSET
#undef TMEM_UP_X_SF_OFFSET
#undef TMEM_DOWN_W2_SF_OFFSET
#undef TMEM_DOWN_ACT_SF_OFFSET
#undef NUM_UP_PIPE_STAGES
#undef NUM_DOWN_PIPE_STAGES
#undef NUM_SINGLE_PIPE_STAGES
#undef SMEM_SMEM_W1_OFF
#undef SMEM_SMEM_W1_STAGE_BYTES
#undef SMEM_SMEM_W1_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_SMEM_W1_GATE_SF_OFF
#undef SMEM_SMEM_W1_GATE_SF_STAGE_BYTES
#undef SMEM_SMEM_W1_GATE_SF_STRIDE
#undef SMEM_SMEM_W1_UP_SF_OFF
#undef SMEM_SMEM_W1_UP_SF_STAGE_BYTES
#undef SMEM_SMEM_W1_UP_SF_STRIDE
#undef SMEM_SMEM_X_SF_OFF
#undef SMEM_SMEM_X_SF_STAGE_BYTES
#undef SMEM_SMEM_X_SF_STRIDE
#undef SMEM_SMEM_W2_OFF
#undef SMEM_SMEM_W2_STAGE_BYTES
#undef SMEM_SMEM_W2_STRIDE
#undef SMEM_SMEM_W2_SF_OFF
#undef SMEM_SMEM_W2_SF_STAGE_BYTES
#undef SMEM_SMEM_W2_SF_STRIDE
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_ACT_SCALE_OFF
#undef SMEM_SMEM_ACT_SCALE_STAGE_BYTES
#undef SMEM_SMEM_ACT_SCALE_STRIDE
#undef SMEM_SMEM_ACT_SF_CP_OFF
#undef SMEM_SMEM_ACT_SF_CP_STAGE_BYTES
#undef SMEM_SMEM_ACT_SF_CP_STRIDE
#undef SMEM_SMEM_OUT_OFF
#undef SMEM_SMEM_OUT_STAGE_BYTES
#undef SMEM_SMEM_OUT_STRIDE
#undef SMEM_TOTAL
#undef THREADS

// clang-format on

namespace flashinfer {
namespace alphamoe_nvfp4_sm100 {

using tvm::ffi::TensorView;

constexpr int kThreads = kAlphaMoeNvfp4GeneratedThreads;
constexpr int kSmemTotal = kAlphaMoeNvfp4GeneratedSmemTotal;
constexpr int kRouteSubtile = 8;
constexpr int kUpBlockK = 256;
constexpr int kRowsPerIntermediateBlock = 256;
constexpr int64_t kIntMax = std::numeric_limits<int>::max();

struct ProblemDims {
  int m;
  int k;
  int n;
  int intermediate;
  int top_k;
  int block_m;
  int64_t num_route_blocks;
};

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckCu(CUresult status, const char* operation) {
  TVM_FFI_ICHECK(status == CUDA_SUCCESS)
      << operation << " failed with CUresult=" << static_cast<int>(status);
}

inline void CheckSm100OrSm103(int device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(compute capability major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(compute capability minor)");
  TVM_FFI_ICHECK(major == 10 && (minor == 0 || minor == 3))
      << "alphamoe_nvfp4_sm100 requires an exact SM100 or SM103 device, got sm_" << major << minor;
}

inline void CheckDtype(const TensorView& tensor, DLDataType expected, const char* name,
                       const char* expected_name) {
  const DLDataType actual = tensor.dtype();
  TVM_FFI_ICHECK(actual.code == expected.code && actual.bits == expected.bits &&
                 actual.lanes == expected.lanes)
      << name << " must have dtype " << expected_name << ", got (code=" << int(actual.code)
      << ", bits=" << int(actual.bits) << ", lanes=" << int(actual.lanes) << ")";
}

inline void CheckTensor(const TensorView& tensor, DLDataType dtype, int ndim, bool contiguous,
                        const char* name, const char* dtype_name) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.ndim() == ndim)
      << name << " must be " << ndim << "D, got ndim=" << tensor.ndim();
  CheckDtype(tensor, dtype, name, dtype_name);
  if (contiguous) {
    TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
  }
}

inline void CheckSameDevice(const TensorView& tensor, int device_id, const char* name) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA && tensor.device().device_id == device_id)
      << name << " must be on CUDA device " << device_id;
}

inline void CheckShape2(const TensorView& tensor, int64_t d0, int64_t d1, const char* name) {
  TVM_FFI_ICHECK(tensor.size(0) == d0 && tensor.size(1) == d1)
      << name << " must have shape (" << d0 << ", " << d1 << "), got (" << tensor.size(0) << ", "
      << tensor.size(1) << ")";
}

inline void CheckShape3(const TensorView& tensor, int64_t d0, int64_t d1, int64_t d2,
                        const char* name) {
  TVM_FFI_ICHECK(tensor.size(0) == d0 && tensor.size(1) == d1 && tensor.size(2) == d2)
      << name << " must have shape (" << d0 << ", " << d1 << ", " << d2 << "), got ("
      << tensor.size(0) << ", " << tensor.size(1) << ", " << tensor.size(2) << ")";
}

inline void CheckIntIndexExtent(int64_t extent, const char* name) {
  TVM_FFI_ICHECK(extent >= 0 && extent <= kIntMax)
      << name << " (" << extent << ") must fit in the generated kernel's signed int indexing";
}

struct TensorRange {
  uintptr_t begin;
  uintptr_t end;
};

inline TensorRange GetTensorRange(const TensorView& tensor, const char* name) {
  const DLDataType dtype = tensor.dtype();
  const uint64_t bits = static_cast<uint64_t>(dtype.bits) * dtype.lanes;
  TVM_FFI_ICHECK(bits > 0 && bits % 8 == 0) << name << " must have a byte-addressable dtype";
  const uint64_t bytes_per_element = bits / 8;
  uint64_t max_element_offset = 0;
  for (int dim = 0; dim < tensor.ndim(); ++dim) {
    const int64_t extent = tensor.size(dim);
    const int64_t stride = tensor.stride(dim);
    TVM_FFI_ICHECK(extent > 0 && stride >= 0)
        << name << " must have positive extents and non-negative strides";
    const uint64_t steps = static_cast<uint64_t>(extent - 1);
    const uint64_t unsigned_stride = static_cast<uint64_t>(stride);
    TVM_FFI_ICHECK(unsigned_stride == 0 ||
                   steps <= (std::numeric_limits<uint64_t>::max() - max_element_offset) /
                                unsigned_stride)
        << name << " storage extent overflows uint64_t";
    max_element_offset += steps * unsigned_stride;
  }
  TVM_FFI_ICHECK(max_element_offset < std::numeric_limits<uint64_t>::max())
      << name << " storage extent overflows uint64_t";
  const uint64_t span_elements = max_element_offset + 1;
  TVM_FFI_ICHECK(span_elements <= std::numeric_limits<uint64_t>::max() / bytes_per_element)
      << name << " byte extent overflows uint64_t";
  const uint64_t bytes = span_elements * bytes_per_element;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  TVM_FFI_ICHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin)
      << name << " byte range overflows uintptr_t";
  return {begin, begin + static_cast<uintptr_t>(bytes)};
}

inline void CheckNoOverlap(const TensorView& output, const TensorView& input,
                           const char* input_name) {
  const TensorRange output_range = GetTensorRange(output, "out");
  const TensorRange input_range = GetTensorRange(input, input_name);
  TVM_FFI_ICHECK(!(output_range.begin < input_range.end && input_range.begin < output_range.end))
      << "out must not overlap " << input_name
      << ": the frozen kernel asynchronously reads inputs through __restrict__ pointers/TMA";
}

inline ProblemDims CheckInputs(
    const TensorView& hidden_states, const TensorView& hidden_states_scale,
    const TensorView& gemm1_weights, const TensorView& gemm1_weights_scale,
    const TensorView& gemm2_weights, const TensorView& gemm2_weights_scale,
    const TensorView& sorted_token_ids, const TensorView& expert_ids,
    const TensorView& num_tokens_post_padded, const TensorView& topk_weights, const TensorView& out,
    int64_t top_k, int64_t block_m, double routed_scaling_factor) {
  CheckTensor(hidden_states, dl_uint8, 2, false, "hidden_states", "uint8");
  CheckTensor(hidden_states_scale, dl_float8_e4m3fn, 2, true, "hidden_states_scale",
              "float8_e4m3fn");
  CheckTensor(gemm1_weights, dl_uint8, 3, true, "gemm1_weights", "uint8");
  CheckTensor(gemm1_weights_scale, dl_float8_e4m3fn, 3, true, "gemm1_weights_scale",
              "float8_e4m3fn");
  CheckTensor(gemm2_weights, dl_uint8, 3, true, "gemm2_weights", "uint8");
  CheckTensor(gemm2_weights_scale, dl_float8_e4m3fn, 3, true, "gemm2_weights_scale",
              "float8_e4m3fn");
  CheckTensor(sorted_token_ids, dl_int32, 1, true, "sorted_token_ids", "int32");
  CheckTensor(expert_ids, dl_int32, 1, true, "expert_ids", "int32");
  CheckTensor(num_tokens_post_padded, dl_int32, 1, true, "num_tokens_post_padded", "int32");
  CheckTensor(topk_weights, dl_float32, 2, true, "topk_weights", "float32");
  CheckTensor(out, dl_bfloat16, 2, true, "out", "bfloat16");

  const int device_id = hidden_states.device().device_id;
  CheckSameDevice(hidden_states_scale, device_id, "hidden_states_scale");
  CheckSameDevice(gemm1_weights, device_id, "gemm1_weights");
  CheckSameDevice(gemm1_weights_scale, device_id, "gemm1_weights_scale");
  CheckSameDevice(gemm2_weights, device_id, "gemm2_weights");
  CheckSameDevice(gemm2_weights_scale, device_id, "gemm2_weights_scale");
  CheckSameDevice(sorted_token_ids, device_id, "sorted_token_ids");
  CheckSameDevice(expert_ids, device_id, "expert_ids");
  CheckSameDevice(num_tokens_post_padded, device_id, "num_tokens_post_padded");
  CheckSameDevice(topk_weights, device_id, "topk_weights");
  CheckSameDevice(out, device_id, "out");
  CheckSm100OrSm103(device_id);

  TVM_FFI_ICHECK(hidden_states.stride(1) == 1)
      << "hidden_states must have unit innermost stride, got " << hidden_states.stride(1);
  TVM_FFI_ICHECK(hidden_states.stride(0) > 0 && hidden_states.stride(0) >= hidden_states.size(1) &&
                 hidden_states.stride(0) % 16 == 0)
      << "hidden_states row stride must be positive, non-overlapping, and 16-byte aligned, got "
      << hidden_states.stride(0) << " for packed row width " << hidden_states.size(1);
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(hidden_states.data_ptr()) % 16 == 0)
      << "hidden_states data pointer must be 16-byte aligned for TMA";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(gemm1_weights.data_ptr()) % 16 == 0)
      << "gemm1_weights data pointer must be 16-byte aligned for TMA";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(gemm2_weights.data_ptr()) % 16 == 0)
      << "gemm2_weights data pointer must be 16-byte aligned for TMA";

  const int64_t m = hidden_states.size(0);
  const int64_t packed_k = hidden_states.size(1);
  TVM_FFI_ICHECK(m > 0) << "hidden_states must contain at least one token";
  TVM_FFI_ICHECK(packed_k > 0 && packed_k <= kIntMax / 2)
      << "hidden_states packed K extent is out of range: " << packed_k;
  const int64_t k = packed_k * 2;
  TVM_FFI_ICHECK(k >= kUpBlockK && k % kUpBlockK == 0)
      << "logical K (" << k << ") must be at least " << kUpBlockK << " and divisible by "
      << kUpBlockK;

  const int64_t num_experts = gemm1_weights.size(0);
  const int64_t n = gemm1_weights.size(1);
  TVM_FFI_ICHECK(num_experts > 0 && num_experts <= kIntMax)
      << "num_experts must be positive and fit in int32, got " << num_experts;
  TVM_FFI_ICHECK(n >= kRowsPerIntermediateBlock && n % kRowsPerIntermediateBlock == 0)
      << "gemm1_weights.shape[1] (" << n << ") must be at least " << kRowsPerIntermediateBlock
      << " and divisible by " << kRowsPerIntermediateBlock;
  TVM_FFI_ICHECK(n <= kIntMax) << "N must fit in int32, got " << n;
  TVM_FFI_ICHECK(gemm1_weights.size(2) == packed_k)
      << "gemm1_weights.shape[2] (" << gemm1_weights.size(2)
      << ") must equal hidden_states.shape[1] (" << packed_k << ")";
  const int64_t intermediate = n / 2;

  TVM_FFI_ICHECK(top_k > 0 && top_k <= num_experts && top_k <= kIntMax)
      << "top_k must be positive, no larger than num_experts, and fit in int32; got " << top_k;
  TVM_FFI_ICHECK(block_m >= kRouteSubtile && block_m % kRouteSubtile == 0 && block_m <= kIntMax)
      << "block_m must be a positive multiple of " << kRouteSubtile << ", got " << block_m;
  TVM_FFI_ICHECK(std::isfinite(routed_scaling_factor))
      << "routed_scaling_factor must be finite, got " << routed_scaling_factor;

  CheckShape2(hidden_states_scale, m, k / 16, "hidden_states_scale");
  CheckShape3(gemm1_weights_scale, num_experts, n, k / 16, "gemm1_weights_scale");
  CheckShape3(gemm2_weights, num_experts, k, intermediate / 2, "gemm2_weights");
  CheckShape3(gemm2_weights_scale, num_experts, k, intermediate / 16, "gemm2_weights_scale");
  CheckShape2(topk_weights, m, top_k, "topk_weights");
  CheckShape2(out, m, k, "out");
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0)
      << "out data pointer must be 16-byte aligned for cp.reduce.async.bulk";

  CheckNoOverlap(out, hidden_states, "hidden_states");
  CheckNoOverlap(out, hidden_states_scale, "hidden_states_scale");
  CheckNoOverlap(out, gemm1_weights, "gemm1_weights");
  CheckNoOverlap(out, gemm1_weights_scale, "gemm1_weights_scale");
  CheckNoOverlap(out, gemm2_weights, "gemm2_weights");
  CheckNoOverlap(out, gemm2_weights_scale, "gemm2_weights_scale");
  CheckNoOverlap(out, sorted_token_ids, "sorted_token_ids");
  CheckNoOverlap(out, expert_ids, "expert_ids");
  CheckNoOverlap(out, num_tokens_post_padded, "num_tokens_post_padded");
  CheckNoOverlap(out, topk_weights, "topk_weights");

  TVM_FFI_ICHECK(num_tokens_post_padded.numel() == 1)
      << "num_tokens_post_padded must contain exactly one device-side int32 value";
  const int64_t num_route_blocks = expert_ids.numel();
  TVM_FFI_ICHECK(num_route_blocks > 0) << "expert_ids must not be empty";
  TVM_FFI_ICHECK(num_route_blocks <= kIntMax / block_m)
      << "expert_ids.numel() * block_m exceeds signed int indexing";
  const int64_t required_plan_capacity = num_route_blocks * block_m;
  TVM_FFI_ICHECK(sorted_token_ids.numel() >= required_plan_capacity)
      << "sorted_token_ids capacity (" << sorted_token_ids.numel()
      << ") must be at least expert_ids.numel() * block_m (" << required_plan_capacity << ")";

  TVM_FFI_ICHECK(m <= kIntMax / top_k) << "M * top_k exceeds the signed int routing sentinel range";
  TVM_FFI_ICHECK(m <= kIntMax / k) << "M * K exceeds signed int indexing";
  CheckIntIndexExtent(m, "M");
  CheckIntIndexExtent(k, "K");
  CheckIntIndexExtent(m * k, "M * K output extent");
  CheckIntIndexExtent(hidden_states_scale.numel(), "hidden_states_scale.numel()");
  CheckIntIndexExtent(gemm1_weights_scale.numel(), "gemm1_weights_scale.numel()");
  CheckIntIndexExtent(gemm2_weights_scale.numel(), "gemm2_weights_scale.numel()");
  CheckIntIndexExtent(required_plan_capacity, "routing plan capacity");

  const int64_t grid_x = num_route_blocks * (block_m / kRouteSubtile);
  TVM_FFI_ICHECK(grid_x > 0 && grid_x <= kIntMax)
      << "launch grid.x (" << grid_x << ") is out of range";
  const int64_t grid_y = n / kRowsPerIntermediateBlock;
  TVM_FFI_ICHECK(grid_y > 0 && grid_y <= 65535)
      << "launch grid.y (" << grid_y << ") is out of range";

  return ProblemDims{static_cast<int>(m),     static_cast<int>(k),
                     static_cast<int>(n),     static_cast<int>(intermediate),
                     static_cast<int>(top_k), static_cast<int>(block_m),
                     num_route_blocks};
}

inline CUtensorMap EncodeHiddenStatesTma(const TensorView& tensor) {
  uint64_t global_dim[2] = {static_cast<uint64_t>(tensor.size(1)),
                            static_cast<uint64_t>(tensor.size(0))};
  uint64_t global_strides[1] = {static_cast<uint64_t>(tensor.stride(0))};
  uint32_t box_dim[2] = {128, 1};
  uint32_t element_strides[2] = {1, 1};
  CUtensorMap map;
  CheckCu(cuTensorMapEncodeTiled(
              &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, tensor.data_ptr(), global_dim, global_strides,
              box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
              CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled(hidden_states)");
  return map;
}

inline CUtensorMap EncodeGemm1WeightsTma(const TensorView& tensor) {
  const uint64_t num_experts = static_cast<uint64_t>(tensor.size(0));
  const uint64_t n = static_cast<uint64_t>(tensor.size(1));
  const uint64_t packed_k = static_cast<uint64_t>(tensor.size(2));
  uint64_t global_dim[4] = {128, n, packed_k / 128, num_experts};
  uint64_t global_strides[3] = {packed_k, 128, n * packed_k};
  uint32_t box_dim[4] = {128, 128, 1, 1};
  uint32_t element_strides[4] = {1, 1, 1, 1};
  CUtensorMap map;
  CheckCu(cuTensorMapEncodeTiled(
              &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, tensor.data_ptr(), global_dim, global_strides,
              box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
              CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled(gemm1_weights)");
  return map;
}

inline CUtensorMap EncodeGemm2WeightsTma(const TensorView& tensor) {
  const uint64_t num_experts = static_cast<uint64_t>(tensor.size(0));
  const uint64_t k = static_cast<uint64_t>(tensor.size(1));
  const uint64_t packed_intermediate = static_cast<uint64_t>(tensor.size(2));
  uint64_t global_dim[3] = {packed_intermediate, k, num_experts};
  uint64_t global_strides[2] = {packed_intermediate, k * packed_intermediate};
  uint32_t box_dim[3] = {64, 128, 1};
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap map;
  CheckCu(cuTensorMapEncodeTiled(
              &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, tensor.data_ptr(), global_dim, global_strides,
              box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
              CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled(gemm2_weights)");
  return map;
}

inline LoomTensorMapPack<3> MakeTensorMapPack(const CUtensorMap& hidden_states_map,
                                              const CUtensorMap& gemm1_map,
                                              const CUtensorMap& gemm2_map) {
  static_assert(sizeof(LoomTensorMap) == sizeof(CUtensorMap),
                "Loom and CUDA tensor-map descriptors must have identical size");
  static_assert(sizeof(LoomTensorMapPack<3>) == 3 * sizeof(LoomTensorMap),
                "generated tensor-map pack must contain three dense descriptors");
  LoomTensorMapPack<3> pack{};
  std::memcpy(&pack.maps[0], &hidden_states_map, sizeof(CUtensorMap));
  std::memcpy(&pack.maps[1], &gemm1_map, sizeof(CUtensorMap));
  std::memcpy(&pack.maps[2], &gemm2_map, sizeof(CUtensorMap));
  return pack;
}

inline void Launch(const TensorView& hidden_states, const TensorView& hidden_states_scale,
                   const TensorView& gemm1_weights, const TensorView& gemm1_weights_scale,
                   const TensorView& gemm2_weights, const TensorView& gemm2_weights_scale,
                   const TensorView& sorted_token_ids, const TensorView& expert_ids,
                   const TensorView& num_tokens_post_padded, const TensorView& topk_weights,
                   const TensorView& out, const ProblemDims& dims, float routed_scaling_factor,
                   cudaStream_t stream) {
  const CUtensorMap hidden_states_map = EncodeHiddenStatesTma(hidden_states);
  const CUtensorMap gemm1_map = EncodeGemm1WeightsTma(gemm1_weights);
  const CUtensorMap gemm2_map = EncodeGemm2WeightsTma(gemm2_weights);
  const LoomTensorMapPack<3> tensor_maps =
      MakeTensorMapPack(hidden_states_map, gemm1_map, gemm2_map);

  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_nvfp4_up_down,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemTotal),
            "cudaFuncSetAttribute(alphamoe_nvfp4_sm100 dynamic smem)");

  const dim3 grid(static_cast<unsigned int>(dims.num_route_blocks * (dims.block_m / kRouteSubtile)),
                  static_cast<unsigned int>(dims.n / kRowsPerIntermediateBlock), 1);
  const dim3 block(kThreads, 1, 1);
  kernel_alpha_moe_nvfp4_up_down<<<grid, block, kSmemTotal, stream>>>(
      static_cast<uint8_t*>(hidden_states_scale.data_ptr()),
      static_cast<uint8_t*>(gemm1_weights_scale.data_ptr()),
      static_cast<uint8_t*>(gemm2_weights_scale.data_ptr()),
      static_cast<int*>(sorted_token_ids.data_ptr()), static_cast<int*>(expert_ids.data_ptr()),
      static_cast<int*>(num_tokens_post_padded.data_ptr()),
      static_cast<float*>(topk_weights.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), dims.m, dims.k, dims.top_k, dims.block_m,
      routed_scaling_factor, tensor_maps);
  CheckCuda(cudaGetLastError(), "alphamoe_nvfp4_sm100 kernel launch");
}

void Run(TensorView hidden_states, TensorView hidden_states_scale, TensorView gemm1_weights,
         TensorView gemm1_weights_scale, TensorView gemm2_weights, TensorView gemm2_weights_scale,
         TensorView sorted_token_ids, TensorView expert_ids, TensorView num_tokens_post_padded,
         TensorView topk_weights, TensorView out, int64_t top_k, int64_t block_m,
         double routed_scaling_factor) {
  // Establish the active CUDA device before capability queries, TMA encoding,
  // function-attribute updates, stream lookup, or kernel launch.
  TVM_FFI_ICHECK(hidden_states.device().device_type == kDLCUDA)
      << "hidden_states must be a CUDA tensor";
  ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);

  const ProblemDims dims =
      CheckInputs(hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                  gemm2_weights, gemm2_weights_scale, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, topk_weights, out, top_k, block_m, routed_scaling_factor);
  const cudaStream_t stream = get_stream(hidden_states.device());
  Launch(hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights,
         gemm2_weights_scale, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights,
         out, dims, static_cast<float>(routed_scaling_factor), stream);
}

}  // namespace alphamoe_nvfp4_sm100
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_aligned_moe_op, flashinfer::alphamoe_nvfp4_sm100::Run);
