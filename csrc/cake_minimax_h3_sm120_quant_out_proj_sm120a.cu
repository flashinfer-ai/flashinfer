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
// clang-format off
// MiniMax-H3 quantized attention output projection with the fused indexed gate + residual for
// SM120 (GB202: RTX 5090 / RTX PRO 6000 Blackwell), generated from the Cake kernel schedules.
// One launch per call (out_proj_gemm_fused_{fp8,nvfp4}): a persistent 256x128 GEMM (7168 -> 5376,
// mma.sync e4m3 / kind::mxf4nvf4 block-scaled, 3-stage 64-byte-K TMA ring, one 384-thread CTA per
// SM).  Warps 1-3 of the producer warpgroup quantize the BF16 attention output rows [M, 7168] in
// place of a separate kernel (per-token E4M3, scale = amax / 448, or block-16 NVFP4 with FlashInfer
// fp4_quantize semantics), counting finished rows per 256-row M tile in a zeroed flag array that the
// TMA producer acquires before it loads a tile; the MMA warps run the fused
// dequant -> BF16 -> out = BF16(residual + BF16(gate[gate_index] * o)) epilogue writing [M, 5376].
// Device code: TMA, ldmatrix, mma.sync (kind::f8f6f4 / kind::mxf4nvf4), mbarrier pipelines,
// setmaxnreg, gpu-scope release/acquire flags.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

static_assert(sizeof(CUtensorMap) == 128, "CUDA tensor-map ABI size mismatch");
static_assert(alignof(CUtensorMap) >= 64, "CUDA tensor-map ABI requires at least 64-byte alignment");

#include <math_constants.h>

namespace h3_out_proj_gemm_fp8_sm120a {

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
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_AB_PIPE_STAGES 3
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 16384
#define SMEM_A_STAGE_STRIDE 16384
#define SMEM_B_STAGE_OFF 50176
#define SMEM_B_STAGE_STAGE_BYTES 8192
#define SMEM_B_STAGE_STRIDE 8192
#define SMEM_STAGING_OFF 74752
#define SMEM_STAGING_STAGE_BYTES 8192
#define SMEM_STAGING_STRIDE 8192
#define SMEM_SFA_SLOT_OFF 82944
#define SMEM_SFA_SLOT_STAGE_BYTES 4096
#define SMEM_SFA_SLOT_STRIDE 4096
#define SMEM_SFB_STAGE_OFF 91136
#define SMEM_SFB_STAGE_STAGE_BYTES 1024
#define SMEM_SFB_STAGE_STRIDE 1024
#define SMEM_TOTAL 94208
#define THREADS 384
#define GROUP_M 1
#define LAUNCH_MIN_BLOCKS 1


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





__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
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


__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_h3_out_proj_gemm_fused(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, unsigned int* __restrict__ attn_out_w, unsigned int* __restrict__ act_q_w, uint8_t* __restrict__ act_sf_w, float* __restrict__ act_scale, float* __restrict__ act_global_scale, float* __restrict__ w_scale, __nv_bfloat16* __restrict__ gate, int* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, unsigned int* __restrict__ out, unsigned int* __restrict__ flags, int M, int num_m_tiles, int total_tiles, float alpha)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_empty_addr (mbar_base + 24)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int B_stage_addr = smem + 50176;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int staging_addr = smem + 74752;
    unsigned int* SFA_slot = reinterpret_cast<unsigned int*>(smem_raw + 82944);
    const int SFA_slot_addr = smem + 82944;
    unsigned int* SFB_stage = reinterpret_cast<unsigned int*>(smem_raw + 91136);
    const int SFB_stage_addr = smem + 91136;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // ab_empty: 3 barriers, init_count=8
            mbarrier_init(smem + 24, 8);
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Warpgroup: 0 ----
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        // ---- Role: producer ----
        if (warp == 0) {
            { // producer_main
                unsigned int load_stage = 0;
                unsigned int _phase_ab_empty = 1;
                #pragma unroll 1
                for (int tile = bid; tile < total_tiles; tile += num_bids) {
                    int tile_m = tile / (GROUP_M * 42) * GROUP_M + (tile - tile / (GROUP_M * 42) * (GROUP_M * 42)) % ((num_m_tiles - tile / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 42) * GROUP_M : GROUP_M);
                    int tile_n = (tile - tile / (GROUP_M * 42) * (GROUP_M * 42)) / ((num_m_tiles - tile / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 42) * GROUP_M : GROUP_M);
                    int rows_left = M - tile_m * 256;
                    unsigned int rows_ready = ((rows_left < 256) ? rows_left : 256);
                    if (elect_sync()) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(flags) + (tile_m))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(rows_ready)) >= static_cast<unsigned int>(1));
                        }
                        asm volatile("fence.proxy.async;" ::: "memory");
                        #pragma unroll 1
                        for (int kp = 0; kp < 56; kp++) {
                            mbarrier_wait_hint(ab_empty_addr + (load_stage) * 8, _phase_ab_empty, 10000000);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 24576);
                            tma_2d_gmem2smem(A_stage_addr + load_stage * 16384, (&A), kp * 2 * 64, tile_m * 256, ab_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(B_stage_addr + load_stage * 8192, (&B), kp * 2 * 64, tile_n * 128, ab_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_ab_empty ^= 1; }
                            mbarrier_wait_hint(ab_empty_addr + (load_stage) * 8, _phase_ab_empty, 10000000);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 24576);
                            tma_2d_gmem2smem(A_stage_addr + load_stage * 16384, (&A), (kp * 2 + 1) * 64, tile_m * 256, ab_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(B_stage_addr + load_stage * 8192, (&B), (kp * 2 + 1) * 64, tile_n * 128, ab_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_ab_empty ^= 1; }
                        }
                    }
                }
            }
        // ---- Role: quant ----
        } else if (warp >= 1 && warp <= 3) {
            { // quant_main
                unsigned int qw[16];
                float qx[16];
                float qacc[4];
                unsigned int qwords[4];
                float global_scale = act_global_scale[0];
                int warp_id_in_role = warp - 1;
                int qrow0 = bid * 3 + warp_id_in_role;
                #pragma unroll 1
                for (int row = qrow0; row < M; row += num_bids * 3) {
                    qacc[0] = 0.0f;
                    #pragma unroll 2
                    for (int j = 0; j < 7; j++) {
                        {
                            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j * 2)) * 8) + 0);
                            qw[0 + 0] = _uv4_0.x;
                            qw[0 + 1] = _uv4_0.y;
                            qw[0 + 2] = _uv4_0.z;
                            qw[0 + 3] = _uv4_0.w;
                        }
                        {
                            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j * 2)) * 8 + 4) + 0);
                            qw[4 + 0] = _uv4_1.x;
                            qw[4 + 1] = _uv4_1.y;
                            qw[4 + 2] = _uv4_1.z;
                            qw[4 + 3] = _uv4_1.w;
                        }
                        {
                            uint4 _uv4_2 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j * 2 + 1)) * 8) + 0);
                            qw[8 + 0] = _uv4_2.x;
                            qw[8 + 1] = _uv4_2.y;
                            qw[8 + 2] = _uv4_2.z;
                            qw[8 + 3] = _uv4_2.w;
                        }
                        {
                            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j * 2 + 1)) * 8 + 4) + 0);
                            qw[12 + 0] = _uv4_3.x;
                            qw[12 + 1] = _uv4_3.y;
                            qw[12 + 2] = _uv4_3.z;
                            qw[12 + 3] = _uv4_3.w;
                        }
                        float _unpack_bf16_0[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) << 16);
                            (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) & 0xffff0000u);
                        }
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[0]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[1]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[2]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[3]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[4]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[5]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[6]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[7]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[8]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[9]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[10]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[11]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[12]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[13]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[14]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[15]));
                        float _unpack_bf16_1[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) << 16);
                            (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) & 0xffff0000u);
                        }
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[0]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[1]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[2]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[3]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[4]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[5]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[6]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[7]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[8]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[9]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[10]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[11]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[12]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[13]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[14]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[15]));
                    }
                    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, qacc[0], 16);
                    qacc[0] = fmaxf(qacc[0], _shfl_xor_sync_4);
                    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, qacc[0], 8);
                    qacc[0] = fmaxf(qacc[0], _shfl_xor_sync_5);
                    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, qacc[0], 4);
                    qacc[0] = fmaxf(qacc[0], _shfl_xor_sync_6);
                    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, qacc[0], 2);
                    qacc[0] = fmaxf(qacc[0], _shfl_xor_sync_7);
                    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, qacc[0], 1);
                    qacc[0] = fmaxf(qacc[0], _shfl_xor_sync_8);
                    float _fdiv_9 = __fdiv_rn(fmaxf(qacc[0], 1e-12f), 448.0f);
                    qacc[1] = _fdiv_9;
                    if (elect_sync()) {
                        *(reinterpret_cast<float*>(act_scale + row) + (0)) = qacc[1];
                    }
                    #pragma unroll 2
                    for (int j_1 = 0; j_1 < 7; j_1++) {
                        {
                            uint4 _uv4_10 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j_1 * 2)) * 8) + 0);
                            qw[0 + 0] = _uv4_10.x;
                            qw[0 + 1] = _uv4_10.y;
                            qw[0 + 2] = _uv4_10.z;
                            qw[0 + 3] = _uv4_10.w;
                        }
                        {
                            uint4 _uv4_11 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j_1 * 2)) * 8 + 4) + 0);
                            qw[4 + 0] = _uv4_11.x;
                            qw[4 + 1] = _uv4_11.y;
                            qw[4 + 2] = _uv4_11.z;
                            qw[4 + 3] = _uv4_11.w;
                        }
                        {
                            uint4 _uv4_12 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j_1 * 2 + 1)) * 8) + 0);
                            qw[8 + 0] = _uv4_12.x;
                            qw[8 + 1] = _uv4_12.y;
                            qw[8 + 2] = _uv4_12.z;
                            qw[8 + 3] = _uv4_12.w;
                        }
                        {
                            uint4 _uv4_13 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j_1 * 2 + 1)) * 8 + 4) + 0);
                            qw[12 + 0] = _uv4_13.x;
                            qw[12 + 1] = _uv4_13.y;
                            qw[12 + 2] = _uv4_13.z;
                            qw[12 + 3] = _uv4_13.w;
                        }
                        float _unpack_bf16_2[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) << 16);
                            (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) & 0xffff0000u);
                        }
                        float _fdiv_16 = __fdiv_rn(_unpack_bf16_2[1], qacc[1]);
                        float _fdiv_17 = __fdiv_rn(_unpack_bf16_2[0], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_15;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_15) : "f"(_fdiv_16), "f"(_fdiv_17));
                        float _fdiv_19 = __fdiv_rn(_unpack_bf16_2[3], qacc[1]);
                        float _fdiv_20 = __fdiv_rn(_unpack_bf16_2[2], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_18;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(_fdiv_19), "f"(_fdiv_20));
                        uint32_t _mov_b32_pack_u16x2_14;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_14) : "h"(_cvt_satfinite_e4m3x2_f32_15), "h"(_cvt_satfinite_e4m3x2_f32_18));
                        qwords[0] = _mov_b32_pack_u16x2_14;
                        float _fdiv_23 = __fdiv_rn(_unpack_bf16_2[5], qacc[1]);
                        float _fdiv_24 = __fdiv_rn(_unpack_bf16_2[4], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_22;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_22) : "f"(_fdiv_23), "f"(_fdiv_24));
                        float _fdiv_26 = __fdiv_rn(_unpack_bf16_2[7], qacc[1]);
                        float _fdiv_27 = __fdiv_rn(_unpack_bf16_2[6], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_25;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_25) : "f"(_fdiv_26), "f"(_fdiv_27));
                        uint32_t _mov_b32_pack_u16x2_21;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_21) : "h"(_cvt_satfinite_e4m3x2_f32_22), "h"(_cvt_satfinite_e4m3x2_f32_25));
                        qwords[1] = _mov_b32_pack_u16x2_21;
                        float _fdiv_30 = __fdiv_rn(_unpack_bf16_2[9], qacc[1]);
                        float _fdiv_31 = __fdiv_rn(_unpack_bf16_2[8], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_29;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_29) : "f"(_fdiv_30), "f"(_fdiv_31));
                        float _fdiv_33 = __fdiv_rn(_unpack_bf16_2[11], qacc[1]);
                        float _fdiv_34 = __fdiv_rn(_unpack_bf16_2[10], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_32;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_32) : "f"(_fdiv_33), "f"(_fdiv_34));
                        uint32_t _mov_b32_pack_u16x2_28;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_28) : "h"(_cvt_satfinite_e4m3x2_f32_29), "h"(_cvt_satfinite_e4m3x2_f32_32));
                        qwords[2] = _mov_b32_pack_u16x2_28;
                        float _fdiv_37 = __fdiv_rn(_unpack_bf16_2[13], qacc[1]);
                        float _fdiv_38 = __fdiv_rn(_unpack_bf16_2[12], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_36;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_36) : "f"(_fdiv_37), "f"(_fdiv_38));
                        float _fdiv_40 = __fdiv_rn(_unpack_bf16_2[15], qacc[1]);
                        float _fdiv_41 = __fdiv_rn(_unpack_bf16_2[14], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_39;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_39) : "f"(_fdiv_40), "f"(_fdiv_41));
                        uint32_t _mov_b32_pack_u16x2_35;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_35) : "h"(_cvt_satfinite_e4m3x2_f32_36), "h"(_cvt_satfinite_e4m3x2_f32_39));
                        qwords[3] = _mov_b32_pack_u16x2_35;
                        reinterpret_cast<int4*>(act_q_w + ((row * 7168 + (lane + 32 * (j_1 * 2)) * 16) / 4))[0] = reinterpret_cast<int4*>(qwords)[0];
                        float _unpack_bf16_3[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) << 16);
                            (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) & 0xffff0000u);
                        }
                        float _fdiv_44 = __fdiv_rn(_unpack_bf16_3[1], qacc[1]);
                        float _fdiv_45 = __fdiv_rn(_unpack_bf16_3[0], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_43;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_43) : "f"(_fdiv_44), "f"(_fdiv_45));
                        float _fdiv_47 = __fdiv_rn(_unpack_bf16_3[3], qacc[1]);
                        float _fdiv_48 = __fdiv_rn(_unpack_bf16_3[2], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_46;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_46) : "f"(_fdiv_47), "f"(_fdiv_48));
                        uint32_t _mov_b32_pack_u16x2_42;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_42) : "h"(_cvt_satfinite_e4m3x2_f32_43), "h"(_cvt_satfinite_e4m3x2_f32_46));
                        qwords[0] = _mov_b32_pack_u16x2_42;
                        float _fdiv_51 = __fdiv_rn(_unpack_bf16_3[5], qacc[1]);
                        float _fdiv_52 = __fdiv_rn(_unpack_bf16_3[4], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_50;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_50) : "f"(_fdiv_51), "f"(_fdiv_52));
                        float _fdiv_54 = __fdiv_rn(_unpack_bf16_3[7], qacc[1]);
                        float _fdiv_55 = __fdiv_rn(_unpack_bf16_3[6], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_53;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_53) : "f"(_fdiv_54), "f"(_fdiv_55));
                        uint32_t _mov_b32_pack_u16x2_49;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_49) : "h"(_cvt_satfinite_e4m3x2_f32_50), "h"(_cvt_satfinite_e4m3x2_f32_53));
                        qwords[1] = _mov_b32_pack_u16x2_49;
                        float _fdiv_58 = __fdiv_rn(_unpack_bf16_3[9], qacc[1]);
                        float _fdiv_59 = __fdiv_rn(_unpack_bf16_3[8], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_57;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_57) : "f"(_fdiv_58), "f"(_fdiv_59));
                        float _fdiv_61 = __fdiv_rn(_unpack_bf16_3[11], qacc[1]);
                        float _fdiv_62 = __fdiv_rn(_unpack_bf16_3[10], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_60;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_60) : "f"(_fdiv_61), "f"(_fdiv_62));
                        uint32_t _mov_b32_pack_u16x2_56;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_56) : "h"(_cvt_satfinite_e4m3x2_f32_57), "h"(_cvt_satfinite_e4m3x2_f32_60));
                        qwords[2] = _mov_b32_pack_u16x2_56;
                        float _fdiv_65 = __fdiv_rn(_unpack_bf16_3[13], qacc[1]);
                        float _fdiv_66 = __fdiv_rn(_unpack_bf16_3[12], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_64;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_64) : "f"(_fdiv_65), "f"(_fdiv_66));
                        float _fdiv_68 = __fdiv_rn(_unpack_bf16_3[15], qacc[1]);
                        float _fdiv_69 = __fdiv_rn(_unpack_bf16_3[14], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_67;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_67) : "f"(_fdiv_68), "f"(_fdiv_69));
                        uint32_t _mov_b32_pack_u16x2_63;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_63) : "h"(_cvt_satfinite_e4m3x2_f32_64), "h"(_cvt_satfinite_e4m3x2_f32_67));
                        qwords[3] = _mov_b32_pack_u16x2_63;
                        reinterpret_cast<int4*>(act_q_w + ((row * 7168 + (lane + 32 * (j_1 * 2 + 1)) * 16) / 4))[0] = reinterpret_cast<int4*>(qwords)[0];
                    }
                    __threadfence();
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(flags) + (row / 256))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // mma_main
            unsigned int mma_stage = 0;
            int warp_id_in_role_1 = warp - 4;
            int warp_m = warp_id_in_role_1 % 4;
            int warp_n = warp_id_in_role_1 / 4;
            int role_tid = warp_id_in_role_1 * 32 + lane;
            float accum[128];
            unsigned int a_frag[16];
            unsigned int b_frag[16];
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / (GROUP_M * 42) * GROUP_M + (tile_1 - tile_1 / (GROUP_M * 42) * (GROUP_M * 42)) % ((num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M : GROUP_M);
                int tile_n_1 = (tile_1 - tile_1 / (GROUP_M * 42) * (GROUP_M * 42)) / ((num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M : GROUP_M);
                accum[0] = 0.0f;
                accum[1] = 0.0f;
                accum[2] = 0.0f;
                accum[3] = 0.0f;
                accum[4] = 0.0f;
                accum[5] = 0.0f;
                accum[6] = 0.0f;
                accum[7] = 0.0f;
                accum[8] = 0.0f;
                accum[9] = 0.0f;
                accum[10] = 0.0f;
                accum[11] = 0.0f;
                accum[12] = 0.0f;
                accum[13] = 0.0f;
                accum[14] = 0.0f;
                accum[15] = 0.0f;
                accum[16] = 0.0f;
                accum[17] = 0.0f;
                accum[18] = 0.0f;
                accum[19] = 0.0f;
                accum[20] = 0.0f;
                accum[21] = 0.0f;
                accum[22] = 0.0f;
                accum[23] = 0.0f;
                accum[24] = 0.0f;
                accum[25] = 0.0f;
                accum[26] = 0.0f;
                accum[27] = 0.0f;
                accum[28] = 0.0f;
                accum[29] = 0.0f;
                accum[30] = 0.0f;
                accum[31] = 0.0f;
                accum[32] = 0.0f;
                accum[33] = 0.0f;
                accum[34] = 0.0f;
                accum[35] = 0.0f;
                accum[36] = 0.0f;
                accum[37] = 0.0f;
                accum[38] = 0.0f;
                accum[39] = 0.0f;
                accum[40] = 0.0f;
                accum[41] = 0.0f;
                accum[42] = 0.0f;
                accum[43] = 0.0f;
                accum[44] = 0.0f;
                accum[45] = 0.0f;
                accum[46] = 0.0f;
                accum[47] = 0.0f;
                accum[48] = 0.0f;
                accum[49] = 0.0f;
                accum[50] = 0.0f;
                accum[51] = 0.0f;
                accum[52] = 0.0f;
                accum[53] = 0.0f;
                accum[54] = 0.0f;
                accum[55] = 0.0f;
                accum[56] = 0.0f;
                accum[57] = 0.0f;
                accum[58] = 0.0f;
                accum[59] = 0.0f;
                accum[60] = 0.0f;
                accum[61] = 0.0f;
                accum[62] = 0.0f;
                accum[63] = 0.0f;
                accum[64] = 0.0f;
                accum[65] = 0.0f;
                accum[66] = 0.0f;
                accum[67] = 0.0f;
                accum[68] = 0.0f;
                accum[69] = 0.0f;
                accum[70] = 0.0f;
                accum[71] = 0.0f;
                accum[72] = 0.0f;
                accum[73] = 0.0f;
                accum[74] = 0.0f;
                accum[75] = 0.0f;
                accum[76] = 0.0f;
                accum[77] = 0.0f;
                accum[78] = 0.0f;
                accum[79] = 0.0f;
                accum[80] = 0.0f;
                accum[81] = 0.0f;
                accum[82] = 0.0f;
                accum[83] = 0.0f;
                accum[84] = 0.0f;
                accum[85] = 0.0f;
                accum[86] = 0.0f;
                accum[87] = 0.0f;
                accum[88] = 0.0f;
                accum[89] = 0.0f;
                accum[90] = 0.0f;
                accum[91] = 0.0f;
                accum[92] = 0.0f;
                accum[93] = 0.0f;
                accum[94] = 0.0f;
                accum[95] = 0.0f;
                accum[96] = 0.0f;
                accum[97] = 0.0f;
                accum[98] = 0.0f;
                accum[99] = 0.0f;
                accum[100] = 0.0f;
                accum[101] = 0.0f;
                accum[102] = 0.0f;
                accum[103] = 0.0f;
                accum[104] = 0.0f;
                accum[105] = 0.0f;
                accum[106] = 0.0f;
                accum[107] = 0.0f;
                accum[108] = 0.0f;
                accum[109] = 0.0f;
                accum[110] = 0.0f;
                accum[111] = 0.0f;
                accum[112] = 0.0f;
                accum[113] = 0.0f;
                accum[114] = 0.0f;
                accum[115] = 0.0f;
                accum[116] = 0.0f;
                accum[117] = 0.0f;
                accum[118] = 0.0f;
                accum[119] = 0.0f;
                accum[120] = 0.0f;
                accum[121] = 0.0f;
                accum[122] = 0.0f;
                accum[123] = 0.0f;
                accum[124] = 0.0f;
                accum[125] = 0.0f;
                accum[126] = 0.0f;
                accum[127] = 0.0f;
                int col_base = tile_n_1 * 128 + warp_n * 64 + (lane & 3) * 2;
                float sw[16];
                sw[0] = w_scale[col_base];
                sw[1] = w_scale[col_base + 1];
                sw[2] = w_scale[col_base + 8];
                sw[3] = w_scale[col_base + 8 + 1];
                sw[4] = w_scale[col_base + 16];
                sw[5] = w_scale[col_base + 16 + 1];
                sw[6] = w_scale[col_base + 24];
                sw[7] = w_scale[col_base + 24 + 1];
                sw[8] = w_scale[col_base + 32];
                sw[9] = w_scale[col_base + 32 + 1];
                sw[10] = w_scale[col_base + 40];
                sw[11] = w_scale[col_base + 40 + 1];
                sw[12] = w_scale[col_base + 48];
                sw[13] = w_scale[col_base + 48 + 1];
                sw[14] = w_scale[col_base + 56];
                sw[15] = w_scale[col_base + 56 + 1];
                float sa[8];
                #pragma unroll 1
                for (int kp_1 = 0; kp_1 < 56; kp_1++) {
                    mbarrier_wait_hint(ab_full_addr + (mma_stage) * 8, _phase_ab_full, 10000000);
                    if (kp_1 == 52) {
                        int arow = tile_m_1 * 256 + warp_m * 64 + (lane >> 2);
                        sa[0] = ((arow < M) ? act_scale[arow] : 0.0f);
                        int arow_0 = tile_m_1 * 256 + warp_m * 64 + 8 + (lane >> 2);
                        sa[1] = ((arow_0 < M) ? act_scale[arow_0] : 0.0f);
                        int arow_1 = tile_m_1 * 256 + warp_m * 64 + 16 + (lane >> 2);
                        sa[2] = ((arow_1 < M) ? act_scale[arow_1] : 0.0f);
                        int arow_2 = tile_m_1 * 256 + warp_m * 64 + 16 + 8 + (lane >> 2);
                        sa[3] = ((arow_2 < M) ? act_scale[arow_2] : 0.0f);
                        int arow_3 = tile_m_1 * 256 + warp_m * 64 + 32 + (lane >> 2);
                        sa[4] = ((arow_3 < M) ? act_scale[arow_3] : 0.0f);
                        int arow_4 = tile_m_1 * 256 + warp_m * 64 + 32 + 8 + (lane >> 2);
                        sa[5] = ((arow_4 < M) ? act_scale[arow_4] : 0.0f);
                        int arow_5 = tile_m_1 * 256 + warp_m * 64 + 48 + (lane >> 2);
                        sa[6] = ((arow_5 < M) ? act_scale[arow_5] : 0.0f);
                        int arow_6 = tile_m_1 * 256 + warp_m * 64 + 48 + 8 + (lane >> 2);
                        sa[7] = ((arow_6 < M) ? act_scale[arow_6] : 0.0f);
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_ab_full ^= 1; }
                    mbarrier_wait_hint(ab_full_addr + (mma_stage) * 8, _phase_ab_full, 10000000);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_ab_full ^= 1; }
                }
                if (tile_m_1 > 2147483647) {
                    float chk = 0.0f;
                    chk = chk + accum[0];
                    chk = chk + accum[1];
                    chk = chk + accum[2];
                    chk = chk + accum[3];
                    chk = chk + accum[4];
                    chk = chk + accum[5];
                    chk = chk + accum[6];
                    chk = chk + accum[7];
                    chk = chk + accum[8];
                    chk = chk + accum[9];
                    chk = chk + accum[10];
                    chk = chk + accum[11];
                    chk = chk + accum[12];
                    chk = chk + accum[13];
                    chk = chk + accum[14];
                    chk = chk + accum[15];
                    chk = chk + accum[16];
                    chk = chk + accum[17];
                    chk = chk + accum[18];
                    chk = chk + accum[19];
                    chk = chk + accum[20];
                    chk = chk + accum[21];
                    chk = chk + accum[22];
                    chk = chk + accum[23];
                    chk = chk + accum[24];
                    chk = chk + accum[25];
                    chk = chk + accum[26];
                    chk = chk + accum[27];
                    chk = chk + accum[28];
                    chk = chk + accum[29];
                    chk = chk + accum[30];
                    chk = chk + accum[31];
                    chk = chk + accum[32];
                    chk = chk + accum[33];
                    chk = chk + accum[34];
                    chk = chk + accum[35];
                    chk = chk + accum[36];
                    chk = chk + accum[37];
                    chk = chk + accum[38];
                    chk = chk + accum[39];
                    chk = chk + accum[40];
                    chk = chk + accum[41];
                    chk = chk + accum[42];
                    chk = chk + accum[43];
                    chk = chk + accum[44];
                    chk = chk + accum[45];
                    chk = chk + accum[46];
                    chk = chk + accum[47];
                    chk = chk + accum[48];
                    chk = chk + accum[49];
                    chk = chk + accum[50];
                    chk = chk + accum[51];
                    chk = chk + accum[52];
                    chk = chk + accum[53];
                    chk = chk + accum[54];
                    chk = chk + accum[55];
                    chk = chk + accum[56];
                    chk = chk + accum[57];
                    chk = chk + accum[58];
                    chk = chk + accum[59];
                    chk = chk + accum[60];
                    chk = chk + accum[61];
                    chk = chk + accum[62];
                    chk = chk + accum[63];
                    chk = chk + accum[64];
                    chk = chk + accum[65];
                    chk = chk + accum[66];
                    chk = chk + accum[67];
                    chk = chk + accum[68];
                    chk = chk + accum[69];
                    chk = chk + accum[70];
                    chk = chk + accum[71];
                    chk = chk + accum[72];
                    chk = chk + accum[73];
                    chk = chk + accum[74];
                    chk = chk + accum[75];
                    chk = chk + accum[76];
                    chk = chk + accum[77];
                    chk = chk + accum[78];
                    chk = chk + accum[79];
                    chk = chk + accum[80];
                    chk = chk + accum[81];
                    chk = chk + accum[82];
                    chk = chk + accum[83];
                    chk = chk + accum[84];
                    chk = chk + accum[85];
                    chk = chk + accum[86];
                    chk = chk + accum[87];
                    chk = chk + accum[88];
                    chk = chk + accum[89];
                    chk = chk + accum[90];
                    chk = chk + accum[91];
                    chk = chk + accum[92];
                    chk = chk + accum[93];
                    chk = chk + accum[94];
                    chk = chk + accum[95];
                    chk = chk + accum[96];
                    chk = chk + accum[97];
                    chk = chk + accum[98];
                    chk = chk + accum[99];
                    chk = chk + accum[100];
                    chk = chk + accum[101];
                    chk = chk + accum[102];
                    chk = chk + accum[103];
                    chk = chk + accum[104];
                    chk = chk + accum[105];
                    chk = chk + accum[106];
                    chk = chk + accum[107];
                    chk = chk + accum[108];
                    chk = chk + accum[109];
                    chk = chk + accum[110];
                    chk = chk + accum[111];
                    chk = chk + accum[112];
                    chk = chk + accum[113];
                    chk = chk + accum[114];
                    chk = chk + accum[115];
                    chk = chk + accum[116];
                    chk = chk + accum[117];
                    chk = chk + accum[118];
                    chk = chk + accum[119];
                    chk = chk + accum[120];
                    chk = chk + accum[121];
                    chk = chk + accum[122];
                    chk = chk + accum[123];
                    chk = chk + accum[124];
                    chk = chk + accum[125];
                    chk = chk + accum[126];
                    chk = chk + accum[127];
                    if (lane == 0) {
                        out[tile_1] = __as_u32(chk);
                    }
                }
                int srow_st = warp_m * 8 + (lane & 7);
                int smat = lane >> 3;
                int srow = role_tid >> 3;
                int c = role_tid & 7;
                int col = tile_n_1 * 128 + c * 16;
                int grow_base = tile_m_1 * 256 + (srow >> 3) * 64 + (srow & 7);
                float ops[32];
                int meta[2];
                meta[0] = gate_index[((grow_base < M) ? grow_base : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(residual + (((grow_base < M) ? grow_base : 0) * 5376 + col) + 0);
                    uint4 _vld_2[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_2[_blk] = _vptr_2[_blk];
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(residual + (((grow_base < M) ? grow_base : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_3[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled[16];
                scaled[0] = accum[0] * sa[0] * sw[0];
                scaled[1] = accum[1] * sa[0] * sw[1];
                scaled[2] = accum[4] * sa[0] * sw[2];
                scaled[3] = accum[5] * sa[0] * sw[3];
                scaled[4] = accum[8] * sa[0] * sw[4];
                scaled[5] = accum[9] * sa[0] * sw[5];
                scaled[6] = accum[12] * sa[0] * sw[6];
                scaled[7] = accum[13] * sa[0] * sw[7];
                scaled[8] = accum[16] * sa[0] * sw[8];
                scaled[9] = accum[17] * sa[0] * sw[9];
                scaled[10] = accum[20] * sa[0] * sw[10];
                scaled[11] = accum[21] * sa[0] * sw[11];
                scaled[12] = accum[24] * sa[0] * sw[12];
                scaled[13] = accum[25] * sa[0] * sw[13];
                scaled[14] = accum[28] * sa[0] * sw[14];
                scaled[15] = accum[29] * sa[0] * sw[15];
                unsigned int scaled_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                    scaled_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow = grow_base;
                if (grow < M) {
                    unsigned int o_words[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_4[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words[_pair]) << 16);
                        (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words[_pair]) & 0xffff0000u);
                    }
                    int valid = meta[1];
                    float p_raw[16];
                    for (int e = 0; e < 8; e++) {
                        float gv0 = ops[e];
                        float gv1 = ops[8 + e];
                        float gs0 = ((valid == 1) ? gv0 : 0.0f);
                        float gs1 = ((valid == 1) ? gv1 : 0.0f);
                        p_raw[e] = gs0 * _unpack_bf16_4[e];
                        p_raw[8 + e] = gs1 * _unpack_bf16_4[8 + e];
                    }
                    unsigned int p_raw_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw[_lp*2 + 0], p_raw[_lp*2+1 + 0]));
                        p_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_5[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16[_pair]) << 16);
                        (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16[_pair]) & 0xffff0000u);
                    }
                    float out_raw[16];
                    for (int e_1 = 0; e_1 < 8; e_1++) {
                        float rv0 = ops[16 + e_1];
                        float rv1 = ops[24 + e_1];
                        out_raw[e_1] = rv0 + _unpack_bf16_5[e_1];
                        out_raw[8 + e_1] = rv1 + _unpack_bf16_5[8 + e_1];
                    }
                    unsigned int out_raw_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw[_lp*2 + 0], out_raw[_lp*2+1 + 0]));
                        out_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow < M) {
                        reinterpret_cast<int4*>(out + ((grow * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 8 < M) ? grow_base + 8 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_6[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_6[_blk] = _vptr_6[_blk];
                        uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_7[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_7[_blk] = _vptr_7[_blk];
                        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(residual + (((grow_base + 8 < M) ? grow_base + 8 : 0) * 5376 + col) + 0);
                    uint4 _vld_8[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_8[_blk] = _vptr_8[_blk];
                        uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(residual + (((grow_base + 8 < M) ? grow_base + 8 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_9[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_9[_blk] = _vptr_9[_blk];
                        uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled_0[16];
                scaled_0[0] = accum[2] * sa[1] * sw[0];
                scaled_0[1] = accum[3] * sa[1] * sw[1];
                scaled_0[2] = accum[6] * sa[1] * sw[2];
                scaled_0[3] = accum[7] * sa[1] * sw[3];
                scaled_0[4] = accum[10] * sa[1] * sw[4];
                scaled_0[5] = accum[11] * sa[1] * sw[5];
                scaled_0[6] = accum[14] * sa[1] * sw[6];
                scaled_0[7] = accum[15] * sa[1] * sw[7];
                scaled_0[8] = accum[18] * sa[1] * sw[8];
                scaled_0[9] = accum[19] * sa[1] * sw[9];
                scaled_0[10] = accum[22] * sa[1] * sw[10];
                scaled_0[11] = accum[23] * sa[1] * sw[11];
                scaled_0[12] = accum[26] * sa[1] * sw[12];
                scaled_0[13] = accum[27] * sa[1] * sw[13];
                scaled_0[14] = accum[30] * sa[1] * sw[14];
                scaled_0[15] = accum[31] * sa[1] * sw[15];
                unsigned int scaled_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_0[_lp*2 + 0], scaled_0[_lp*2+1 + 0]));
                    scaled_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow_1 = grow_base + 8;
                if (grow_1 < M) {
                    unsigned int o_words_1[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_6[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words_1[_pair]) << 16);
                        (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words_1[_pair]) & 0xffff0000u);
                    }
                    int valid_1 = meta[1];
                    float p_raw_1[16];
                    for (int e_2 = 0; e_2 < 8; e_2++) {
                        float gv0_1 = ops[e_2];
                        float gv1_1 = ops[8 + e_2];
                        float gs0_1 = ((valid_1 == 1) ? gv0_1 : 0.0f);
                        float gs1_1 = ((valid_1 == 1) ? gv1_1 : 0.0f);
                        p_raw_1[e_2] = gs0_1 * _unpack_bf16_6[e_2];
                        p_raw_1[8 + e_2] = gs1_1 * _unpack_bf16_6[8 + e_2];
                    }
                    unsigned int p_raw_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_1[_lp*2 + 0], p_raw_1[_lp*2+1 + 0]));
                        p_raw_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_7[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_1[_pair]) << 16);
                        (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_1[_pair]) & 0xffff0000u);
                    }
                    float out_raw_1[16];
                    for (int e_3 = 0; e_3 < 8; e_3++) {
                        float rv0_1 = ops[16 + e_3];
                        float rv1_1 = ops[24 + e_3];
                        out_raw_1[e_3] = rv0_1 + _unpack_bf16_7[e_3];
                        out_raw_1[8 + e_3] = rv1_1 + _unpack_bf16_7[8 + e_3];
                    }
                    unsigned int out_raw_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_1[_lp*2 + 0], out_raw_1[_lp*2+1 + 0]));
                        out_raw_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_1 < M) {
                        reinterpret_cast<int4*>(out + ((grow_1 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_1 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_1 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_1 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 16 < M) ? grow_base + 16 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_12 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_12[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_12[_blk] = _vptr_12[_blk];
                        uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_13 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_13[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_13[_blk] = _vptr_13[_blk];
                        uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_14 = reinterpret_cast<const uint4*>(residual + (((grow_base + 16 < M) ? grow_base + 16 : 0) * 5376 + col) + 0);
                    uint4 _vld_14[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_14[_blk] = _vptr_14[_blk];
                        uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_15 = reinterpret_cast<const uint4*>(residual + (((grow_base + 16 < M) ? grow_base + 16 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_15[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_15[_blk] = _vptr_15[_blk];
                        uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled_2[16];
                scaled_2[0] = accum[32] * sa[2] * sw[0];
                scaled_2[1] = accum[33] * sa[2] * sw[1];
                scaled_2[2] = accum[36] * sa[2] * sw[2];
                scaled_2[3] = accum[37] * sa[2] * sw[3];
                scaled_2[4] = accum[40] * sa[2] * sw[4];
                scaled_2[5] = accum[41] * sa[2] * sw[5];
                scaled_2[6] = accum[44] * sa[2] * sw[6];
                scaled_2[7] = accum[45] * sa[2] * sw[7];
                scaled_2[8] = accum[48] * sa[2] * sw[8];
                scaled_2[9] = accum[49] * sa[2] * sw[9];
                scaled_2[10] = accum[52] * sa[2] * sw[10];
                scaled_2[11] = accum[53] * sa[2] * sw[11];
                scaled_2[12] = accum[56] * sa[2] * sw[12];
                scaled_2[13] = accum[57] * sa[2] * sw[13];
                scaled_2[14] = accum[60] * sa[2] * sw[14];
                scaled_2[15] = accum[61] * sa[2] * sw[15];
                unsigned int scaled_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_2[_lp*2 + 0], scaled_2[_lp*2+1 + 0]));
                    scaled_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow_3 = grow_base + 16;
                if (grow_3 < M) {
                    unsigned int o_words_2[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_8[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words_2[_pair]) << 16);
                        (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words_2[_pair]) & 0xffff0000u);
                    }
                    int valid_2 = meta[1];
                    float p_raw_2[16];
                    for (int e_4 = 0; e_4 < 8; e_4++) {
                        float gv0_2 = ops[e_4];
                        float gv1_2 = ops[8 + e_4];
                        float gs0_2 = ((valid_2 == 1) ? gv0_2 : 0.0f);
                        float gs1_2 = ((valid_2 == 1) ? gv1_2 : 0.0f);
                        p_raw_2[e_4] = gs0_2 * _unpack_bf16_8[e_4];
                        p_raw_2[8 + e_4] = gs1_2 * _unpack_bf16_8[8 + e_4];
                    }
                    unsigned int p_raw_bf16_2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_2[_lp*2 + 0], p_raw_2[_lp*2+1 + 0]));
                        p_raw_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_9[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_2[_pair]) << 16);
                        (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_2[_pair]) & 0xffff0000u);
                    }
                    float out_raw_2[16];
                    for (int e_5 = 0; e_5 < 8; e_5++) {
                        float rv0_2 = ops[16 + e_5];
                        float rv1_2 = ops[24 + e_5];
                        out_raw_2[e_5] = rv0_2 + _unpack_bf16_9[e_5];
                        out_raw_2[8 + e_5] = rv1_2 + _unpack_bf16_9[8 + e_5];
                    }
                    unsigned int out_raw_bf16_2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_2[_lp*2 + 0], out_raw_2[_lp*2+1 + 0]));
                        out_raw_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_3 < M) {
                        reinterpret_cast<int4*>(out + ((grow_3 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_2 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_3 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_2 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 24 < M) ? grow_base + 24 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_18 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_18[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_18[_blk] = _vptr_18[_blk];
                        uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_19 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_19[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_19[_blk] = _vptr_19[_blk];
                        uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_20 = reinterpret_cast<const uint4*>(residual + (((grow_base + 24 < M) ? grow_base + 24 : 0) * 5376 + col) + 0);
                    uint4 _vld_20[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_20[_blk] = _vptr_20[_blk];
                        uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_21 = reinterpret_cast<const uint4*>(residual + (((grow_base + 24 < M) ? grow_base + 24 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_21[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_21[_blk] = _vptr_21[_blk];
                        uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled_4[16];
                scaled_4[0] = accum[34] * sa[3] * sw[0];
                scaled_4[1] = accum[35] * sa[3] * sw[1];
                scaled_4[2] = accum[38] * sa[3] * sw[2];
                scaled_4[3] = accum[39] * sa[3] * sw[3];
                scaled_4[4] = accum[42] * sa[3] * sw[4];
                scaled_4[5] = accum[43] * sa[3] * sw[5];
                scaled_4[6] = accum[46] * sa[3] * sw[6];
                scaled_4[7] = accum[47] * sa[3] * sw[7];
                scaled_4[8] = accum[50] * sa[3] * sw[8];
                scaled_4[9] = accum[51] * sa[3] * sw[9];
                scaled_4[10] = accum[54] * sa[3] * sw[10];
                scaled_4[11] = accum[55] * sa[3] * sw[11];
                scaled_4[12] = accum[58] * sa[3] * sw[12];
                scaled_4[13] = accum[59] * sa[3] * sw[13];
                scaled_4[14] = accum[62] * sa[3] * sw[14];
                scaled_4[15] = accum[63] * sa[3] * sw[15];
                unsigned int scaled_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_4[_lp*2 + 0], scaled_4[_lp*2+1 + 0]));
                    scaled_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow_5 = grow_base + 24;
                if (grow_5 < M) {
                    unsigned int o_words_3[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_10[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words_3[_pair]) << 16);
                        (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words_3[_pair]) & 0xffff0000u);
                    }
                    int valid_3 = meta[1];
                    float p_raw_3[16];
                    for (int e_6 = 0; e_6 < 8; e_6++) {
                        float gv0_3 = ops[e_6];
                        float gv1_3 = ops[8 + e_6];
                        float gs0_3 = ((valid_3 == 1) ? gv0_3 : 0.0f);
                        float gs1_3 = ((valid_3 == 1) ? gv1_3 : 0.0f);
                        p_raw_3[e_6] = gs0_3 * _unpack_bf16_10[e_6];
                        p_raw_3[8 + e_6] = gs1_3 * _unpack_bf16_10[8 + e_6];
                    }
                    unsigned int p_raw_bf16_3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_3[_lp*2 + 0], p_raw_3[_lp*2+1 + 0]));
                        p_raw_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_11[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_3[_pair]) << 16);
                        (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_3[_pair]) & 0xffff0000u);
                    }
                    float out_raw_3[16];
                    for (int e_7 = 0; e_7 < 8; e_7++) {
                        float rv0_3 = ops[16 + e_7];
                        float rv1_3 = ops[24 + e_7];
                        out_raw_3[e_7] = rv0_3 + _unpack_bf16_11[e_7];
                        out_raw_3[8 + e_7] = rv1_3 + _unpack_bf16_11[8 + e_7];
                    }
                    unsigned int out_raw_bf16_3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_3[_lp*2 + 0], out_raw_3[_lp*2+1 + 0]));
                        out_raw_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_5 < M) {
                        reinterpret_cast<int4*>(out + ((grow_5 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_3 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_5 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_3 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 32 < M) ? grow_base + 32 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_24 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_24[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_24[_blk] = _vptr_24[_blk];
                        uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_25 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_25[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_25[_blk] = _vptr_25[_blk];
                        uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_26 = reinterpret_cast<const uint4*>(residual + (((grow_base + 32 < M) ? grow_base + 32 : 0) * 5376 + col) + 0);
                    uint4 _vld_26[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_26[_blk] = _vptr_26[_blk];
                        uint32_t* _vpairs_26 = reinterpret_cast<uint32_t*>(&_vld_26[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_27 = reinterpret_cast<const uint4*>(residual + (((grow_base + 32 < M) ? grow_base + 32 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_27[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_27[_blk] = _vptr_27[_blk];
                        uint32_t* _vpairs_27 = reinterpret_cast<uint32_t*>(&_vld_27[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled_6[16];
                scaled_6[0] = accum[64] * sa[4] * sw[0];
                scaled_6[1] = accum[65] * sa[4] * sw[1];
                scaled_6[2] = accum[68] * sa[4] * sw[2];
                scaled_6[3] = accum[69] * sa[4] * sw[3];
                scaled_6[4] = accum[72] * sa[4] * sw[4];
                scaled_6[5] = accum[73] * sa[4] * sw[5];
                scaled_6[6] = accum[76] * sa[4] * sw[6];
                scaled_6[7] = accum[77] * sa[4] * sw[7];
                scaled_6[8] = accum[80] * sa[4] * sw[8];
                scaled_6[9] = accum[81] * sa[4] * sw[9];
                scaled_6[10] = accum[84] * sa[4] * sw[10];
                scaled_6[11] = accum[85] * sa[4] * sw[11];
                scaled_6[12] = accum[88] * sa[4] * sw[12];
                scaled_6[13] = accum[89] * sa[4] * sw[13];
                scaled_6[14] = accum[92] * sa[4] * sw[14];
                scaled_6[15] = accum[93] * sa[4] * sw[15];
                unsigned int scaled_6_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_6[_lp*2 + 0], scaled_6[_lp*2+1 + 0]));
                    scaled_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_28 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_29 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow_7 = grow_base + 32;
                if (grow_7 < M) {
                    unsigned int o_words_4[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_12[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words_4[_pair]) << 16);
                        (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words_4[_pair]) & 0xffff0000u);
                    }
                    int valid_4 = meta[1];
                    float p_raw_4[16];
                    for (int e_8 = 0; e_8 < 8; e_8++) {
                        float gv0_4 = ops[e_8];
                        float gv1_4 = ops[8 + e_8];
                        float gs0_4 = ((valid_4 == 1) ? gv0_4 : 0.0f);
                        float gs1_4 = ((valid_4 == 1) ? gv1_4 : 0.0f);
                        p_raw_4[e_8] = gs0_4 * _unpack_bf16_12[e_8];
                        p_raw_4[8 + e_8] = gs1_4 * _unpack_bf16_12[8 + e_8];
                    }
                    unsigned int p_raw_bf16_4[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_4[_lp*2 + 0], p_raw_4[_lp*2+1 + 0]));
                        p_raw_bf16_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_13[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_4[_pair]) << 16);
                        (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_4[_pair]) & 0xffff0000u);
                    }
                    float out_raw_4[16];
                    for (int e_9 = 0; e_9 < 8; e_9++) {
                        float rv0_4 = ops[16 + e_9];
                        float rv1_4 = ops[24 + e_9];
                        out_raw_4[e_9] = rv0_4 + _unpack_bf16_13[e_9];
                        out_raw_4[8 + e_9] = rv1_4 + _unpack_bf16_13[8 + e_9];
                    }
                    unsigned int out_raw_bf16_4[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_4[_lp*2 + 0], out_raw_4[_lp*2+1 + 0]));
                        out_raw_bf16_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_7 < M) {
                        reinterpret_cast<int4*>(out + ((grow_7 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_4 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_7 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_4 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 40 < M) ? grow_base + 40 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_30 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_30[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_30[_blk] = _vptr_30[_blk];
                        uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_31 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_31[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_31[_blk] = _vptr_31[_blk];
                        uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_32 = reinterpret_cast<const uint4*>(residual + (((grow_base + 40 < M) ? grow_base + 40 : 0) * 5376 + col) + 0);
                    uint4 _vld_32[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_32[_blk] = _vptr_32[_blk];
                        uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_33 = reinterpret_cast<const uint4*>(residual + (((grow_base + 40 < M) ? grow_base + 40 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_33[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_33[_blk] = _vptr_33[_blk];
                        uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled_8[16];
                scaled_8[0] = accum[66] * sa[5] * sw[0];
                scaled_8[1] = accum[67] * sa[5] * sw[1];
                scaled_8[2] = accum[70] * sa[5] * sw[2];
                scaled_8[3] = accum[71] * sa[5] * sw[3];
                scaled_8[4] = accum[74] * sa[5] * sw[4];
                scaled_8[5] = accum[75] * sa[5] * sw[5];
                scaled_8[6] = accum[78] * sa[5] * sw[6];
                scaled_8[7] = accum[79] * sa[5] * sw[7];
                scaled_8[8] = accum[82] * sa[5] * sw[8];
                scaled_8[9] = accum[83] * sa[5] * sw[9];
                scaled_8[10] = accum[86] * sa[5] * sw[10];
                scaled_8[11] = accum[87] * sa[5] * sw[11];
                scaled_8[12] = accum[90] * sa[5] * sw[12];
                scaled_8[13] = accum[91] * sa[5] * sw[13];
                scaled_8[14] = accum[94] * sa[5] * sw[14];
                scaled_8[15] = accum[95] * sa[5] * sw[15];
                unsigned int scaled_8_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_8[_lp*2 + 0], scaled_8[_lp*2+1 + 0]));
                    scaled_8_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow_9 = grow_base + 40;
                if (grow_9 < M) {
                    unsigned int o_words_5[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_14[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words_5[_pair]) << 16);
                        (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words_5[_pair]) & 0xffff0000u);
                    }
                    int valid_5 = meta[1];
                    float p_raw_5[16];
                    for (int e_10 = 0; e_10 < 8; e_10++) {
                        float gv0_5 = ops[e_10];
                        float gv1_5 = ops[8 + e_10];
                        float gs0_5 = ((valid_5 == 1) ? gv0_5 : 0.0f);
                        float gs1_5 = ((valid_5 == 1) ? gv1_5 : 0.0f);
                        p_raw_5[e_10] = gs0_5 * _unpack_bf16_14[e_10];
                        p_raw_5[8 + e_10] = gs1_5 * _unpack_bf16_14[8 + e_10];
                    }
                    unsigned int p_raw_bf16_5[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_5[_lp*2 + 0], p_raw_5[_lp*2+1 + 0]));
                        p_raw_bf16_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_15[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_5[_pair]) << 16);
                        (&_unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_5[_pair]) & 0xffff0000u);
                    }
                    float out_raw_5[16];
                    for (int e_11 = 0; e_11 < 8; e_11++) {
                        float rv0_5 = ops[16 + e_11];
                        float rv1_5 = ops[24 + e_11];
                        out_raw_5[e_11] = rv0_5 + _unpack_bf16_15[e_11];
                        out_raw_5[8 + e_11] = rv1_5 + _unpack_bf16_15[8 + e_11];
                    }
                    unsigned int out_raw_bf16_5[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_5[_lp*2 + 0], out_raw_5[_lp*2+1 + 0]));
                        out_raw_bf16_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_9 < M) {
                        reinterpret_cast<int4*>(out + ((grow_9 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_5 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_9 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_5 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 48 < M) ? grow_base + 48 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_36 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_36[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_36[_blk] = _vptr_36[_blk];
                        uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_37 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_37[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_37[_blk] = _vptr_37[_blk];
                        uint32_t* _vpairs_37 = reinterpret_cast<uint32_t*>(&_vld_37[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_38 = reinterpret_cast<const uint4*>(residual + (((grow_base + 48 < M) ? grow_base + 48 : 0) * 5376 + col) + 0);
                    uint4 _vld_38[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_38[_blk] = _vptr_38[_blk];
                        uint32_t* _vpairs_38 = reinterpret_cast<uint32_t*>(&_vld_38[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_39 = reinterpret_cast<const uint4*>(residual + (((grow_base + 48 < M) ? grow_base + 48 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_39[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_39[_blk] = _vptr_39[_blk];
                        uint32_t* _vpairs_39 = reinterpret_cast<uint32_t*>(&_vld_39[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled_10[16];
                scaled_10[0] = accum[96] * sa[6] * sw[0];
                scaled_10[1] = accum[97] * sa[6] * sw[1];
                scaled_10[2] = accum[100] * sa[6] * sw[2];
                scaled_10[3] = accum[101] * sa[6] * sw[3];
                scaled_10[4] = accum[104] * sa[6] * sw[4];
                scaled_10[5] = accum[105] * sa[6] * sw[5];
                scaled_10[6] = accum[108] * sa[6] * sw[6];
                scaled_10[7] = accum[109] * sa[6] * sw[7];
                scaled_10[8] = accum[112] * sa[6] * sw[8];
                scaled_10[9] = accum[113] * sa[6] * sw[9];
                scaled_10[10] = accum[116] * sa[6] * sw[10];
                scaled_10[11] = accum[117] * sa[6] * sw[11];
                scaled_10[12] = accum[120] * sa[6] * sw[12];
                scaled_10[13] = accum[121] * sa[6] * sw[13];
                scaled_10[14] = accum[124] * sa[6] * sw[14];
                scaled_10[15] = accum[125] * sa[6] * sw[15];
                unsigned int scaled_10_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_10[_lp*2 + 0], scaled_10[_lp*2+1 + 0]));
                    scaled_10_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_40 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_40), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow_11 = grow_base + 48;
                if (grow_11 < M) {
                    unsigned int o_words_6[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_16[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words_6[_pair]) << 16);
                        (&_unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words_6[_pair]) & 0xffff0000u);
                    }
                    int valid_6 = meta[1];
                    float p_raw_6[16];
                    for (int e_12 = 0; e_12 < 8; e_12++) {
                        float gv0_6 = ops[e_12];
                        float gv1_6 = ops[8 + e_12];
                        float gs0_6 = ((valid_6 == 1) ? gv0_6 : 0.0f);
                        float gs1_6 = ((valid_6 == 1) ? gv1_6 : 0.0f);
                        p_raw_6[e_12] = gs0_6 * _unpack_bf16_16[e_12];
                        p_raw_6[8 + e_12] = gs1_6 * _unpack_bf16_16[8 + e_12];
                    }
                    unsigned int p_raw_bf16_6[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_6[_lp*2 + 0], p_raw_6[_lp*2+1 + 0]));
                        p_raw_bf16_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_17[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_6[_pair]) << 16);
                        (&_unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_6[_pair]) & 0xffff0000u);
                    }
                    float out_raw_6[16];
                    for (int e_13 = 0; e_13 < 8; e_13++) {
                        float rv0_6 = ops[16 + e_13];
                        float rv1_6 = ops[24 + e_13];
                        out_raw_6[e_13] = rv0_6 + _unpack_bf16_17[e_13];
                        out_raw_6[8 + e_13] = rv1_6 + _unpack_bf16_17[8 + e_13];
                    }
                    unsigned int out_raw_bf16_6[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_6[_lp*2 + 0], out_raw_6[_lp*2+1 + 0]));
                        out_raw_bf16_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_11 < M) {
                        reinterpret_cast<int4*>(out + ((grow_11 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_6 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_11 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_6 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 56 < M) ? grow_base + 56 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_42 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_42[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_42[_blk] = _vptr_42[_blk];
                        uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_43 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_43[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_43[_blk] = _vptr_43[_blk];
                        uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_44 = reinterpret_cast<const uint4*>(residual + (((grow_base + 56 < M) ? grow_base + 56 : 0) * 5376 + col) + 0);
                    uint4 _vld_44[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_44[_blk] = _vptr_44[_blk];
                        uint32_t* _vpairs_44 = reinterpret_cast<uint32_t*>(&_vld_44[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_45 = reinterpret_cast<const uint4*>(residual + (((grow_base + 56 < M) ? grow_base + 56 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_45[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_45[_blk] = _vptr_45[_blk];
                        uint32_t* _vpairs_45 = reinterpret_cast<uint32_t*>(&_vld_45[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled_12[16];
                scaled_12[0] = accum[98] * sa[7] * sw[0];
                scaled_12[1] = accum[99] * sa[7] * sw[1];
                scaled_12[2] = accum[102] * sa[7] * sw[2];
                scaled_12[3] = accum[103] * sa[7] * sw[3];
                scaled_12[4] = accum[106] * sa[7] * sw[4];
                scaled_12[5] = accum[107] * sa[7] * sw[5];
                scaled_12[6] = accum[110] * sa[7] * sw[6];
                scaled_12[7] = accum[111] * sa[7] * sw[7];
                scaled_12[8] = accum[114] * sa[7] * sw[8];
                scaled_12[9] = accum[115] * sa[7] * sw[9];
                scaled_12[10] = accum[118] * sa[7] * sw[10];
                scaled_12[11] = accum[119] * sa[7] * sw[11];
                scaled_12[12] = accum[122] * sa[7] * sw[12];
                scaled_12[13] = accum[123] * sa[7] * sw[13];
                scaled_12[14] = accum[126] * sa[7] * sw[14];
                scaled_12[15] = accum[127] * sa[7] * sw[15];
                unsigned int scaled_12_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_12[_lp*2 + 0], scaled_12[_lp*2+1 + 0]));
                    scaled_12_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_46 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_46), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_47 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_47), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow_13 = grow_base + 56;
                if (grow_13 < M) {
                    unsigned int o_words_7[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_18[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words_7[_pair]) << 16);
                        (&_unpack_bf16_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words_7[_pair]) & 0xffff0000u);
                    }
                    int valid_7 = meta[1];
                    float p_raw_7[16];
                    for (int e_14 = 0; e_14 < 8; e_14++) {
                        float gv0_7 = ops[e_14];
                        float gv1_7 = ops[8 + e_14];
                        float gs0_7 = ((valid_7 == 1) ? gv0_7 : 0.0f);
                        float gs1_7 = ((valid_7 == 1) ? gv1_7 : 0.0f);
                        p_raw_7[e_14] = gs0_7 * _unpack_bf16_18[e_14];
                        p_raw_7[8 + e_14] = gs1_7 * _unpack_bf16_18[8 + e_14];
                    }
                    unsigned int p_raw_bf16_7[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_7[_lp*2 + 0], p_raw_7[_lp*2+1 + 0]));
                        p_raw_bf16_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_19[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_7[_pair]) << 16);
                        (&_unpack_bf16_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_7[_pair]) & 0xffff0000u);
                    }
                    float out_raw_7[16];
                    for (int e_15 = 0; e_15 < 8; e_15++) {
                        float rv0_7 = ops[16 + e_15];
                        float rv1_7 = ops[24 + e_15];
                        out_raw_7[e_15] = rv0_7 + _unpack_bf16_19[e_15];
                        out_raw_7[8 + e_15] = rv1_7 + _unpack_bf16_19[8 + e_15];
                    }
                    unsigned int out_raw_bf16_7[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_7[_lp*2 + 0], out_raw_7[_lp*2+1 + 0]));
                        out_raw_bf16_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_13 < M) {
                        reinterpret_cast<int4*>(out + ((grow_13 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_7 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_13 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_7 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
            }
        }
    }

    // Cleanup
}

}  // namespace h3_out_proj_gemm_fp8_sm120a
#undef CAKE_INF
#undef GROUP_M
#undef LAUNCH_MIN_BLOCKS
#undef NUM_AB_PIPE_STAGES
#undef SMEM_A_STAGE_OFF
#undef SMEM_A_STAGE_STAGE_BYTES
#undef SMEM_A_STAGE_STRIDE
#undef SMEM_B_STAGE_OFF
#undef SMEM_B_STAGE_STAGE_BYTES
#undef SMEM_B_STAGE_STRIDE
#undef SMEM_SFA_SLOT_OFF
#undef SMEM_SFA_SLOT_STAGE_BYTES
#undef SMEM_SFA_SLOT_STRIDE
#undef SMEM_SFB_STAGE_OFF
#undef SMEM_SFB_STAGE_STAGE_BYTES
#undef SMEM_SFB_STAGE_STRIDE
#undef SMEM_STAGING_OFF
#undef SMEM_STAGING_STAGE_BYTES
#undef SMEM_STAGING_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef ab_empty_addr
#undef ab_full_addr

namespace h3_out_proj_gemm_nvfp4_sm120a {

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
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_AB_PIPE_STAGES 3
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 16384
#define SMEM_A_STAGE_STRIDE 16384
#define SMEM_B_STAGE_OFF 50176
#define SMEM_B_STAGE_STAGE_BYTES 8192
#define SMEM_B_STAGE_STRIDE 8192
#define SMEM_STAGING_OFF 74752
#define SMEM_STAGING_STAGE_BYTES 8192
#define SMEM_STAGING_STRIDE 8192
#define SMEM_SFA_SLOT_OFF 82944
#define SMEM_SFA_SLOT_STAGE_BYTES 4096
#define SMEM_SFA_SLOT_STRIDE 4096
#define SMEM_SFB_STAGE_OFF 91136
#define SMEM_SFB_STAGE_STAGE_BYTES 1024
#define SMEM_SFB_STAGE_STRIDE 1024
#define SMEM_TOTAL 94208
#define THREADS 384
#define GROUP_M 1
#define LAUNCH_MIN_BLOCKS 1


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





__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
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


__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_h3_out_proj_gemm_fused(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, unsigned int* __restrict__ attn_out_w, unsigned int* __restrict__ act_q_w, uint8_t* __restrict__ act_sf_w, float* __restrict__ act_scale, float* __restrict__ act_global_scale, float* __restrict__ w_scale, __nv_bfloat16* __restrict__ gate, int* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, unsigned int* __restrict__ out, unsigned int* __restrict__ flags, int M, int num_m_tiles, int total_tiles, float alpha)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_empty_addr (mbar_base + 24)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int B_stage_addr = smem + 50176;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int staging_addr = smem + 74752;
    unsigned int* SFA_slot = reinterpret_cast<unsigned int*>(smem_raw + 82944);
    const int SFA_slot_addr = smem + 82944;
    unsigned int* SFB_stage = reinterpret_cast<unsigned int*>(smem_raw + 91136);
    const int SFB_stage_addr = smem + 91136;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // ab_empty: 3 barriers, init_count=8
            mbarrier_init(smem + 24, 8);
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Warpgroup: 0 ----
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        // ---- Role: producer ----
        if (warp == 0) {
            { // producer_main
                unsigned int load_stage = 0;
                unsigned int _phase_ab_empty = 1;
                #pragma unroll 1
                for (int tile = bid; tile < total_tiles; tile += num_bids) {
                    int tile_m = tile / (GROUP_M * 42) * GROUP_M + (tile - tile / (GROUP_M * 42) * (GROUP_M * 42)) % ((num_m_tiles - tile / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 42) * GROUP_M : GROUP_M);
                    int tile_n = (tile - tile / (GROUP_M * 42) * (GROUP_M * 42)) / ((num_m_tiles - tile / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 42) * GROUP_M : GROUP_M);
                    int rows_left = M - tile_m * 256;
                    unsigned int rows_ready = ((rows_left < 256) ? rows_left : 256);
                    if (elect_sync()) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(flags) + (tile_m))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(rows_ready)) >= static_cast<unsigned int>(1));
                        }
                        asm volatile("fence.proxy.async;" ::: "memory");
                        #pragma unroll 1
                        for (int kp = 0; kp < 28; kp++) {
                            mbarrier_wait_hint(ab_empty_addr + (load_stage) * 8, _phase_ab_empty, 10000000);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 29696);
                            tma_2d_gmem2smem(A_stage_addr + load_stage * 16384, (&A), kp * 2 * 64, tile_m * 256, ab_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(B_stage_addr + load_stage * 8192, (&B), kp * 2 * 64, tile_n * 128, ab_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(SFB_stage_addr + load_stage * 1024, (&SFB), 0, tile_n * 224 + kp * 2 * 4, ab_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(SFA_slot_addr + (unsigned int)((kp * 2 >> 1 & 1) * 4096), (&SFA), (kp * 2 >> 1) * 16, tile_m * 256, ab_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_ab_empty ^= 1; }
                            mbarrier_wait_hint(ab_empty_addr + (load_stage) * 8, _phase_ab_empty, 10000000);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 25600);
                            tma_2d_gmem2smem(A_stage_addr + load_stage * 16384, (&A), (kp * 2 + 1) * 64, tile_m * 256, ab_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(B_stage_addr + load_stage * 8192, (&B), (kp * 2 + 1) * 64, tile_n * 128, ab_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(SFB_stage_addr + load_stage * 1024, (&SFB), 0, tile_n * 224 + (kp * 2 + 1) * 4, ab_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_ab_empty ^= 1; }
                        }
                    }
                }
            }
        // ---- Role: quant ----
        } else if (warp >= 1 && warp <= 3) {
            { // quant_main
                unsigned int qw[16];
                float qx[16];
                float qacc[4];
                unsigned int qwords[4];
                float global_scale = act_global_scale[0];
                int warp_id_in_role = warp - 1;
                int qrow0 = bid * 3 + warp_id_in_role;
                #pragma unroll 1
                for (int row = qrow0; row < M; row += num_bids * 3) {
                    qacc[0] = 0.0f;
                    #pragma unroll 2
                    for (int j = 0; j < 7; j++) {
                        {
                            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * j) * 16) + 0);
                            qw[0 + 0] = _uv4_0.x;
                            qw[0 + 1] = _uv4_0.y;
                            qw[0 + 2] = _uv4_0.z;
                            qw[0 + 3] = _uv4_0.w;
                        }
                        {
                            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * j) * 16 + 4) + 0);
                            qw[4 + 0] = _uv4_1.x;
                            qw[4 + 1] = _uv4_1.y;
                            qw[4 + 2] = _uv4_1.z;
                            qw[4 + 3] = _uv4_1.w;
                        }
                        {
                            uint4 _uv4_2 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * j) * 16 + 8) + 0);
                            qw[8 + 0] = _uv4_2.x;
                            qw[8 + 1] = _uv4_2.y;
                            qw[8 + 2] = _uv4_2.z;
                            qw[8 + 3] = _uv4_2.w;
                        }
                        {
                            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * j) * 16 + 12) + 0);
                            qw[12 + 0] = _uv4_3.x;
                            qw[12 + 1] = _uv4_3.y;
                            qw[12 + 2] = _uv4_3.z;
                            qw[12 + 3] = _uv4_3.w;
                        }
                        float _unpack_bf16_0[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) << 16);
                            (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) & 0xffff0000u);
                        }
                        qacc[0] = 0.0f;
                        qx[0] = _unpack_bf16_0[0];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[0]));
                        qx[1] = _unpack_bf16_0[1];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[1]));
                        qx[2] = _unpack_bf16_0[2];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[2]));
                        qx[3] = _unpack_bf16_0[3];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[3]));
                        qx[4] = _unpack_bf16_0[4];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[4]));
                        qx[5] = _unpack_bf16_0[5];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[5]));
                        qx[6] = _unpack_bf16_0[6];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[6]));
                        qx[7] = _unpack_bf16_0[7];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[7]));
                        qx[8] = _unpack_bf16_0[8];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[8]));
                        qx[9] = _unpack_bf16_0[9];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[9]));
                        qx[10] = _unpack_bf16_0[10];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[10]));
                        qx[11] = _unpack_bf16_0[11];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[11]));
                        qx[12] = _unpack_bf16_0[12];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[12]));
                        qx[13] = _unpack_bf16_0[13];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[13]));
                        qx[14] = _unpack_bf16_0[14];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[14]));
                        qx[15] = _unpack_bf16_0[15];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[15]));
                        qacc[1] = qacc[0] * 0.16666667f * global_scale;
                        uint16_t _cvt_satfinite_e4m3x2_f32_4;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_4) : "f"(0.0f), "f"(qacc[1]));
                        *reinterpret_cast<unsigned char*>(act_sf_w + (row * 448 + (lane + 32 * j) * 2)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_4 & 127));
                        uint16_t _cvt_satfinite_e4m3x2_f32_6;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_6) : "f"(0.0f), "f"(qacc[1]));
                        uint16_t _e4m3x2_decode_7 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_6 & 0xFFu);
                        uint32_t _f16x2_decode_7;
                        float _fp8_e4m3_decode_5;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_7) : "h"(_e4m3x2_decode_7));
                        uint16_t _f16_decode_7 = (uint16_t)_f16x2_decode_7;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_5) : "h"(_f16_decode_7));
                        qacc[2] = _fp8_e4m3_decode_5;
                        float _fdiv_8 = __fdiv_rn(global_scale, qacc[2]);
                        qacc[3] = ((qacc[2] != 0.0f) ? _fdiv_8 : 0.0f);
                        qx[0] = qx[0] * qacc[3];
                        qx[1] = qx[1] * qacc[3];
                        qx[2] = qx[2] * qacc[3];
                        qx[3] = qx[3] * qacc[3];
                        qx[4] = qx[4] * qacc[3];
                        qx[5] = qx[5] * qacc[3];
                        qx[6] = qx[6] * qacc[3];
                        qx[7] = qx[7] * qacc[3];
                        qx[8] = qx[8] * qacc[3];
                        qx[9] = qx[9] * qacc[3];
                        qx[10] = qx[10] * qacc[3];
                        qx[11] = qx[11] * qacc[3];
                        qx[12] = qx[12] * qacc[3];
                        qx[13] = qx[13] * qacc[3];
                        qx[14] = qx[14] * qacc[3];
                        qx[15] = qx[15] * qacc[3];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qwords[0]) : "f"(qx[0]), "f"(qx[1]), "f"(qx[2]), "f"(qx[3]), "f"(qx[4]), "f"(qx[5]), "f"(qx[6]), "f"(qx[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qwords[1]) : "f"(qx[8]), "f"(qx[9]), "f"(qx[10]), "f"(qx[11]), "f"(qx[12]), "f"(qx[13]), "f"(qx[14]), "f"(qx[15]));
                        float _unpack_bf16_1[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) << 16);
                            (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) & 0xffff0000u);
                        }
                        qacc[0] = 0.0f;
                        qx[0] = _unpack_bf16_1[0];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[0]));
                        qx[1] = _unpack_bf16_1[1];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[1]));
                        qx[2] = _unpack_bf16_1[2];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[2]));
                        qx[3] = _unpack_bf16_1[3];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[3]));
                        qx[4] = _unpack_bf16_1[4];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[4]));
                        qx[5] = _unpack_bf16_1[5];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[5]));
                        qx[6] = _unpack_bf16_1[6];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[6]));
                        qx[7] = _unpack_bf16_1[7];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[7]));
                        qx[8] = _unpack_bf16_1[8];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[8]));
                        qx[9] = _unpack_bf16_1[9];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[9]));
                        qx[10] = _unpack_bf16_1[10];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[10]));
                        qx[11] = _unpack_bf16_1[11];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[11]));
                        qx[12] = _unpack_bf16_1[12];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[12]));
                        qx[13] = _unpack_bf16_1[13];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[13]));
                        qx[14] = _unpack_bf16_1[14];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[14]));
                        qx[15] = _unpack_bf16_1[15];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[15]));
                        qacc[1] = qacc[0] * 0.16666667f * global_scale;
                        uint16_t _cvt_satfinite_e4m3x2_f32_9;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_9) : "f"(0.0f), "f"(qacc[1]));
                        *reinterpret_cast<unsigned char*>(act_sf_w + (row * 448 + (lane + 32 * j) * 2 + 1)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_9 & 127));
                        uint16_t _cvt_satfinite_e4m3x2_f32_11;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_11) : "f"(0.0f), "f"(qacc[1]));
                        uint16_t _e4m3x2_decode_12 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_11 & 0xFFu);
                        uint32_t _f16x2_decode_12;
                        float _fp8_e4m3_decode_10;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_12) : "h"(_e4m3x2_decode_12));
                        uint16_t _f16_decode_12 = (uint16_t)_f16x2_decode_12;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_10) : "h"(_f16_decode_12));
                        qacc[2] = _fp8_e4m3_decode_10;
                        float _fdiv_13 = __fdiv_rn(global_scale, qacc[2]);
                        qacc[3] = ((qacc[2] != 0.0f) ? _fdiv_13 : 0.0f);
                        qx[0] = qx[0] * qacc[3];
                        qx[1] = qx[1] * qacc[3];
                        qx[2] = qx[2] * qacc[3];
                        qx[3] = qx[3] * qacc[3];
                        qx[4] = qx[4] * qacc[3];
                        qx[5] = qx[5] * qacc[3];
                        qx[6] = qx[6] * qacc[3];
                        qx[7] = qx[7] * qacc[3];
                        qx[8] = qx[8] * qacc[3];
                        qx[9] = qx[9] * qacc[3];
                        qx[10] = qx[10] * qacc[3];
                        qx[11] = qx[11] * qacc[3];
                        qx[12] = qx[12] * qacc[3];
                        qx[13] = qx[13] * qacc[3];
                        qx[14] = qx[14] * qacc[3];
                        qx[15] = qx[15] * qacc[3];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qwords[2]) : "f"(qx[0]), "f"(qx[1]), "f"(qx[2]), "f"(qx[3]), "f"(qx[4]), "f"(qx[5]), "f"(qx[6]), "f"(qx[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qwords[3]) : "f"(qx[8]), "f"(qx[9]), "f"(qx[10]), "f"(qx[11]), "f"(qx[12]), "f"(qx[13]), "f"(qx[14]), "f"(qx[15]));
                        reinterpret_cast<int4*>(act_q_w + ((row * 3584 + (lane + 32 * j) * 16) / 4))[0] = reinterpret_cast<int4*>(qwords)[0];
                    }
                    __threadfence();
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(flags) + (row / 256))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // mma_main
            unsigned int mma_stage = 0;
            int warp_id_in_role_1 = warp - 4;
            int warp_m = warp_id_in_role_1 % 4;
            int warp_n = warp_id_in_role_1 / 4;
            int role_tid = warp_id_in_role_1 * 32 + lane;
            double alpha_den = (double)act_global_scale[0] * (double)w_scale[0];
            float alpha_dev = (float)((double)1.0f / alpha_den);
            float accum[128];
            unsigned int a_frag[16];
            unsigned int b_frag[16];
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / (GROUP_M * 42) * GROUP_M + (tile_1 - tile_1 / (GROUP_M * 42) * (GROUP_M * 42)) % ((num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M : GROUP_M);
                int tile_n_1 = (tile_1 - tile_1 / (GROUP_M * 42) * (GROUP_M * 42)) / ((num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M : GROUP_M);
                accum[0] = 0.0f;
                accum[1] = 0.0f;
                accum[2] = 0.0f;
                accum[3] = 0.0f;
                accum[4] = 0.0f;
                accum[5] = 0.0f;
                accum[6] = 0.0f;
                accum[7] = 0.0f;
                accum[8] = 0.0f;
                accum[9] = 0.0f;
                accum[10] = 0.0f;
                accum[11] = 0.0f;
                accum[12] = 0.0f;
                accum[13] = 0.0f;
                accum[14] = 0.0f;
                accum[15] = 0.0f;
                accum[16] = 0.0f;
                accum[17] = 0.0f;
                accum[18] = 0.0f;
                accum[19] = 0.0f;
                accum[20] = 0.0f;
                accum[21] = 0.0f;
                accum[22] = 0.0f;
                accum[23] = 0.0f;
                accum[24] = 0.0f;
                accum[25] = 0.0f;
                accum[26] = 0.0f;
                accum[27] = 0.0f;
                accum[28] = 0.0f;
                accum[29] = 0.0f;
                accum[30] = 0.0f;
                accum[31] = 0.0f;
                accum[32] = 0.0f;
                accum[33] = 0.0f;
                accum[34] = 0.0f;
                accum[35] = 0.0f;
                accum[36] = 0.0f;
                accum[37] = 0.0f;
                accum[38] = 0.0f;
                accum[39] = 0.0f;
                accum[40] = 0.0f;
                accum[41] = 0.0f;
                accum[42] = 0.0f;
                accum[43] = 0.0f;
                accum[44] = 0.0f;
                accum[45] = 0.0f;
                accum[46] = 0.0f;
                accum[47] = 0.0f;
                accum[48] = 0.0f;
                accum[49] = 0.0f;
                accum[50] = 0.0f;
                accum[51] = 0.0f;
                accum[52] = 0.0f;
                accum[53] = 0.0f;
                accum[54] = 0.0f;
                accum[55] = 0.0f;
                accum[56] = 0.0f;
                accum[57] = 0.0f;
                accum[58] = 0.0f;
                accum[59] = 0.0f;
                accum[60] = 0.0f;
                accum[61] = 0.0f;
                accum[62] = 0.0f;
                accum[63] = 0.0f;
                accum[64] = 0.0f;
                accum[65] = 0.0f;
                accum[66] = 0.0f;
                accum[67] = 0.0f;
                accum[68] = 0.0f;
                accum[69] = 0.0f;
                accum[70] = 0.0f;
                accum[71] = 0.0f;
                accum[72] = 0.0f;
                accum[73] = 0.0f;
                accum[74] = 0.0f;
                accum[75] = 0.0f;
                accum[76] = 0.0f;
                accum[77] = 0.0f;
                accum[78] = 0.0f;
                accum[79] = 0.0f;
                accum[80] = 0.0f;
                accum[81] = 0.0f;
                accum[82] = 0.0f;
                accum[83] = 0.0f;
                accum[84] = 0.0f;
                accum[85] = 0.0f;
                accum[86] = 0.0f;
                accum[87] = 0.0f;
                accum[88] = 0.0f;
                accum[89] = 0.0f;
                accum[90] = 0.0f;
                accum[91] = 0.0f;
                accum[92] = 0.0f;
                accum[93] = 0.0f;
                accum[94] = 0.0f;
                accum[95] = 0.0f;
                accum[96] = 0.0f;
                accum[97] = 0.0f;
                accum[98] = 0.0f;
                accum[99] = 0.0f;
                accum[100] = 0.0f;
                accum[101] = 0.0f;
                accum[102] = 0.0f;
                accum[103] = 0.0f;
                accum[104] = 0.0f;
                accum[105] = 0.0f;
                accum[106] = 0.0f;
                accum[107] = 0.0f;
                accum[108] = 0.0f;
                accum[109] = 0.0f;
                accum[110] = 0.0f;
                accum[111] = 0.0f;
                accum[112] = 0.0f;
                accum[113] = 0.0f;
                accum[114] = 0.0f;
                accum[115] = 0.0f;
                accum[116] = 0.0f;
                accum[117] = 0.0f;
                accum[118] = 0.0f;
                accum[119] = 0.0f;
                accum[120] = 0.0f;
                accum[121] = 0.0f;
                accum[122] = 0.0f;
                accum[123] = 0.0f;
                accum[124] = 0.0f;
                accum[125] = 0.0f;
                accum[126] = 0.0f;
                accum[127] = 0.0f;
                int col_base = tile_n_1 * 128 + warp_n * 64 + (lane & 3) * 2;
                float sw[16];
                sw[0] = alpha_dev;
                sw[1] = alpha_dev;
                sw[2] = alpha_dev;
                sw[3] = alpha_dev;
                sw[4] = alpha_dev;
                sw[5] = alpha_dev;
                sw[6] = alpha_dev;
                sw[7] = alpha_dev;
                sw[8] = alpha_dev;
                sw[9] = alpha_dev;
                sw[10] = alpha_dev;
                sw[11] = alpha_dev;
                sw[12] = alpha_dev;
                sw[13] = alpha_dev;
                sw[14] = alpha_dev;
                sw[15] = alpha_dev;
                float sa[8];
                #pragma unroll 1
                for (int kp_1 = 0; kp_1 < 28; kp_1++) {
                    mbarrier_wait_hint(ab_full_addr + (mma_stage) * 8, _phase_ab_full, 10000000);
                    if (kp_1 == 24) {
                        int arow = tile_m_1 * 256 + warp_m * 64 + (lane >> 2);
                        sa[0] = 1.0f;
                        int arow_0 = tile_m_1 * 256 + warp_m * 64 + 8 + (lane >> 2);
                        sa[1] = 1.0f;
                        int arow_1 = tile_m_1 * 256 + warp_m * 64 + 16 + (lane >> 2);
                        sa[2] = 1.0f;
                        int arow_2 = tile_m_1 * 256 + warp_m * 64 + 16 + 8 + (lane >> 2);
                        sa[3] = 1.0f;
                        int arow_3 = tile_m_1 * 256 + warp_m * 64 + 32 + (lane >> 2);
                        sa[4] = 1.0f;
                        int arow_4 = tile_m_1 * 256 + warp_m * 64 + 32 + 8 + (lane >> 2);
                        sa[5] = 1.0f;
                        int arow_5 = tile_m_1 * 256 + warp_m * 64 + 48 + (lane >> 2);
                        sa[6] = 1.0f;
                        int arow_6 = tile_m_1 * 256 + warp_m * 64 + 48 + 8 + (lane >> 2);
                        sa[7] = 1.0f;
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs[1];
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++) {
                        smem_regs[_lr] = SFA_slot[(kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + (warp_m * 64 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr];
                    }
                    unsigned int smem_regs_0[1];
                    #pragma unroll
                    for (int _lr_1 = 0; _lr_1 < 1; _lr_1++) {
                        smem_regs_0[_lr_1] = SFA_slot[(kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + (warp_m * 64 + 16 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr_1];
                    }
                    unsigned int smem_regs_1[1];
                    #pragma unroll
                    for (int _lr_2 = 0; _lr_2 < 1; _lr_2++) {
                        smem_regs_1[_lr_2] = SFA_slot[(kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + (warp_m * 64 + 32 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr_2];
                    }
                    unsigned int smem_regs_2[1];
                    #pragma unroll
                    for (int _lr_3 = 0; _lr_3 < 1; _lr_3++) {
                        smem_regs_2[_lr_3] = SFA_slot[(kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + (warp_m * 64 + 48 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr_3];
                    }
                    unsigned int smem_regs_3[1];
                    #pragma unroll
                    for (int _lr_4 = 0; _lr_4 < 1; _lr_4++) {
                        smem_regs_3[_lr_4] = SFB_stage[mma_stage * 256 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_4];
                    }
                    unsigned int smem_regs_4[1];
                    #pragma unroll
                    for (int _lr_5 = 0; _lr_5 < 1; _lr_5++) {
                        smem_regs_4[_lr_5] = SFB_stage[mma_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_5];
                    }
                    unsigned int smem_regs_5[1];
                    #pragma unroll
                    for (int _lr_6 = 0; _lr_6 < 1; _lr_6++) {
                        smem_regs_5[_lr_6] = SFB_stage[mma_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_6];
                    }
                    unsigned int smem_regs_6[1];
                    #pragma unroll
                    for (int _lr_7 = 0; _lr_7 < 1; _lr_7++) {
                        smem_regs_6[_lr_7] = SFB_stage[mma_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_7];
                    }
                    unsigned int smem_regs_7[1];
                    #pragma unroll
                    for (int _lr_8 = 0; _lr_8 < 1; _lr_8++) {
                        smem_regs_7[_lr_8] = SFB_stage[mma_stage * 256 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_8];
                    }
                    unsigned int smem_regs_8[1];
                    #pragma unroll
                    for (int _lr_9 = 0; _lr_9 < 1; _lr_9++) {
                        smem_regs_8[_lr_9] = SFB_stage[mma_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_9];
                    }
                    unsigned int smem_regs_9[1];
                    #pragma unroll
                    for (int _lr_10 = 0; _lr_10 < 1; _lr_10++) {
                        smem_regs_9[_lr_10] = SFB_stage[mma_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_10];
                    }
                    unsigned int smem_regs_10[1];
                    #pragma unroll
                    for (int _lr_11 = 0; _lr_11 < 1; _lr_11++) {
                        smem_regs_10[_lr_11] = SFB_stage[mma_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_11];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_11[1];
                    #pragma unroll
                    for (int _lr_12 = 0; _lr_12 < 1; _lr_12++) {
                        smem_regs_11[_lr_12] = SFA_slot[(kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + 1 + (warp_m * 64 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr_12];
                    }
                    unsigned int smem_regs_12[1];
                    #pragma unroll
                    for (int _lr_13 = 0; _lr_13 < 1; _lr_13++) {
                        smem_regs_12[_lr_13] = SFA_slot[(kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + 1 + (warp_m * 64 + 16 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr_13];
                    }
                    unsigned int smem_regs_13[1];
                    #pragma unroll
                    for (int _lr_14 = 0; _lr_14 < 1; _lr_14++) {
                        smem_regs_13[_lr_14] = SFA_slot[(kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + 1 + (warp_m * 64 + 32 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr_14];
                    }
                    unsigned int smem_regs_14[1];
                    #pragma unroll
                    for (int _lr_15 = 0; _lr_15 < 1; _lr_15++) {
                        smem_regs_14[_lr_15] = SFA_slot[(kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + 1 + (warp_m * 64 + 48 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr_15];
                    }
                    unsigned int smem_regs_15[1];
                    #pragma unroll
                    for (int _lr_16 = 0; _lr_16 < 1; _lr_16++) {
                        smem_regs_15[_lr_16] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_16];
                    }
                    unsigned int smem_regs_16[1];
                    #pragma unroll
                    for (int _lr_17 = 0; _lr_17 < 1; _lr_17++) {
                        smem_regs_16[_lr_17] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_17];
                    }
                    unsigned int smem_regs_17[1];
                    #pragma unroll
                    for (int _lr_18 = 0; _lr_18 < 1; _lr_18++) {
                        smem_regs_17[_lr_18] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_18];
                    }
                    unsigned int smem_regs_18[1];
                    #pragma unroll
                    for (int _lr_19 = 0; _lr_19 < 1; _lr_19++) {
                        smem_regs_18[_lr_19] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_19];
                    }
                    unsigned int smem_regs_19[1];
                    #pragma unroll
                    for (int _lr_20 = 0; _lr_20 < 1; _lr_20++) {
                        smem_regs_19[_lr_20] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_20];
                    }
                    unsigned int smem_regs_20[1];
                    #pragma unroll
                    for (int _lr_21 = 0; _lr_21 < 1; _lr_21++) {
                        smem_regs_20[_lr_21] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_21];
                    }
                    unsigned int smem_regs_21[1];
                    #pragma unroll
                    for (int _lr_22 = 0; _lr_22 < 1; _lr_22++) {
                        smem_regs_21[_lr_22] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_22];
                    }
                    unsigned int smem_regs_22[1];
                    #pragma unroll
                    for (int _lr_23 = 0; _lr_23 < 1; _lr_23++) {
                        smem_regs_22[_lr_23] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_23];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_ab_full ^= 1; }
                    mbarrier_wait_hint(ab_full_addr + (mma_stage) * 8, _phase_ab_full, 10000000);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_23[1];
                    #pragma unroll
                    for (int _lr_24 = 0; _lr_24 < 1; _lr_24++) {
                        smem_regs_23[_lr_24] = SFA_slot[(kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + (warp_m * 64 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr_24];
                    }
                    unsigned int smem_regs_24[1];
                    #pragma unroll
                    for (int _lr_25 = 0; _lr_25 < 1; _lr_25++) {
                        smem_regs_24[_lr_25] = SFA_slot[(kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + (warp_m * 64 + 16 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr_25];
                    }
                    unsigned int smem_regs_25[1];
                    #pragma unroll
                    for (int _lr_26 = 0; _lr_26 < 1; _lr_26++) {
                        smem_regs_25[_lr_26] = SFA_slot[(kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + (warp_m * 64 + 32 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr_26];
                    }
                    unsigned int smem_regs_26[1];
                    #pragma unroll
                    for (int _lr_27 = 0; _lr_27 < 1; _lr_27++) {
                        smem_regs_26[_lr_27] = SFA_slot[(kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + (warp_m * 64 + 48 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr_27];
                    }
                    unsigned int smem_regs_27[1];
                    #pragma unroll
                    for (int _lr_28 = 0; _lr_28 < 1; _lr_28++) {
                        smem_regs_27[_lr_28] = SFB_stage[mma_stage * 256 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_28];
                    }
                    unsigned int smem_regs_28[1];
                    #pragma unroll
                    for (int _lr_29 = 0; _lr_29 < 1; _lr_29++) {
                        smem_regs_28[_lr_29] = SFB_stage[mma_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_29];
                    }
                    unsigned int smem_regs_29[1];
                    #pragma unroll
                    for (int _lr_30 = 0; _lr_30 < 1; _lr_30++) {
                        smem_regs_29[_lr_30] = SFB_stage[mma_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_30];
                    }
                    unsigned int smem_regs_30[1];
                    #pragma unroll
                    for (int _lr_31 = 0; _lr_31 < 1; _lr_31++) {
                        smem_regs_30[_lr_31] = SFB_stage[mma_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_31];
                    }
                    unsigned int smem_regs_31[1];
                    #pragma unroll
                    for (int _lr_32 = 0; _lr_32 < 1; _lr_32++) {
                        smem_regs_31[_lr_32] = SFB_stage[mma_stage * 256 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_32];
                    }
                    unsigned int smem_regs_32[1];
                    #pragma unroll
                    for (int _lr_33 = 0; _lr_33 < 1; _lr_33++) {
                        smem_regs_32[_lr_33] = SFB_stage[mma_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_33];
                    }
                    unsigned int smem_regs_33[1];
                    #pragma unroll
                    for (int _lr_34 = 0; _lr_34 < 1; _lr_34++) {
                        smem_regs_33[_lr_34] = SFB_stage[mma_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_34];
                    }
                    unsigned int smem_regs_34[1];
                    #pragma unroll
                    for (int _lr_35 = 0; _lr_35 < 1; _lr_35++) {
                        smem_regs_34[_lr_35] = SFB_stage[mma_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_35];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs_23[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs_23[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs_23[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs_23[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs_23[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_31[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs_23[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_32[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs_23[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_33[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs_23[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_34[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs_24[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs_24[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs_24[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs_24[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs_24[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_31[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs_24[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_32[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs_24[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_33[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs_24[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_34[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs_25[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs_25[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs_25[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs_25[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs_25[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_31[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs_25[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_32[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs_25[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_33[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs_25[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_34[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs_26[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs_26[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs_26[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs_26[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs_26[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_31[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs_26[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_32[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs_26[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_33[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs_26[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_34[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_35[1];
                    #pragma unroll
                    for (int _lr_36 = 0; _lr_36 < 1; _lr_36++) {
                        smem_regs_35[_lr_36] = SFA_slot[(kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + 1 + (warp_m * 64 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr_36];
                    }
                    unsigned int smem_regs_36[1];
                    #pragma unroll
                    for (int _lr_37 = 0; _lr_37 < 1; _lr_37++) {
                        smem_regs_36[_lr_37] = SFA_slot[(kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + 1 + (warp_m * 64 + 16 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr_37];
                    }
                    unsigned int smem_regs_37[1];
                    #pragma unroll
                    for (int _lr_38 = 0; _lr_38 < 1; _lr_38++) {
                        smem_regs_37[_lr_38] = SFA_slot[(kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + 1 + (warp_m * 64 + 32 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr_38];
                    }
                    unsigned int smem_regs_38[1];
                    #pragma unroll
                    for (int _lr_39 = 0; _lr_39 < 1; _lr_39++) {
                        smem_regs_38[_lr_39] = SFA_slot[(kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + 1 + (warp_m * 64 + 48 + (lane & 1) * 8 + (lane >> 2)) * 4 + _lr_39];
                    }
                    unsigned int smem_regs_39[1];
                    #pragma unroll
                    for (int _lr_40 = 0; _lr_40 < 1; _lr_40++) {
                        smem_regs_39[_lr_40] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_40];
                    }
                    unsigned int smem_regs_40[1];
                    #pragma unroll
                    for (int _lr_41 = 0; _lr_41 < 1; _lr_41++) {
                        smem_regs_40[_lr_41] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_41];
                    }
                    unsigned int smem_regs_41[1];
                    #pragma unroll
                    for (int _lr_42 = 0; _lr_42 < 1; _lr_42++) {
                        smem_regs_41[_lr_42] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_42];
                    }
                    unsigned int smem_regs_42[1];
                    #pragma unroll
                    for (int _lr_43 = 0; _lr_43 < 1; _lr_43++) {
                        smem_regs_42[_lr_43] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_43];
                    }
                    unsigned int smem_regs_43[1];
                    #pragma unroll
                    for (int _lr_44 = 0; _lr_44 < 1; _lr_44++) {
                        smem_regs_43[_lr_44] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_44];
                    }
                    unsigned int smem_regs_44[1];
                    #pragma unroll
                    for (int _lr_45 = 0; _lr_45 < 1; _lr_45++) {
                        smem_regs_44[_lr_45] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_45];
                    }
                    unsigned int smem_regs_45[1];
                    #pragma unroll
                    for (int _lr_46 = 0; _lr_46 < 1; _lr_46++) {
                        smem_regs_45[_lr_46] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_46];
                    }
                    unsigned int smem_regs_46[1];
                    #pragma unroll
                    for (int _lr_47 = 0; _lr_47 < 1; _lr_47++) {
                        smem_regs_46[_lr_47] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_47];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs_35[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_39[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs_35[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_40[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs_35[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_41[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs_35[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_42[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs_35[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_43[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs_35[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_44[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs_35[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_45[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs_35[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_46[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs_36[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_39[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs_36[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_40[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs_36[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_41[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs_36[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_42[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs_36[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_43[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs_36[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_44[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs_36[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_45[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs_36[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_46[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs_37[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_39[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs_37[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_40[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs_37[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_41[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs_37[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_42[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs_37[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_43[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs_37[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_44[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs_37[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_45[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs_37[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_46[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(smem_regs_38[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_39[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(smem_regs_38[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_40[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(smem_regs_38[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_41[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(smem_regs_38[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_42[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(smem_regs_38[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_43[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(smem_regs_38[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_44[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(smem_regs_38[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_45[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(smem_regs_38[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_46[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_ab_full ^= 1; }
                }
                if (tile_m_1 > 2147483647) {
                    float chk = 0.0f;
                    chk = chk + accum[0];
                    chk = chk + accum[1];
                    chk = chk + accum[2];
                    chk = chk + accum[3];
                    chk = chk + accum[4];
                    chk = chk + accum[5];
                    chk = chk + accum[6];
                    chk = chk + accum[7];
                    chk = chk + accum[8];
                    chk = chk + accum[9];
                    chk = chk + accum[10];
                    chk = chk + accum[11];
                    chk = chk + accum[12];
                    chk = chk + accum[13];
                    chk = chk + accum[14];
                    chk = chk + accum[15];
                    chk = chk + accum[16];
                    chk = chk + accum[17];
                    chk = chk + accum[18];
                    chk = chk + accum[19];
                    chk = chk + accum[20];
                    chk = chk + accum[21];
                    chk = chk + accum[22];
                    chk = chk + accum[23];
                    chk = chk + accum[24];
                    chk = chk + accum[25];
                    chk = chk + accum[26];
                    chk = chk + accum[27];
                    chk = chk + accum[28];
                    chk = chk + accum[29];
                    chk = chk + accum[30];
                    chk = chk + accum[31];
                    chk = chk + accum[32];
                    chk = chk + accum[33];
                    chk = chk + accum[34];
                    chk = chk + accum[35];
                    chk = chk + accum[36];
                    chk = chk + accum[37];
                    chk = chk + accum[38];
                    chk = chk + accum[39];
                    chk = chk + accum[40];
                    chk = chk + accum[41];
                    chk = chk + accum[42];
                    chk = chk + accum[43];
                    chk = chk + accum[44];
                    chk = chk + accum[45];
                    chk = chk + accum[46];
                    chk = chk + accum[47];
                    chk = chk + accum[48];
                    chk = chk + accum[49];
                    chk = chk + accum[50];
                    chk = chk + accum[51];
                    chk = chk + accum[52];
                    chk = chk + accum[53];
                    chk = chk + accum[54];
                    chk = chk + accum[55];
                    chk = chk + accum[56];
                    chk = chk + accum[57];
                    chk = chk + accum[58];
                    chk = chk + accum[59];
                    chk = chk + accum[60];
                    chk = chk + accum[61];
                    chk = chk + accum[62];
                    chk = chk + accum[63];
                    chk = chk + accum[64];
                    chk = chk + accum[65];
                    chk = chk + accum[66];
                    chk = chk + accum[67];
                    chk = chk + accum[68];
                    chk = chk + accum[69];
                    chk = chk + accum[70];
                    chk = chk + accum[71];
                    chk = chk + accum[72];
                    chk = chk + accum[73];
                    chk = chk + accum[74];
                    chk = chk + accum[75];
                    chk = chk + accum[76];
                    chk = chk + accum[77];
                    chk = chk + accum[78];
                    chk = chk + accum[79];
                    chk = chk + accum[80];
                    chk = chk + accum[81];
                    chk = chk + accum[82];
                    chk = chk + accum[83];
                    chk = chk + accum[84];
                    chk = chk + accum[85];
                    chk = chk + accum[86];
                    chk = chk + accum[87];
                    chk = chk + accum[88];
                    chk = chk + accum[89];
                    chk = chk + accum[90];
                    chk = chk + accum[91];
                    chk = chk + accum[92];
                    chk = chk + accum[93];
                    chk = chk + accum[94];
                    chk = chk + accum[95];
                    chk = chk + accum[96];
                    chk = chk + accum[97];
                    chk = chk + accum[98];
                    chk = chk + accum[99];
                    chk = chk + accum[100];
                    chk = chk + accum[101];
                    chk = chk + accum[102];
                    chk = chk + accum[103];
                    chk = chk + accum[104];
                    chk = chk + accum[105];
                    chk = chk + accum[106];
                    chk = chk + accum[107];
                    chk = chk + accum[108];
                    chk = chk + accum[109];
                    chk = chk + accum[110];
                    chk = chk + accum[111];
                    chk = chk + accum[112];
                    chk = chk + accum[113];
                    chk = chk + accum[114];
                    chk = chk + accum[115];
                    chk = chk + accum[116];
                    chk = chk + accum[117];
                    chk = chk + accum[118];
                    chk = chk + accum[119];
                    chk = chk + accum[120];
                    chk = chk + accum[121];
                    chk = chk + accum[122];
                    chk = chk + accum[123];
                    chk = chk + accum[124];
                    chk = chk + accum[125];
                    chk = chk + accum[126];
                    chk = chk + accum[127];
                    if (lane == 0) {
                        out[tile_1] = __as_u32(chk);
                    }
                }
                int srow_st = warp_m * 8 + (lane & 7);
                int smat = lane >> 3;
                int srow = role_tid >> 3;
                int c = role_tid & 7;
                int col = tile_n_1 * 128 + c * 16;
                int grow_base = tile_m_1 * 256 + (srow >> 3) * 64 + (srow & 7);
                float ops[32];
                int meta[2];
                meta[0] = gate_index[((grow_base < M) ? grow_base : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(residual + (((grow_base < M) ? grow_base : 0) * 5376 + col) + 0);
                    uint4 _vld_2[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_2[_blk] = _vptr_2[_blk];
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(residual + (((grow_base < M) ? grow_base : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_3[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled[16];
                scaled[0] = accum[0] * sa[0] * sw[0];
                scaled[1] = accum[1] * sa[0] * sw[1];
                scaled[2] = accum[4] * sa[0] * sw[2];
                scaled[3] = accum[5] * sa[0] * sw[3];
                scaled[4] = accum[8] * sa[0] * sw[4];
                scaled[5] = accum[9] * sa[0] * sw[5];
                scaled[6] = accum[12] * sa[0] * sw[6];
                scaled[7] = accum[13] * sa[0] * sw[7];
                scaled[8] = accum[16] * sa[0] * sw[8];
                scaled[9] = accum[17] * sa[0] * sw[9];
                scaled[10] = accum[20] * sa[0] * sw[10];
                scaled[11] = accum[21] * sa[0] * sw[11];
                scaled[12] = accum[24] * sa[0] * sw[12];
                scaled[13] = accum[25] * sa[0] * sw[13];
                scaled[14] = accum[28] * sa[0] * sw[14];
                scaled[15] = accum[29] * sa[0] * sw[15];
                unsigned int scaled_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                    scaled_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow = grow_base;
                if (grow < M) {
                    unsigned int o_words[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_2[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words[_pair]) << 16);
                        (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words[_pair]) & 0xffff0000u);
                    }
                    int valid = meta[1];
                    float p_raw[16];
                    for (int e = 0; e < 8; e++) {
                        float gv0 = ops[e];
                        float gv1 = ops[8 + e];
                        float gs0 = ((valid == 1) ? gv0 : 0.0f);
                        float gs1 = ((valid == 1) ? gv1 : 0.0f);
                        p_raw[e] = gs0 * _unpack_bf16_2[e];
                        p_raw[8 + e] = gs1 * _unpack_bf16_2[8 + e];
                    }
                    unsigned int p_raw_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw[_lp*2 + 0], p_raw[_lp*2+1 + 0]));
                        p_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_3[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16[_pair]) << 16);
                        (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16[_pair]) & 0xffff0000u);
                    }
                    float out_raw[16];
                    for (int e_1 = 0; e_1 < 8; e_1++) {
                        float rv0 = ops[16 + e_1];
                        float rv1 = ops[24 + e_1];
                        out_raw[e_1] = rv0 + _unpack_bf16_3[e_1];
                        out_raw[8 + e_1] = rv1 + _unpack_bf16_3[8 + e_1];
                    }
                    unsigned int out_raw_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw[_lp*2 + 0], out_raw[_lp*2+1 + 0]));
                        out_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow < M) {
                        reinterpret_cast<int4*>(out + ((grow * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 8 < M) ? grow_base + 8 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_6[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_6[_blk] = _vptr_6[_blk];
                        uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_7[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_7[_blk] = _vptr_7[_blk];
                        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(residual + (((grow_base + 8 < M) ? grow_base + 8 : 0) * 5376 + col) + 0);
                    uint4 _vld_8[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_8[_blk] = _vptr_8[_blk];
                        uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(residual + (((grow_base + 8 < M) ? grow_base + 8 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_9[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_9[_blk] = _vptr_9[_blk];
                        uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled_0[16];
                scaled_0[0] = accum[2] * sa[1] * sw[0];
                scaled_0[1] = accum[3] * sa[1] * sw[1];
                scaled_0[2] = accum[6] * sa[1] * sw[2];
                scaled_0[3] = accum[7] * sa[1] * sw[3];
                scaled_0[4] = accum[10] * sa[1] * sw[4];
                scaled_0[5] = accum[11] * sa[1] * sw[5];
                scaled_0[6] = accum[14] * sa[1] * sw[6];
                scaled_0[7] = accum[15] * sa[1] * sw[7];
                scaled_0[8] = accum[18] * sa[1] * sw[8];
                scaled_0[9] = accum[19] * sa[1] * sw[9];
                scaled_0[10] = accum[22] * sa[1] * sw[10];
                scaled_0[11] = accum[23] * sa[1] * sw[11];
                scaled_0[12] = accum[26] * sa[1] * sw[12];
                scaled_0[13] = accum[27] * sa[1] * sw[13];
                scaled_0[14] = accum[30] * sa[1] * sw[14];
                scaled_0[15] = accum[31] * sa[1] * sw[15];
                unsigned int scaled_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_0[_lp*2 + 0], scaled_0[_lp*2+1 + 0]));
                    scaled_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow_1 = grow_base + 8;
                if (grow_1 < M) {
                    unsigned int o_words_1[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_4[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words_1[_pair]) << 16);
                        (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words_1[_pair]) & 0xffff0000u);
                    }
                    int valid_1 = meta[1];
                    float p_raw_1[16];
                    for (int e_2 = 0; e_2 < 8; e_2++) {
                        float gv0_1 = ops[e_2];
                        float gv1_1 = ops[8 + e_2];
                        float gs0_1 = ((valid_1 == 1) ? gv0_1 : 0.0f);
                        float gs1_1 = ((valid_1 == 1) ? gv1_1 : 0.0f);
                        p_raw_1[e_2] = gs0_1 * _unpack_bf16_4[e_2];
                        p_raw_1[8 + e_2] = gs1_1 * _unpack_bf16_4[8 + e_2];
                    }
                    unsigned int p_raw_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_1[_lp*2 + 0], p_raw_1[_lp*2+1 + 0]));
                        p_raw_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_5[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_1[_pair]) << 16);
                        (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_1[_pair]) & 0xffff0000u);
                    }
                    float out_raw_1[16];
                    for (int e_3 = 0; e_3 < 8; e_3++) {
                        float rv0_1 = ops[16 + e_3];
                        float rv1_1 = ops[24 + e_3];
                        out_raw_1[e_3] = rv0_1 + _unpack_bf16_5[e_3];
                        out_raw_1[8 + e_3] = rv1_1 + _unpack_bf16_5[8 + e_3];
                    }
                    unsigned int out_raw_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_1[_lp*2 + 0], out_raw_1[_lp*2+1 + 0]));
                        out_raw_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_1 < M) {
                        reinterpret_cast<int4*>(out + ((grow_1 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_1 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_1 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_1 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 16 < M) ? grow_base + 16 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_12 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_12[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_12[_blk] = _vptr_12[_blk];
                        uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_13 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_13[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_13[_blk] = _vptr_13[_blk];
                        uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_14 = reinterpret_cast<const uint4*>(residual + (((grow_base + 16 < M) ? grow_base + 16 : 0) * 5376 + col) + 0);
                    uint4 _vld_14[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_14[_blk] = _vptr_14[_blk];
                        uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_15 = reinterpret_cast<const uint4*>(residual + (((grow_base + 16 < M) ? grow_base + 16 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_15[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_15[_blk] = _vptr_15[_blk];
                        uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled_2[16];
                scaled_2[0] = accum[32] * sa[2] * sw[0];
                scaled_2[1] = accum[33] * sa[2] * sw[1];
                scaled_2[2] = accum[36] * sa[2] * sw[2];
                scaled_2[3] = accum[37] * sa[2] * sw[3];
                scaled_2[4] = accum[40] * sa[2] * sw[4];
                scaled_2[5] = accum[41] * sa[2] * sw[5];
                scaled_2[6] = accum[44] * sa[2] * sw[6];
                scaled_2[7] = accum[45] * sa[2] * sw[7];
                scaled_2[8] = accum[48] * sa[2] * sw[8];
                scaled_2[9] = accum[49] * sa[2] * sw[9];
                scaled_2[10] = accum[52] * sa[2] * sw[10];
                scaled_2[11] = accum[53] * sa[2] * sw[11];
                scaled_2[12] = accum[56] * sa[2] * sw[12];
                scaled_2[13] = accum[57] * sa[2] * sw[13];
                scaled_2[14] = accum[60] * sa[2] * sw[14];
                scaled_2[15] = accum[61] * sa[2] * sw[15];
                unsigned int scaled_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_2[_lp*2 + 0], scaled_2[_lp*2+1 + 0]));
                    scaled_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow_3 = grow_base + 16;
                if (grow_3 < M) {
                    unsigned int o_words_2[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_6[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words_2[_pair]) << 16);
                        (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words_2[_pair]) & 0xffff0000u);
                    }
                    int valid_2 = meta[1];
                    float p_raw_2[16];
                    for (int e_4 = 0; e_4 < 8; e_4++) {
                        float gv0_2 = ops[e_4];
                        float gv1_2 = ops[8 + e_4];
                        float gs0_2 = ((valid_2 == 1) ? gv0_2 : 0.0f);
                        float gs1_2 = ((valid_2 == 1) ? gv1_2 : 0.0f);
                        p_raw_2[e_4] = gs0_2 * _unpack_bf16_6[e_4];
                        p_raw_2[8 + e_4] = gs1_2 * _unpack_bf16_6[8 + e_4];
                    }
                    unsigned int p_raw_bf16_2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_2[_lp*2 + 0], p_raw_2[_lp*2+1 + 0]));
                        p_raw_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_7[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_2[_pair]) << 16);
                        (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_2[_pair]) & 0xffff0000u);
                    }
                    float out_raw_2[16];
                    for (int e_5 = 0; e_5 < 8; e_5++) {
                        float rv0_2 = ops[16 + e_5];
                        float rv1_2 = ops[24 + e_5];
                        out_raw_2[e_5] = rv0_2 + _unpack_bf16_7[e_5];
                        out_raw_2[8 + e_5] = rv1_2 + _unpack_bf16_7[8 + e_5];
                    }
                    unsigned int out_raw_bf16_2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_2[_lp*2 + 0], out_raw_2[_lp*2+1 + 0]));
                        out_raw_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_3 < M) {
                        reinterpret_cast<int4*>(out + ((grow_3 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_2 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_3 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_2 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 24 < M) ? grow_base + 24 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_18 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_18[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_18[_blk] = _vptr_18[_blk];
                        uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_19 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_19[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_19[_blk] = _vptr_19[_blk];
                        uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_20 = reinterpret_cast<const uint4*>(residual + (((grow_base + 24 < M) ? grow_base + 24 : 0) * 5376 + col) + 0);
                    uint4 _vld_20[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_20[_blk] = _vptr_20[_blk];
                        uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_21 = reinterpret_cast<const uint4*>(residual + (((grow_base + 24 < M) ? grow_base + 24 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_21[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_21[_blk] = _vptr_21[_blk];
                        uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled_4[16];
                scaled_4[0] = accum[34] * sa[3] * sw[0];
                scaled_4[1] = accum[35] * sa[3] * sw[1];
                scaled_4[2] = accum[38] * sa[3] * sw[2];
                scaled_4[3] = accum[39] * sa[3] * sw[3];
                scaled_4[4] = accum[42] * sa[3] * sw[4];
                scaled_4[5] = accum[43] * sa[3] * sw[5];
                scaled_4[6] = accum[46] * sa[3] * sw[6];
                scaled_4[7] = accum[47] * sa[3] * sw[7];
                scaled_4[8] = accum[50] * sa[3] * sw[8];
                scaled_4[9] = accum[51] * sa[3] * sw[9];
                scaled_4[10] = accum[54] * sa[3] * sw[10];
                scaled_4[11] = accum[55] * sa[3] * sw[11];
                scaled_4[12] = accum[58] * sa[3] * sw[12];
                scaled_4[13] = accum[59] * sa[3] * sw[13];
                scaled_4[14] = accum[62] * sa[3] * sw[14];
                scaled_4[15] = accum[63] * sa[3] * sw[15];
                unsigned int scaled_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_4[_lp*2 + 0], scaled_4[_lp*2+1 + 0]));
                    scaled_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow_5 = grow_base + 24;
                if (grow_5 < M) {
                    unsigned int o_words_3[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_8[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words_3[_pair]) << 16);
                        (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words_3[_pair]) & 0xffff0000u);
                    }
                    int valid_3 = meta[1];
                    float p_raw_3[16];
                    for (int e_6 = 0; e_6 < 8; e_6++) {
                        float gv0_3 = ops[e_6];
                        float gv1_3 = ops[8 + e_6];
                        float gs0_3 = ((valid_3 == 1) ? gv0_3 : 0.0f);
                        float gs1_3 = ((valid_3 == 1) ? gv1_3 : 0.0f);
                        p_raw_3[e_6] = gs0_3 * _unpack_bf16_8[e_6];
                        p_raw_3[8 + e_6] = gs1_3 * _unpack_bf16_8[8 + e_6];
                    }
                    unsigned int p_raw_bf16_3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_3[_lp*2 + 0], p_raw_3[_lp*2+1 + 0]));
                        p_raw_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_9[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_3[_pair]) << 16);
                        (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_3[_pair]) & 0xffff0000u);
                    }
                    float out_raw_3[16];
                    for (int e_7 = 0; e_7 < 8; e_7++) {
                        float rv0_3 = ops[16 + e_7];
                        float rv1_3 = ops[24 + e_7];
                        out_raw_3[e_7] = rv0_3 + _unpack_bf16_9[e_7];
                        out_raw_3[8 + e_7] = rv1_3 + _unpack_bf16_9[8 + e_7];
                    }
                    unsigned int out_raw_bf16_3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_3[_lp*2 + 0], out_raw_3[_lp*2+1 + 0]));
                        out_raw_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_5 < M) {
                        reinterpret_cast<int4*>(out + ((grow_5 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_3 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_5 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_3 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 32 < M) ? grow_base + 32 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_24 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_24[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_24[_blk] = _vptr_24[_blk];
                        uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_25 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_25[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_25[_blk] = _vptr_25[_blk];
                        uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_26 = reinterpret_cast<const uint4*>(residual + (((grow_base + 32 < M) ? grow_base + 32 : 0) * 5376 + col) + 0);
                    uint4 _vld_26[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_26[_blk] = _vptr_26[_blk];
                        uint32_t* _vpairs_26 = reinterpret_cast<uint32_t*>(&_vld_26[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_27 = reinterpret_cast<const uint4*>(residual + (((grow_base + 32 < M) ? grow_base + 32 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_27[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_27[_blk] = _vptr_27[_blk];
                        uint32_t* _vpairs_27 = reinterpret_cast<uint32_t*>(&_vld_27[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled_6[16];
                scaled_6[0] = accum[64] * sa[4] * sw[0];
                scaled_6[1] = accum[65] * sa[4] * sw[1];
                scaled_6[2] = accum[68] * sa[4] * sw[2];
                scaled_6[3] = accum[69] * sa[4] * sw[3];
                scaled_6[4] = accum[72] * sa[4] * sw[4];
                scaled_6[5] = accum[73] * sa[4] * sw[5];
                scaled_6[6] = accum[76] * sa[4] * sw[6];
                scaled_6[7] = accum[77] * sa[4] * sw[7];
                scaled_6[8] = accum[80] * sa[4] * sw[8];
                scaled_6[9] = accum[81] * sa[4] * sw[9];
                scaled_6[10] = accum[84] * sa[4] * sw[10];
                scaled_6[11] = accum[85] * sa[4] * sw[11];
                scaled_6[12] = accum[88] * sa[4] * sw[12];
                scaled_6[13] = accum[89] * sa[4] * sw[13];
                scaled_6[14] = accum[92] * sa[4] * sw[14];
                scaled_6[15] = accum[93] * sa[4] * sw[15];
                unsigned int scaled_6_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_6[_lp*2 + 0], scaled_6[_lp*2+1 + 0]));
                    scaled_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_28 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_29 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow_7 = grow_base + 32;
                if (grow_7 < M) {
                    unsigned int o_words_4[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_10[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words_4[_pair]) << 16);
                        (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words_4[_pair]) & 0xffff0000u);
                    }
                    int valid_4 = meta[1];
                    float p_raw_4[16];
                    for (int e_8 = 0; e_8 < 8; e_8++) {
                        float gv0_4 = ops[e_8];
                        float gv1_4 = ops[8 + e_8];
                        float gs0_4 = ((valid_4 == 1) ? gv0_4 : 0.0f);
                        float gs1_4 = ((valid_4 == 1) ? gv1_4 : 0.0f);
                        p_raw_4[e_8] = gs0_4 * _unpack_bf16_10[e_8];
                        p_raw_4[8 + e_8] = gs1_4 * _unpack_bf16_10[8 + e_8];
                    }
                    unsigned int p_raw_bf16_4[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_4[_lp*2 + 0], p_raw_4[_lp*2+1 + 0]));
                        p_raw_bf16_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_11[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_4[_pair]) << 16);
                        (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_4[_pair]) & 0xffff0000u);
                    }
                    float out_raw_4[16];
                    for (int e_9 = 0; e_9 < 8; e_9++) {
                        float rv0_4 = ops[16 + e_9];
                        float rv1_4 = ops[24 + e_9];
                        out_raw_4[e_9] = rv0_4 + _unpack_bf16_11[e_9];
                        out_raw_4[8 + e_9] = rv1_4 + _unpack_bf16_11[8 + e_9];
                    }
                    unsigned int out_raw_bf16_4[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_4[_lp*2 + 0], out_raw_4[_lp*2+1 + 0]));
                        out_raw_bf16_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_7 < M) {
                        reinterpret_cast<int4*>(out + ((grow_7 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_4 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_7 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_4 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 40 < M) ? grow_base + 40 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_30 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_30[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_30[_blk] = _vptr_30[_blk];
                        uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_31 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_31[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_31[_blk] = _vptr_31[_blk];
                        uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_32 = reinterpret_cast<const uint4*>(residual + (((grow_base + 40 < M) ? grow_base + 40 : 0) * 5376 + col) + 0);
                    uint4 _vld_32[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_32[_blk] = _vptr_32[_blk];
                        uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_33 = reinterpret_cast<const uint4*>(residual + (((grow_base + 40 < M) ? grow_base + 40 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_33[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_33[_blk] = _vptr_33[_blk];
                        uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled_8[16];
                scaled_8[0] = accum[66] * sa[5] * sw[0];
                scaled_8[1] = accum[67] * sa[5] * sw[1];
                scaled_8[2] = accum[70] * sa[5] * sw[2];
                scaled_8[3] = accum[71] * sa[5] * sw[3];
                scaled_8[4] = accum[74] * sa[5] * sw[4];
                scaled_8[5] = accum[75] * sa[5] * sw[5];
                scaled_8[6] = accum[78] * sa[5] * sw[6];
                scaled_8[7] = accum[79] * sa[5] * sw[7];
                scaled_8[8] = accum[82] * sa[5] * sw[8];
                scaled_8[9] = accum[83] * sa[5] * sw[9];
                scaled_8[10] = accum[86] * sa[5] * sw[10];
                scaled_8[11] = accum[87] * sa[5] * sw[11];
                scaled_8[12] = accum[90] * sa[5] * sw[12];
                scaled_8[13] = accum[91] * sa[5] * sw[13];
                scaled_8[14] = accum[94] * sa[5] * sw[14];
                scaled_8[15] = accum[95] * sa[5] * sw[15];
                unsigned int scaled_8_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_8[_lp*2 + 0], scaled_8[_lp*2+1 + 0]));
                    scaled_8_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow_9 = grow_base + 40;
                if (grow_9 < M) {
                    unsigned int o_words_5[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_12[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words_5[_pair]) << 16);
                        (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words_5[_pair]) & 0xffff0000u);
                    }
                    int valid_5 = meta[1];
                    float p_raw_5[16];
                    for (int e_10 = 0; e_10 < 8; e_10++) {
                        float gv0_5 = ops[e_10];
                        float gv1_5 = ops[8 + e_10];
                        float gs0_5 = ((valid_5 == 1) ? gv0_5 : 0.0f);
                        float gs1_5 = ((valid_5 == 1) ? gv1_5 : 0.0f);
                        p_raw_5[e_10] = gs0_5 * _unpack_bf16_12[e_10];
                        p_raw_5[8 + e_10] = gs1_5 * _unpack_bf16_12[8 + e_10];
                    }
                    unsigned int p_raw_bf16_5[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_5[_lp*2 + 0], p_raw_5[_lp*2+1 + 0]));
                        p_raw_bf16_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_13[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_5[_pair]) << 16);
                        (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_5[_pair]) & 0xffff0000u);
                    }
                    float out_raw_5[16];
                    for (int e_11 = 0; e_11 < 8; e_11++) {
                        float rv0_5 = ops[16 + e_11];
                        float rv1_5 = ops[24 + e_11];
                        out_raw_5[e_11] = rv0_5 + _unpack_bf16_13[e_11];
                        out_raw_5[8 + e_11] = rv1_5 + _unpack_bf16_13[8 + e_11];
                    }
                    unsigned int out_raw_bf16_5[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_5[_lp*2 + 0], out_raw_5[_lp*2+1 + 0]));
                        out_raw_bf16_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_9 < M) {
                        reinterpret_cast<int4*>(out + ((grow_9 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_5 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_9 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_5 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 48 < M) ? grow_base + 48 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_36 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_36[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_36[_blk] = _vptr_36[_blk];
                        uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_37 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_37[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_37[_blk] = _vptr_37[_blk];
                        uint32_t* _vpairs_37 = reinterpret_cast<uint32_t*>(&_vld_37[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_38 = reinterpret_cast<const uint4*>(residual + (((grow_base + 48 < M) ? grow_base + 48 : 0) * 5376 + col) + 0);
                    uint4 _vld_38[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_38[_blk] = _vptr_38[_blk];
                        uint32_t* _vpairs_38 = reinterpret_cast<uint32_t*>(&_vld_38[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_39 = reinterpret_cast<const uint4*>(residual + (((grow_base + 48 < M) ? grow_base + 48 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_39[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_39[_blk] = _vptr_39[_blk];
                        uint32_t* _vpairs_39 = reinterpret_cast<uint32_t*>(&_vld_39[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled_10[16];
                scaled_10[0] = accum[96] * sa[6] * sw[0];
                scaled_10[1] = accum[97] * sa[6] * sw[1];
                scaled_10[2] = accum[100] * sa[6] * sw[2];
                scaled_10[3] = accum[101] * sa[6] * sw[3];
                scaled_10[4] = accum[104] * sa[6] * sw[4];
                scaled_10[5] = accum[105] * sa[6] * sw[5];
                scaled_10[6] = accum[108] * sa[6] * sw[6];
                scaled_10[7] = accum[109] * sa[6] * sw[7];
                scaled_10[8] = accum[112] * sa[6] * sw[8];
                scaled_10[9] = accum[113] * sa[6] * sw[9];
                scaled_10[10] = accum[116] * sa[6] * sw[10];
                scaled_10[11] = accum[117] * sa[6] * sw[11];
                scaled_10[12] = accum[120] * sa[6] * sw[12];
                scaled_10[13] = accum[121] * sa[6] * sw[13];
                scaled_10[14] = accum[124] * sa[6] * sw[14];
                scaled_10[15] = accum[125] * sa[6] * sw[15];
                unsigned int scaled_10_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_10[_lp*2 + 0], scaled_10[_lp*2+1 + 0]));
                    scaled_10_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_40 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_40), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow_11 = grow_base + 48;
                if (grow_11 < M) {
                    unsigned int o_words_6[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_14[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words_6[_pair]) << 16);
                        (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words_6[_pair]) & 0xffff0000u);
                    }
                    int valid_6 = meta[1];
                    float p_raw_6[16];
                    for (int e_12 = 0; e_12 < 8; e_12++) {
                        float gv0_6 = ops[e_12];
                        float gv1_6 = ops[8 + e_12];
                        float gs0_6 = ((valid_6 == 1) ? gv0_6 : 0.0f);
                        float gs1_6 = ((valid_6 == 1) ? gv1_6 : 0.0f);
                        p_raw_6[e_12] = gs0_6 * _unpack_bf16_14[e_12];
                        p_raw_6[8 + e_12] = gs1_6 * _unpack_bf16_14[8 + e_12];
                    }
                    unsigned int p_raw_bf16_6[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_6[_lp*2 + 0], p_raw_6[_lp*2+1 + 0]));
                        p_raw_bf16_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_15[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_6[_pair]) << 16);
                        (&_unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_6[_pair]) & 0xffff0000u);
                    }
                    float out_raw_6[16];
                    for (int e_13 = 0; e_13 < 8; e_13++) {
                        float rv0_6 = ops[16 + e_13];
                        float rv1_6 = ops[24 + e_13];
                        out_raw_6[e_13] = rv0_6 + _unpack_bf16_15[e_13];
                        out_raw_6[8 + e_13] = rv1_6 + _unpack_bf16_15[8 + e_13];
                    }
                    unsigned int out_raw_bf16_6[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_6[_lp*2 + 0], out_raw_6[_lp*2+1 + 0]));
                        out_raw_bf16_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_11 < M) {
                        reinterpret_cast<int4*>(out + ((grow_11 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_6 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_11 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_6 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 56 < M) ? grow_base + 56 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_42 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_42[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_42[_blk] = _vptr_42[_blk];
                        uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) << 16);
                            (&ops[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_43 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_43[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_43[_blk] = _vptr_43[_blk];
                        uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) << 16);
                            (&ops[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_44 = reinterpret_cast<const uint4*>(residual + (((grow_base + 56 < M) ? grow_base + 56 : 0) * 5376 + col) + 0);
                    uint4 _vld_44[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_44[_blk] = _vptr_44[_blk];
                        uint32_t* _vpairs_44 = reinterpret_cast<uint32_t*>(&_vld_44[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) << 16);
                            (&ops[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_45 = reinterpret_cast<const uint4*>(residual + (((grow_base + 56 < M) ? grow_base + 56 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_45[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_45[_blk] = _vptr_45[_blk];
                        uint32_t* _vpairs_45 = reinterpret_cast<uint32_t*>(&_vld_45[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ops[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) << 16);
                            (&ops[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float scaled_12[16];
                scaled_12[0] = accum[98] * sa[7] * sw[0];
                scaled_12[1] = accum[99] * sa[7] * sw[1];
                scaled_12[2] = accum[102] * sa[7] * sw[2];
                scaled_12[3] = accum[103] * sa[7] * sw[3];
                scaled_12[4] = accum[106] * sa[7] * sw[4];
                scaled_12[5] = accum[107] * sa[7] * sw[5];
                scaled_12[6] = accum[110] * sa[7] * sw[6];
                scaled_12[7] = accum[111] * sa[7] * sw[7];
                scaled_12[8] = accum[114] * sa[7] * sw[8];
                scaled_12[9] = accum[115] * sa[7] * sw[9];
                scaled_12[10] = accum[118] * sa[7] * sw[10];
                scaled_12[11] = accum[119] * sa[7] * sw[11];
                scaled_12[12] = accum[122] * sa[7] * sw[12];
                scaled_12[13] = accum[123] * sa[7] * sw[13];
                scaled_12[14] = accum[126] * sa[7] * sw[14];
                scaled_12[15] = accum[127] * sa[7] * sw[15];
                unsigned int scaled_12_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_12[_lp*2 + 0], scaled_12[_lp*2+1 + 0]));
                    scaled_12_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_46 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_46), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_47 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_47), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                int grow_13 = grow_base + 56;
                if (grow_13 < M) {
                    unsigned int o_words_7[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float _unpack_bf16_16[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(o_words_7[_pair]) << 16);
                        (&_unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(o_words_7[_pair]) & 0xffff0000u);
                    }
                    int valid_7 = meta[1];
                    float p_raw_7[16];
                    for (int e_14 = 0; e_14 < 8; e_14++) {
                        float gv0_7 = ops[e_14];
                        float gv1_7 = ops[8 + e_14];
                        float gs0_7 = ((valid_7 == 1) ? gv0_7 : 0.0f);
                        float gs1_7 = ((valid_7 == 1) ? gv1_7 : 0.0f);
                        p_raw_7[e_14] = gs0_7 * _unpack_bf16_16[e_14];
                        p_raw_7[8 + e_14] = gs1_7 * _unpack_bf16_16[8 + e_14];
                    }
                    unsigned int p_raw_bf16_7[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_7[_lp*2 + 0], p_raw_7[_lp*2+1 + 0]));
                        p_raw_bf16_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_17[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_7[_pair]) << 16);
                        (&_unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(p_raw_bf16_7[_pair]) & 0xffff0000u);
                    }
                    float out_raw_7[16];
                    for (int e_15 = 0; e_15 < 8; e_15++) {
                        float rv0_7 = ops[16 + e_15];
                        float rv1_7 = ops[24 + e_15];
                        out_raw_7[e_15] = rv0_7 + _unpack_bf16_17[e_15];
                        out_raw_7[8 + e_15] = rv1_7 + _unpack_bf16_17[8 + e_15];
                    }
                    unsigned int out_raw_bf16_7[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_7[_lp*2 + 0], out_raw_7[_lp*2+1 + 0]));
                        out_raw_bf16_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_13 < M) {
                        reinterpret_cast<int4*>(out + ((grow_13 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_7 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_13 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_7 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
            }
        }
    }

    // Cleanup
}

}  // namespace h3_out_proj_gemm_nvfp4_sm120a
#undef CAKE_INF
#undef GROUP_M
#undef LAUNCH_MIN_BLOCKS
#undef NUM_AB_PIPE_STAGES
#undef SMEM_A_STAGE_OFF
#undef SMEM_A_STAGE_STAGE_BYTES
#undef SMEM_A_STAGE_STRIDE
#undef SMEM_B_STAGE_OFF
#undef SMEM_B_STAGE_STAGE_BYTES
#undef SMEM_B_STAGE_STRIDE
#undef SMEM_SFA_SLOT_OFF
#undef SMEM_SFA_SLOT_STAGE_BYTES
#undef SMEM_SFA_SLOT_STRIDE
#undef SMEM_SFB_STAGE_OFF
#undef SMEM_SFB_STAGE_STAGE_BYTES
#undef SMEM_SFB_STAGE_STRIDE
#undef SMEM_STAGING_OFF
#undef SMEM_STAGING_STAGE_BYTES
#undef SMEM_STAGING_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef ab_empty_addr
#undef ab_full_addr

namespace h3_out_proj_gemm_fp8_warp_epilogue_sm120a {

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
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_AB_PIPE_STAGES 3
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 16384
#define SMEM_A_STAGE_STRIDE 16384
#define SMEM_B_STAGE_OFF 50176
#define SMEM_B_STAGE_STAGE_BYTES 8192
#define SMEM_B_STAGE_STRIDE 8192
#define SMEM_STAGING_OFF 74752
#define SMEM_STAGING_STAGE_BYTES 8192
#define SMEM_STAGING_STRIDE 8192
#define SMEM_SFA_SLOT_OFF 82944
#define SMEM_SFA_SLOT_STAGE_BYTES 4096
#define SMEM_SFA_SLOT_STRIDE 4096
#define SMEM_SFB_STAGE_OFF 91136
#define SMEM_SFB_STAGE_STAGE_BYTES 1024
#define SMEM_SFB_STAGE_STRIDE 1024
#define SMEM_TOTAL 94208
#define THREADS 384
#define GROUP_M 1
#define LAUNCH_MIN_BLOCKS 1


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





__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
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


__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_h3_out_proj_gemm_fused(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, unsigned int* __restrict__ attn_out_w, unsigned int* __restrict__ act_q_w, uint8_t* __restrict__ act_sf_w, float* __restrict__ act_scale, float* __restrict__ act_global_scale, float* __restrict__ w_scale, __nv_bfloat16* __restrict__ gate, int* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, unsigned int* __restrict__ out, unsigned int* __restrict__ flags, int M, int num_m_tiles, int total_tiles, float alpha)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_empty_addr (mbar_base + 24)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int B_stage_addr = smem + 50176;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int staging_addr = smem + 74752;
    unsigned int* SFA_slot = reinterpret_cast<unsigned int*>(smem_raw + 82944);
    const int SFA_slot_addr = smem + 82944;
    unsigned int* SFB_stage = reinterpret_cast<unsigned int*>(smem_raw + 91136);
    const int SFB_stage_addr = smem + 91136;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // ab_empty: 3 barriers, init_count=8
            mbarrier_init(smem + 24, 8);
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Warpgroup: 0 ----
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        // ---- Role: producer ----
        if (warp == 0) {
            { // producer_main
                unsigned int load_stage = 0;
                unsigned int _phase_ab_empty = 1;
                #pragma unroll 1
                for (int tile = bid; tile < total_tiles; tile += num_bids) {
                    int tile_m = tile / (GROUP_M * 42) * GROUP_M + (tile - tile / (GROUP_M * 42) * (GROUP_M * 42)) % ((num_m_tiles - tile / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 42) * GROUP_M : GROUP_M);
                    int tile_n = (tile - tile / (GROUP_M * 42) * (GROUP_M * 42)) / ((num_m_tiles - tile / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 42) * GROUP_M : GROUP_M);
                    int rows_left = M - tile_m * 256;
                    unsigned int rows_ready = ((rows_left < 256) ? rows_left : 256);
                    if (elect_sync()) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(flags) + (tile_m))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(rows_ready)) >= static_cast<unsigned int>(1));
                        }
                        asm volatile("fence.proxy.async;" ::: "memory");
                        #pragma unroll 1
                        for (int kp = 0; kp < 56; kp++) {
                            mbarrier_wait_hint(ab_empty_addr + (load_stage) * 8, _phase_ab_empty, 10000000);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 24576);
                            tma_2d_gmem2smem(A_stage_addr + load_stage * 16384, (&A), kp * 2 * 64, tile_m * 256, ab_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(B_stage_addr + load_stage * 8192, (&B), kp * 2 * 64, tile_n * 128, ab_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_ab_empty ^= 1; }
                            mbarrier_wait_hint(ab_empty_addr + (load_stage) * 8, _phase_ab_empty, 10000000);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 24576);
                            tma_2d_gmem2smem(A_stage_addr + load_stage * 16384, (&A), (kp * 2 + 1) * 64, tile_m * 256, ab_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(B_stage_addr + load_stage * 8192, (&B), (kp * 2 + 1) * 64, tile_n * 128, ab_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_ab_empty ^= 1; }
                        }
                    }
                }
            }
        // ---- Role: quant ----
        } else if (warp >= 1 && warp <= 3) {
            { // quant_main
                unsigned int qw[16];
                float qx[16];
                float qacc[4];
                unsigned int qwords[4];
                float global_scale = act_global_scale[0];
                int warp_id_in_role = warp - 1;
                int qrow0 = bid * 3 + warp_id_in_role;
                #pragma unroll 1
                for (int row = qrow0; row < M; row += num_bids * 3) {
                    qacc[0] = 0.0f;
                    #pragma unroll 2
                    for (int j = 0; j < 7; j++) {
                        {
                            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j * 2)) * 8) + 0);
                            qw[0 + 0] = _uv4_0.x;
                            qw[0 + 1] = _uv4_0.y;
                            qw[0 + 2] = _uv4_0.z;
                            qw[0 + 3] = _uv4_0.w;
                        }
                        {
                            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j * 2)) * 8 + 4) + 0);
                            qw[4 + 0] = _uv4_1.x;
                            qw[4 + 1] = _uv4_1.y;
                            qw[4 + 2] = _uv4_1.z;
                            qw[4 + 3] = _uv4_1.w;
                        }
                        {
                            uint4 _uv4_2 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j * 2 + 1)) * 8) + 0);
                            qw[8 + 0] = _uv4_2.x;
                            qw[8 + 1] = _uv4_2.y;
                            qw[8 + 2] = _uv4_2.z;
                            qw[8 + 3] = _uv4_2.w;
                        }
                        {
                            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j * 2 + 1)) * 8 + 4) + 0);
                            qw[12 + 0] = _uv4_3.x;
                            qw[12 + 1] = _uv4_3.y;
                            qw[12 + 2] = _uv4_3.z;
                            qw[12 + 3] = _uv4_3.w;
                        }
                        float _unpack_bf16_0[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) << 16);
                            (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) & 0xffff0000u);
                        }
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[0]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[1]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[2]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[3]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[4]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[5]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[6]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[7]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[8]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[9]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[10]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[11]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[12]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[13]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[14]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[15]));
                        float _unpack_bf16_1[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) << 16);
                            (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) & 0xffff0000u);
                        }
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[0]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[1]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[2]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[3]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[4]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[5]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[6]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[7]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[8]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[9]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[10]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[11]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[12]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[13]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[14]));
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[15]));
                    }
                    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, qacc[0], 16);
                    qacc[0] = fmaxf(qacc[0], _shfl_xor_sync_4);
                    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, qacc[0], 8);
                    qacc[0] = fmaxf(qacc[0], _shfl_xor_sync_5);
                    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, qacc[0], 4);
                    qacc[0] = fmaxf(qacc[0], _shfl_xor_sync_6);
                    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, qacc[0], 2);
                    qacc[0] = fmaxf(qacc[0], _shfl_xor_sync_7);
                    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, qacc[0], 1);
                    qacc[0] = fmaxf(qacc[0], _shfl_xor_sync_8);
                    float _fdiv_9 = __fdiv_rn(fmaxf(qacc[0], 1e-12f), 448.0f);
                    qacc[1] = _fdiv_9;
                    if (elect_sync()) {
                        *(reinterpret_cast<float*>(act_scale + row) + (0)) = qacc[1];
                    }
                    #pragma unroll 2
                    for (int j_1 = 0; j_1 < 7; j_1++) {
                        {
                            uint4 _uv4_10 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j_1 * 2)) * 8) + 0);
                            qw[0 + 0] = _uv4_10.x;
                            qw[0 + 1] = _uv4_10.y;
                            qw[0 + 2] = _uv4_10.z;
                            qw[0 + 3] = _uv4_10.w;
                        }
                        {
                            uint4 _uv4_11 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j_1 * 2)) * 8 + 4) + 0);
                            qw[4 + 0] = _uv4_11.x;
                            qw[4 + 1] = _uv4_11.y;
                            qw[4 + 2] = _uv4_11.z;
                            qw[4 + 3] = _uv4_11.w;
                        }
                        {
                            uint4 _uv4_12 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j_1 * 2 + 1)) * 8) + 0);
                            qw[8 + 0] = _uv4_12.x;
                            qw[8 + 1] = _uv4_12.y;
                            qw[8 + 2] = _uv4_12.z;
                            qw[8 + 3] = _uv4_12.w;
                        }
                        {
                            uint4 _uv4_13 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j_1 * 2 + 1)) * 8 + 4) + 0);
                            qw[12 + 0] = _uv4_13.x;
                            qw[12 + 1] = _uv4_13.y;
                            qw[12 + 2] = _uv4_13.z;
                            qw[12 + 3] = _uv4_13.w;
                        }
                        float _unpack_bf16_2[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) << 16);
                            (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) & 0xffff0000u);
                        }
                        float _fdiv_16 = __fdiv_rn(_unpack_bf16_2[1], qacc[1]);
                        float _fdiv_17 = __fdiv_rn(_unpack_bf16_2[0], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_15;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_15) : "f"(_fdiv_16), "f"(_fdiv_17));
                        float _fdiv_19 = __fdiv_rn(_unpack_bf16_2[3], qacc[1]);
                        float _fdiv_20 = __fdiv_rn(_unpack_bf16_2[2], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_18;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(_fdiv_19), "f"(_fdiv_20));
                        uint32_t _mov_b32_pack_u16x2_14;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_14) : "h"(_cvt_satfinite_e4m3x2_f32_15), "h"(_cvt_satfinite_e4m3x2_f32_18));
                        qwords[0] = _mov_b32_pack_u16x2_14;
                        float _fdiv_23 = __fdiv_rn(_unpack_bf16_2[5], qacc[1]);
                        float _fdiv_24 = __fdiv_rn(_unpack_bf16_2[4], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_22;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_22) : "f"(_fdiv_23), "f"(_fdiv_24));
                        float _fdiv_26 = __fdiv_rn(_unpack_bf16_2[7], qacc[1]);
                        float _fdiv_27 = __fdiv_rn(_unpack_bf16_2[6], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_25;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_25) : "f"(_fdiv_26), "f"(_fdiv_27));
                        uint32_t _mov_b32_pack_u16x2_21;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_21) : "h"(_cvt_satfinite_e4m3x2_f32_22), "h"(_cvt_satfinite_e4m3x2_f32_25));
                        qwords[1] = _mov_b32_pack_u16x2_21;
                        float _fdiv_30 = __fdiv_rn(_unpack_bf16_2[9], qacc[1]);
                        float _fdiv_31 = __fdiv_rn(_unpack_bf16_2[8], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_29;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_29) : "f"(_fdiv_30), "f"(_fdiv_31));
                        float _fdiv_33 = __fdiv_rn(_unpack_bf16_2[11], qacc[1]);
                        float _fdiv_34 = __fdiv_rn(_unpack_bf16_2[10], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_32;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_32) : "f"(_fdiv_33), "f"(_fdiv_34));
                        uint32_t _mov_b32_pack_u16x2_28;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_28) : "h"(_cvt_satfinite_e4m3x2_f32_29), "h"(_cvt_satfinite_e4m3x2_f32_32));
                        qwords[2] = _mov_b32_pack_u16x2_28;
                        float _fdiv_37 = __fdiv_rn(_unpack_bf16_2[13], qacc[1]);
                        float _fdiv_38 = __fdiv_rn(_unpack_bf16_2[12], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_36;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_36) : "f"(_fdiv_37), "f"(_fdiv_38));
                        float _fdiv_40 = __fdiv_rn(_unpack_bf16_2[15], qacc[1]);
                        float _fdiv_41 = __fdiv_rn(_unpack_bf16_2[14], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_39;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_39) : "f"(_fdiv_40), "f"(_fdiv_41));
                        uint32_t _mov_b32_pack_u16x2_35;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_35) : "h"(_cvt_satfinite_e4m3x2_f32_36), "h"(_cvt_satfinite_e4m3x2_f32_39));
                        qwords[3] = _mov_b32_pack_u16x2_35;
                        reinterpret_cast<int4*>(act_q_w + ((row * 7168 + (lane + 32 * (j_1 * 2)) * 16) / 4))[0] = reinterpret_cast<int4*>(qwords)[0];
                        float _unpack_bf16_3[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) << 16);
                            (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) & 0xffff0000u);
                        }
                        float _fdiv_44 = __fdiv_rn(_unpack_bf16_3[1], qacc[1]);
                        float _fdiv_45 = __fdiv_rn(_unpack_bf16_3[0], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_43;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_43) : "f"(_fdiv_44), "f"(_fdiv_45));
                        float _fdiv_47 = __fdiv_rn(_unpack_bf16_3[3], qacc[1]);
                        float _fdiv_48 = __fdiv_rn(_unpack_bf16_3[2], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_46;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_46) : "f"(_fdiv_47), "f"(_fdiv_48));
                        uint32_t _mov_b32_pack_u16x2_42;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_42) : "h"(_cvt_satfinite_e4m3x2_f32_43), "h"(_cvt_satfinite_e4m3x2_f32_46));
                        qwords[0] = _mov_b32_pack_u16x2_42;
                        float _fdiv_51 = __fdiv_rn(_unpack_bf16_3[5], qacc[1]);
                        float _fdiv_52 = __fdiv_rn(_unpack_bf16_3[4], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_50;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_50) : "f"(_fdiv_51), "f"(_fdiv_52));
                        float _fdiv_54 = __fdiv_rn(_unpack_bf16_3[7], qacc[1]);
                        float _fdiv_55 = __fdiv_rn(_unpack_bf16_3[6], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_53;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_53) : "f"(_fdiv_54), "f"(_fdiv_55));
                        uint32_t _mov_b32_pack_u16x2_49;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_49) : "h"(_cvt_satfinite_e4m3x2_f32_50), "h"(_cvt_satfinite_e4m3x2_f32_53));
                        qwords[1] = _mov_b32_pack_u16x2_49;
                        float _fdiv_58 = __fdiv_rn(_unpack_bf16_3[9], qacc[1]);
                        float _fdiv_59 = __fdiv_rn(_unpack_bf16_3[8], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_57;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_57) : "f"(_fdiv_58), "f"(_fdiv_59));
                        float _fdiv_61 = __fdiv_rn(_unpack_bf16_3[11], qacc[1]);
                        float _fdiv_62 = __fdiv_rn(_unpack_bf16_3[10], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_60;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_60) : "f"(_fdiv_61), "f"(_fdiv_62));
                        uint32_t _mov_b32_pack_u16x2_56;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_56) : "h"(_cvt_satfinite_e4m3x2_f32_57), "h"(_cvt_satfinite_e4m3x2_f32_60));
                        qwords[2] = _mov_b32_pack_u16x2_56;
                        float _fdiv_65 = __fdiv_rn(_unpack_bf16_3[13], qacc[1]);
                        float _fdiv_66 = __fdiv_rn(_unpack_bf16_3[12], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_64;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_64) : "f"(_fdiv_65), "f"(_fdiv_66));
                        float _fdiv_68 = __fdiv_rn(_unpack_bf16_3[15], qacc[1]);
                        float _fdiv_69 = __fdiv_rn(_unpack_bf16_3[14], qacc[1]);
                        uint16_t _cvt_satfinite_e4m3x2_f32_67;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_67) : "f"(_fdiv_68), "f"(_fdiv_69));
                        uint32_t _mov_b32_pack_u16x2_63;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_63) : "h"(_cvt_satfinite_e4m3x2_f32_64), "h"(_cvt_satfinite_e4m3x2_f32_67));
                        qwords[3] = _mov_b32_pack_u16x2_63;
                        reinterpret_cast<int4*>(act_q_w + ((row * 7168 + (lane + 32 * (j_1 * 2 + 1)) * 16) / 4))[0] = reinterpret_cast<int4*>(qwords)[0];
                    }
                    __threadfence();
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(flags) + (row / 256))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // mma_main
            unsigned int mma_stage = 0;
            int warp_id_in_role_1 = warp - 4;
            int warp_m = warp_id_in_role_1 % 4;
            int warp_n = warp_id_in_role_1 / 4;
            int role_tid = warp_id_in_role_1 * 32 + lane;
            float accum[128];
            unsigned int a_frag[16];
            unsigned int b_frag[16];
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / (GROUP_M * 42) * GROUP_M + (tile_1 - tile_1 / (GROUP_M * 42) * (GROUP_M * 42)) % ((num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M : GROUP_M);
                int tile_n_1 = (tile_1 - tile_1 / (GROUP_M * 42) * (GROUP_M * 42)) / ((num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M : GROUP_M);
                accum[0] = 0.0f;
                accum[1] = 0.0f;
                accum[2] = 0.0f;
                accum[3] = 0.0f;
                accum[4] = 0.0f;
                accum[5] = 0.0f;
                accum[6] = 0.0f;
                accum[7] = 0.0f;
                accum[8] = 0.0f;
                accum[9] = 0.0f;
                accum[10] = 0.0f;
                accum[11] = 0.0f;
                accum[12] = 0.0f;
                accum[13] = 0.0f;
                accum[14] = 0.0f;
                accum[15] = 0.0f;
                accum[16] = 0.0f;
                accum[17] = 0.0f;
                accum[18] = 0.0f;
                accum[19] = 0.0f;
                accum[20] = 0.0f;
                accum[21] = 0.0f;
                accum[22] = 0.0f;
                accum[23] = 0.0f;
                accum[24] = 0.0f;
                accum[25] = 0.0f;
                accum[26] = 0.0f;
                accum[27] = 0.0f;
                accum[28] = 0.0f;
                accum[29] = 0.0f;
                accum[30] = 0.0f;
                accum[31] = 0.0f;
                accum[32] = 0.0f;
                accum[33] = 0.0f;
                accum[34] = 0.0f;
                accum[35] = 0.0f;
                accum[36] = 0.0f;
                accum[37] = 0.0f;
                accum[38] = 0.0f;
                accum[39] = 0.0f;
                accum[40] = 0.0f;
                accum[41] = 0.0f;
                accum[42] = 0.0f;
                accum[43] = 0.0f;
                accum[44] = 0.0f;
                accum[45] = 0.0f;
                accum[46] = 0.0f;
                accum[47] = 0.0f;
                accum[48] = 0.0f;
                accum[49] = 0.0f;
                accum[50] = 0.0f;
                accum[51] = 0.0f;
                accum[52] = 0.0f;
                accum[53] = 0.0f;
                accum[54] = 0.0f;
                accum[55] = 0.0f;
                accum[56] = 0.0f;
                accum[57] = 0.0f;
                accum[58] = 0.0f;
                accum[59] = 0.0f;
                accum[60] = 0.0f;
                accum[61] = 0.0f;
                accum[62] = 0.0f;
                accum[63] = 0.0f;
                accum[64] = 0.0f;
                accum[65] = 0.0f;
                accum[66] = 0.0f;
                accum[67] = 0.0f;
                accum[68] = 0.0f;
                accum[69] = 0.0f;
                accum[70] = 0.0f;
                accum[71] = 0.0f;
                accum[72] = 0.0f;
                accum[73] = 0.0f;
                accum[74] = 0.0f;
                accum[75] = 0.0f;
                accum[76] = 0.0f;
                accum[77] = 0.0f;
                accum[78] = 0.0f;
                accum[79] = 0.0f;
                accum[80] = 0.0f;
                accum[81] = 0.0f;
                accum[82] = 0.0f;
                accum[83] = 0.0f;
                accum[84] = 0.0f;
                accum[85] = 0.0f;
                accum[86] = 0.0f;
                accum[87] = 0.0f;
                accum[88] = 0.0f;
                accum[89] = 0.0f;
                accum[90] = 0.0f;
                accum[91] = 0.0f;
                accum[92] = 0.0f;
                accum[93] = 0.0f;
                accum[94] = 0.0f;
                accum[95] = 0.0f;
                accum[96] = 0.0f;
                accum[97] = 0.0f;
                accum[98] = 0.0f;
                accum[99] = 0.0f;
                accum[100] = 0.0f;
                accum[101] = 0.0f;
                accum[102] = 0.0f;
                accum[103] = 0.0f;
                accum[104] = 0.0f;
                accum[105] = 0.0f;
                accum[106] = 0.0f;
                accum[107] = 0.0f;
                accum[108] = 0.0f;
                accum[109] = 0.0f;
                accum[110] = 0.0f;
                accum[111] = 0.0f;
                accum[112] = 0.0f;
                accum[113] = 0.0f;
                accum[114] = 0.0f;
                accum[115] = 0.0f;
                accum[116] = 0.0f;
                accum[117] = 0.0f;
                accum[118] = 0.0f;
                accum[119] = 0.0f;
                accum[120] = 0.0f;
                accum[121] = 0.0f;
                accum[122] = 0.0f;
                accum[123] = 0.0f;
                accum[124] = 0.0f;
                accum[125] = 0.0f;
                accum[126] = 0.0f;
                accum[127] = 0.0f;
                int col_base = tile_n_1 * 128 + warp_n * 64 + (lane & 3) * 2;
                float sw[16];
                sw[0] = w_scale[col_base];
                sw[1] = w_scale[col_base + 1];
                sw[2] = w_scale[col_base + 8];
                sw[3] = w_scale[col_base + 8 + 1];
                sw[4] = w_scale[col_base + 16];
                sw[5] = w_scale[col_base + 16 + 1];
                sw[6] = w_scale[col_base + 24];
                sw[7] = w_scale[col_base + 24 + 1];
                sw[8] = w_scale[col_base + 32];
                sw[9] = w_scale[col_base + 32 + 1];
                sw[10] = w_scale[col_base + 40];
                sw[11] = w_scale[col_base + 40 + 1];
                sw[12] = w_scale[col_base + 48];
                sw[13] = w_scale[col_base + 48 + 1];
                sw[14] = w_scale[col_base + 56];
                sw[15] = w_scale[col_base + 56 + 1];
                float sa[8];
                int wepi_gidx[16];
                #pragma unroll 1
                for (int kp_1 = 0; kp_1 < 56; kp_1++) {
                    mbarrier_wait_hint(ab_full_addr + (mma_stage) * 8, _phase_ab_full, 10000000);
                    if (kp_1 == 52) {
                        int arow = tile_m_1 * 256 + warp_m * 64 + (lane >> 2);
                        sa[0] = ((arow < M) ? act_scale[arow] : 0.0f);
                        int arow_0 = tile_m_1 * 256 + warp_m * 64 + 8 + (lane >> 2);
                        sa[1] = ((arow_0 < M) ? act_scale[arow_0] : 0.0f);
                        int arow_1 = tile_m_1 * 256 + warp_m * 64 + 16 + (lane >> 2);
                        sa[2] = ((arow_1 < M) ? act_scale[arow_1] : 0.0f);
                        int arow_2 = tile_m_1 * 256 + warp_m * 64 + 16 + 8 + (lane >> 2);
                        sa[3] = ((arow_2 < M) ? act_scale[arow_2] : 0.0f);
                        int arow_3 = tile_m_1 * 256 + warp_m * 64 + 32 + (lane >> 2);
                        sa[4] = ((arow_3 < M) ? act_scale[arow_3] : 0.0f);
                        int arow_4 = tile_m_1 * 256 + warp_m * 64 + 32 + 8 + (lane >> 2);
                        sa[5] = ((arow_4 < M) ? act_scale[arow_4] : 0.0f);
                        int arow_5 = tile_m_1 * 256 + warp_m * 64 + 48 + (lane >> 2);
                        sa[6] = ((arow_5 < M) ? act_scale[arow_5] : 0.0f);
                        int arow_6 = tile_m_1 * 256 + warp_m * 64 + 48 + 8 + (lane >> 2);
                        sa[7] = ((arow_6 < M) ? act_scale[arow_6] : 0.0f);
                        wepi_gidx[0] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) : 0)];
                        wepi_gidx[1] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 4 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 4 : 0)];
                        wepi_gidx[2] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 8 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 8 : 0)];
                        wepi_gidx[3] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 12 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 12 : 0)];
                        wepi_gidx[4] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 16 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 16 : 0)];
                        wepi_gidx[5] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 20 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 20 : 0)];
                        wepi_gidx[6] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 24 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 24 : 0)];
                        wepi_gidx[7] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 28 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 28 : 0)];
                        wepi_gidx[8] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 32 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 32 : 0)];
                        wepi_gidx[9] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 36 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 36 : 0)];
                        wepi_gidx[10] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 40 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 40 : 0)];
                        wepi_gidx[11] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 44 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 44 : 0)];
                        wepi_gidx[12] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 48 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 48 : 0)];
                        wepi_gidx[13] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 52 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 52 : 0)];
                        wepi_gidx[14] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 56 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 56 : 0)];
                        wepi_gidx[15] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 60 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 60 : 0)];
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_ab_full ^= 1; }
                    mbarrier_wait_hint(ab_full_addr + (mma_stage) * 8, _phase_ab_full, 10000000);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_ab_full ^= 1; }
                }
                if (tile_m_1 > 2147483647) {
                    float chk = 0.0f;
                    chk = chk + accum[0];
                    chk = chk + accum[1];
                    chk = chk + accum[2];
                    chk = chk + accum[3];
                    chk = chk + accum[4];
                    chk = chk + accum[5];
                    chk = chk + accum[6];
                    chk = chk + accum[7];
                    chk = chk + accum[8];
                    chk = chk + accum[9];
                    chk = chk + accum[10];
                    chk = chk + accum[11];
                    chk = chk + accum[12];
                    chk = chk + accum[13];
                    chk = chk + accum[14];
                    chk = chk + accum[15];
                    chk = chk + accum[16];
                    chk = chk + accum[17];
                    chk = chk + accum[18];
                    chk = chk + accum[19];
                    chk = chk + accum[20];
                    chk = chk + accum[21];
                    chk = chk + accum[22];
                    chk = chk + accum[23];
                    chk = chk + accum[24];
                    chk = chk + accum[25];
                    chk = chk + accum[26];
                    chk = chk + accum[27];
                    chk = chk + accum[28];
                    chk = chk + accum[29];
                    chk = chk + accum[30];
                    chk = chk + accum[31];
                    chk = chk + accum[32];
                    chk = chk + accum[33];
                    chk = chk + accum[34];
                    chk = chk + accum[35];
                    chk = chk + accum[36];
                    chk = chk + accum[37];
                    chk = chk + accum[38];
                    chk = chk + accum[39];
                    chk = chk + accum[40];
                    chk = chk + accum[41];
                    chk = chk + accum[42];
                    chk = chk + accum[43];
                    chk = chk + accum[44];
                    chk = chk + accum[45];
                    chk = chk + accum[46];
                    chk = chk + accum[47];
                    chk = chk + accum[48];
                    chk = chk + accum[49];
                    chk = chk + accum[50];
                    chk = chk + accum[51];
                    chk = chk + accum[52];
                    chk = chk + accum[53];
                    chk = chk + accum[54];
                    chk = chk + accum[55];
                    chk = chk + accum[56];
                    chk = chk + accum[57];
                    chk = chk + accum[58];
                    chk = chk + accum[59];
                    chk = chk + accum[60];
                    chk = chk + accum[61];
                    chk = chk + accum[62];
                    chk = chk + accum[63];
                    chk = chk + accum[64];
                    chk = chk + accum[65];
                    chk = chk + accum[66];
                    chk = chk + accum[67];
                    chk = chk + accum[68];
                    chk = chk + accum[69];
                    chk = chk + accum[70];
                    chk = chk + accum[71];
                    chk = chk + accum[72];
                    chk = chk + accum[73];
                    chk = chk + accum[74];
                    chk = chk + accum[75];
                    chk = chk + accum[76];
                    chk = chk + accum[77];
                    chk = chk + accum[78];
                    chk = chk + accum[79];
                    chk = chk + accum[80];
                    chk = chk + accum[81];
                    chk = chk + accum[82];
                    chk = chk + accum[83];
                    chk = chk + accum[84];
                    chk = chk + accum[85];
                    chk = chk + accum[86];
                    chk = chk + accum[87];
                    chk = chk + accum[88];
                    chk = chk + accum[89];
                    chk = chk + accum[90];
                    chk = chk + accum[91];
                    chk = chk + accum[92];
                    chk = chk + accum[93];
                    chk = chk + accum[94];
                    chk = chk + accum[95];
                    chk = chk + accum[96];
                    chk = chk + accum[97];
                    chk = chk + accum[98];
                    chk = chk + accum[99];
                    chk = chk + accum[100];
                    chk = chk + accum[101];
                    chk = chk + accum[102];
                    chk = chk + accum[103];
                    chk = chk + accum[104];
                    chk = chk + accum[105];
                    chk = chk + accum[106];
                    chk = chk + accum[107];
                    chk = chk + accum[108];
                    chk = chk + accum[109];
                    chk = chk + accum[110];
                    chk = chk + accum[111];
                    chk = chk + accum[112];
                    chk = chk + accum[113];
                    chk = chk + accum[114];
                    chk = chk + accum[115];
                    chk = chk + accum[116];
                    chk = chk + accum[117];
                    chk = chk + accum[118];
                    chk = chk + accum[119];
                    chk = chk + accum[120];
                    chk = chk + accum[121];
                    chk = chk + accum[122];
                    chk = chk + accum[123];
                    chk = chk + accum[124];
                    chk = chk + accum[125];
                    chk = chk + accum[126];
                    chk = chk + accum[127];
                    if (lane == 0) {
                        out[tile_1] = __as_u32(chk);
                    }
                }
                int wsmat = lane >> 3;
                int wrow_base = tile_m_1 * 256 + warp_m * 64 + (lane >> 3);
                int wsrow_st = warp_m * 8 + (lane & 7);
                int wcol = warp_n * 64 + (lane & 7) * 8;
                int wgcol = tile_n_1 * 128 + wcol;
                int wsrow0 = warp_m * 8 + wsmat;
                int wsrow1 = warp_m * 8 + 4 + wsmat;
                unsigned int wrw[64];
                unsigned int wgw[64];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? wepi_gidx[0] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_0 = reinterpret_cast<uint4*>(&wgw[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_0[_blk] = _vptr_0[_blk];
                    }
                }
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(residual + (((wrow_base < M) ? wrow_base : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_1 = reinterpret_cast<uint4*>(&wrw[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_1[_blk] = _vptr_1[_blk];
                    }
                }
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? wepi_gidx[1] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_2 = reinterpret_cast<uint4*>(&wgw[4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_2[_blk] = _vptr_2[_blk];
                    }
                }
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 4 < M) ? wrow_base + 4 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_3 = reinterpret_cast<uint4*>(&wrw[4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_3[_blk] = _vptr_3[_blk];
                    }
                }
                float wscaled[16];
                wscaled[0] = accum[0] * sa[0] * sw[0];
                wscaled[1] = accum[1] * sa[0] * sw[1];
                wscaled[2] = accum[4] * sa[0] * sw[2];
                wscaled[3] = accum[5] * sa[0] * sw[3];
                wscaled[4] = accum[8] * sa[0] * sw[4];
                wscaled[5] = accum[9] * sa[0] * sw[5];
                wscaled[6] = accum[12] * sa[0] * sw[6];
                wscaled[7] = accum[13] * sa[0] * sw[7];
                wscaled[8] = accum[16] * sa[0] * sw[8];
                wscaled[9] = accum[17] * sa[0] * sw[9];
                wscaled[10] = accum[20] * sa[0] * sw[10];
                wscaled[11] = accum[21] * sa[0] * sw[11];
                wscaled[12] = accum[24] * sa[0] * sw[12];
                wscaled[13] = accum[25] * sa[0] * sw[13];
                wscaled[14] = accum[28] * sa[0] * sw[14];
                wscaled[15] = accum[29] * sa[0] * sw[15];
                unsigned int wscaled_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled[_lp*2 + 0], wscaled[_lp*2+1 + 0]));
                    wscaled_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? wepi_gidx[2] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_6 = reinterpret_cast<uint4*>(&wgw[8]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_6[_blk] = _vptr_6[_blk];
                    }
                }
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 8 < M) ? wrow_base + 8 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_7 = reinterpret_cast<uint4*>(&wrw[8]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_7[_blk] = _vptr_7[_blk];
                    }
                }
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? wepi_gidx[3] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_8 = reinterpret_cast<uint4*>(&wgw[12]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_8[_blk] = _vptr_8[_blk];
                    }
                }
                {
                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 12 < M) ? wrow_base + 12 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_9 = reinterpret_cast<uint4*>(&wrw[12]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_9[_blk] = _vptr_9[_blk];
                    }
                }
                float _unpack_bf16_4[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words[_pair]) << 16);
                    (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_5[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[_pair]) << 16);
                    (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[_pair]) & 0xffff0000u);
                }
                float wepi_p0_0[8];
                wepi_p0_0[0] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_5[0] : 0.0f) * _unpack_bf16_4[0];
                wepi_p0_0[1] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_5[1] : 0.0f) * _unpack_bf16_4[1];
                wepi_p0_0[2] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_5[2] : 0.0f) * _unpack_bf16_4[2];
                wepi_p0_0[3] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_5[3] : 0.0f) * _unpack_bf16_4[3];
                wepi_p0_0[4] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_5[4] : 0.0f) * _unpack_bf16_4[4];
                wepi_p0_0[5] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_5[5] : 0.0f) * _unpack_bf16_4[5];
                wepi_p0_0[6] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_5[6] : 0.0f) * _unpack_bf16_4[6];
                wepi_p0_0[7] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_5[7] : 0.0f) * _unpack_bf16_4[7];
                unsigned int wepi_p0_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p0_0[_lp*2 + 0], wepi_p0_0[_lp*2+1 + 0]));
                    wepi_p0_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_6[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p0_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p0_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_7[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[_pair]) << 16);
                    (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[_pair]) & 0xffff0000u);
                }
                float wepi_out0_0[8];
                wepi_out0_0[0] = _unpack_bf16_7[0] + _unpack_bf16_6[0];
                wepi_out0_0[1] = _unpack_bf16_7[1] + _unpack_bf16_6[1];
                wepi_out0_0[2] = _unpack_bf16_7[2] + _unpack_bf16_6[2];
                wepi_out0_0[3] = _unpack_bf16_7[3] + _unpack_bf16_6[3];
                wepi_out0_0[4] = _unpack_bf16_7[4] + _unpack_bf16_6[4];
                wepi_out0_0[5] = _unpack_bf16_7[5] + _unpack_bf16_6[5];
                wepi_out0_0[6] = _unpack_bf16_7[6] + _unpack_bf16_6[6];
                wepi_out0_0[7] = _unpack_bf16_7[7] + _unpack_bf16_6[7];
                unsigned int wepi_out0_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out0_0[_lp*2 + 0], wepi_out0_0[_lp*2+1 + 0]));
                    wepi_out0_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + ((wrow_base * 5376 + wgcol) / 2)), "r"((wepi_out0_0_bf16 + 0)[(0) + 0]), "r"((wepi_out0_0_bf16 + 0)[(0) + 1]), "r"((wepi_out0_0_bf16 + 0)[(0) + 2]), "r"((wepi_out0_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base < M)) : "memory");
                float _unpack_bf16_8[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words[4 + _pair]) << 16);
                    (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_9[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[4 + _pair]) << 16);
                    (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[4 + _pair]) & 0xffff0000u);
                }
                float wepi_p0_1[8];
                wepi_p0_1[0] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_9[0] : 0.0f) * _unpack_bf16_8[0];
                wepi_p0_1[1] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_9[1] : 0.0f) * _unpack_bf16_8[1];
                wepi_p0_1[2] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_9[2] : 0.0f) * _unpack_bf16_8[2];
                wepi_p0_1[3] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_9[3] : 0.0f) * _unpack_bf16_8[3];
                wepi_p0_1[4] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_9[4] : 0.0f) * _unpack_bf16_8[4];
                wepi_p0_1[5] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_9[5] : 0.0f) * _unpack_bf16_8[5];
                wepi_p0_1[6] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_9[6] : 0.0f) * _unpack_bf16_8[6];
                wepi_p0_1[7] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_9[7] : 0.0f) * _unpack_bf16_8[7];
                unsigned int wepi_p0_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p0_1[_lp*2 + 0], wepi_p0_1[_lp*2+1 + 0]));
                    wepi_p0_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_10[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p0_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p0_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_11[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[4 + _pair]) << 16);
                    (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[4 + _pair]) & 0xffff0000u);
                }
                float wepi_out0_1[8];
                wepi_out0_1[0] = _unpack_bf16_11[0] + _unpack_bf16_10[0];
                wepi_out0_1[1] = _unpack_bf16_11[1] + _unpack_bf16_10[1];
                wepi_out0_1[2] = _unpack_bf16_11[2] + _unpack_bf16_10[2];
                wepi_out0_1[3] = _unpack_bf16_11[3] + _unpack_bf16_10[3];
                wepi_out0_1[4] = _unpack_bf16_11[4] + _unpack_bf16_10[4];
                wepi_out0_1[5] = _unpack_bf16_11[5] + _unpack_bf16_10[5];
                wepi_out0_1[6] = _unpack_bf16_11[6] + _unpack_bf16_10[6];
                wepi_out0_1[7] = _unpack_bf16_11[7] + _unpack_bf16_10[7];
                unsigned int wepi_out0_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out0_1[_lp*2 + 0], wepi_out0_1[_lp*2+1 + 0]));
                    wepi_out0_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 4) * 5376 + wgcol) / 2)), "r"((wepi_out0_1_bf16 + 0)[(0) + 0]), "r"((wepi_out0_1_bf16 + 0)[(0) + 1]), "r"((wepi_out0_1_bf16 + 0)[(0) + 2]), "r"((wepi_out0_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 4 < M)) : "memory");
                float wscaled_0[16];
                wscaled_0[0] = accum[2] * sa[1] * sw[0];
                wscaled_0[1] = accum[3] * sa[1] * sw[1];
                wscaled_0[2] = accum[6] * sa[1] * sw[2];
                wscaled_0[3] = accum[7] * sa[1] * sw[3];
                wscaled_0[4] = accum[10] * sa[1] * sw[4];
                wscaled_0[5] = accum[11] * sa[1] * sw[5];
                wscaled_0[6] = accum[14] * sa[1] * sw[6];
                wscaled_0[7] = accum[15] * sa[1] * sw[7];
                wscaled_0[8] = accum[18] * sa[1] * sw[8];
                wscaled_0[9] = accum[19] * sa[1] * sw[9];
                wscaled_0[10] = accum[22] * sa[1] * sw[10];
                wscaled_0[11] = accum[23] * sa[1] * sw[11];
                wscaled_0[12] = accum[26] * sa[1] * sw[12];
                wscaled_0[13] = accum[27] * sa[1] * sw[13];
                wscaled_0[14] = accum[30] * sa[1] * sw[14];
                wscaled_0[15] = accum[31] * sa[1] * sw[15];
                unsigned int wscaled_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled_0[_lp*2 + 0], wscaled_0[_lp*2+1 + 0]));
                    wscaled_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words_1[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                {
                    const uint4* _vptr_12 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? wepi_gidx[4] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_12 = reinterpret_cast<uint4*>(&wgw[16]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_12[_blk] = _vptr_12[_blk];
                    }
                }
                {
                    const uint4* _vptr_13 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 16 < M) ? wrow_base + 16 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_13 = reinterpret_cast<uint4*>(&wrw[16]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_13[_blk] = _vptr_13[_blk];
                    }
                }
                {
                    const uint4* _vptr_14 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? wepi_gidx[5] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_14 = reinterpret_cast<uint4*>(&wgw[20]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_14[_blk] = _vptr_14[_blk];
                    }
                }
                {
                    const uint4* _vptr_15 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 20 < M) ? wrow_base + 20 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_15 = reinterpret_cast<uint4*>(&wrw[20]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_15[_blk] = _vptr_15[_blk];
                    }
                }
                float _unpack_bf16_12[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_1[_pair]) << 16);
                    (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_1[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_13[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[8 + _pair]) << 16);
                    (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[8 + _pair]) & 0xffff0000u);
                }
                float wepi_p1_0[8];
                wepi_p1_0[0] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_13[0] : 0.0f) * _unpack_bf16_12[0];
                wepi_p1_0[1] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_13[1] : 0.0f) * _unpack_bf16_12[1];
                wepi_p1_0[2] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_13[2] : 0.0f) * _unpack_bf16_12[2];
                wepi_p1_0[3] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_13[3] : 0.0f) * _unpack_bf16_12[3];
                wepi_p1_0[4] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_13[4] : 0.0f) * _unpack_bf16_12[4];
                wepi_p1_0[5] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_13[5] : 0.0f) * _unpack_bf16_12[5];
                wepi_p1_0[6] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_13[6] : 0.0f) * _unpack_bf16_12[6];
                wepi_p1_0[7] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_13[7] : 0.0f) * _unpack_bf16_12[7];
                unsigned int wepi_p1_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p1_0[_lp*2 + 0], wepi_p1_0[_lp*2+1 + 0]));
                    wepi_p1_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_14[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p1_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p1_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_15[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[8 + _pair]) << 16);
                    (&_unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[8 + _pair]) & 0xffff0000u);
                }
                float wepi_out1_0[8];
                wepi_out1_0[0] = _unpack_bf16_15[0] + _unpack_bf16_14[0];
                wepi_out1_0[1] = _unpack_bf16_15[1] + _unpack_bf16_14[1];
                wepi_out1_0[2] = _unpack_bf16_15[2] + _unpack_bf16_14[2];
                wepi_out1_0[3] = _unpack_bf16_15[3] + _unpack_bf16_14[3];
                wepi_out1_0[4] = _unpack_bf16_15[4] + _unpack_bf16_14[4];
                wepi_out1_0[5] = _unpack_bf16_15[5] + _unpack_bf16_14[5];
                wepi_out1_0[6] = _unpack_bf16_15[6] + _unpack_bf16_14[6];
                wepi_out1_0[7] = _unpack_bf16_15[7] + _unpack_bf16_14[7];
                unsigned int wepi_out1_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out1_0[_lp*2 + 0], wepi_out1_0[_lp*2+1 + 0]));
                    wepi_out1_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 8) * 5376 + wgcol) / 2)), "r"((wepi_out1_0_bf16 + 0)[(0) + 0]), "r"((wepi_out1_0_bf16 + 0)[(0) + 1]), "r"((wepi_out1_0_bf16 + 0)[(0) + 2]), "r"((wepi_out1_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 8 < M)) : "memory");
                float _unpack_bf16_16[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_1[4 + _pair]) << 16);
                    (&_unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_1[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_17[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[12 + _pair]) << 16);
                    (&_unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[12 + _pair]) & 0xffff0000u);
                }
                float wepi_p1_1[8];
                wepi_p1_1[0] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_17[0] : 0.0f) * _unpack_bf16_16[0];
                wepi_p1_1[1] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_17[1] : 0.0f) * _unpack_bf16_16[1];
                wepi_p1_1[2] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_17[2] : 0.0f) * _unpack_bf16_16[2];
                wepi_p1_1[3] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_17[3] : 0.0f) * _unpack_bf16_16[3];
                wepi_p1_1[4] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_17[4] : 0.0f) * _unpack_bf16_16[4];
                wepi_p1_1[5] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_17[5] : 0.0f) * _unpack_bf16_16[5];
                wepi_p1_1[6] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_17[6] : 0.0f) * _unpack_bf16_16[6];
                wepi_p1_1[7] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_17[7] : 0.0f) * _unpack_bf16_16[7];
                unsigned int wepi_p1_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p1_1[_lp*2 + 0], wepi_p1_1[_lp*2+1 + 0]));
                    wepi_p1_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_18[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p1_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p1_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_19[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[12 + _pair]) << 16);
                    (&_unpack_bf16_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[12 + _pair]) & 0xffff0000u);
                }
                float wepi_out1_1[8];
                wepi_out1_1[0] = _unpack_bf16_19[0] + _unpack_bf16_18[0];
                wepi_out1_1[1] = _unpack_bf16_19[1] + _unpack_bf16_18[1];
                wepi_out1_1[2] = _unpack_bf16_19[2] + _unpack_bf16_18[2];
                wepi_out1_1[3] = _unpack_bf16_19[3] + _unpack_bf16_18[3];
                wepi_out1_1[4] = _unpack_bf16_19[4] + _unpack_bf16_18[4];
                wepi_out1_1[5] = _unpack_bf16_19[5] + _unpack_bf16_18[5];
                wepi_out1_1[6] = _unpack_bf16_19[6] + _unpack_bf16_18[6];
                wepi_out1_1[7] = _unpack_bf16_19[7] + _unpack_bf16_18[7];
                unsigned int wepi_out1_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out1_1[_lp*2 + 0], wepi_out1_1[_lp*2+1 + 0]));
                    wepi_out1_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 12) * 5376 + wgcol) / 2)), "r"((wepi_out1_1_bf16 + 0)[(0) + 0]), "r"((wepi_out1_1_bf16 + 0)[(0) + 1]), "r"((wepi_out1_1_bf16 + 0)[(0) + 2]), "r"((wepi_out1_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 12 < M)) : "memory");
                float wscaled_2[16];
                wscaled_2[0] = accum[32] * sa[2] * sw[0];
                wscaled_2[1] = accum[33] * sa[2] * sw[1];
                wscaled_2[2] = accum[36] * sa[2] * sw[2];
                wscaled_2[3] = accum[37] * sa[2] * sw[3];
                wscaled_2[4] = accum[40] * sa[2] * sw[4];
                wscaled_2[5] = accum[41] * sa[2] * sw[5];
                wscaled_2[6] = accum[44] * sa[2] * sw[6];
                wscaled_2[7] = accum[45] * sa[2] * sw[7];
                wscaled_2[8] = accum[48] * sa[2] * sw[8];
                wscaled_2[9] = accum[49] * sa[2] * sw[9];
                wscaled_2[10] = accum[52] * sa[2] * sw[10];
                wscaled_2[11] = accum[53] * sa[2] * sw[11];
                wscaled_2[12] = accum[56] * sa[2] * sw[12];
                wscaled_2[13] = accum[57] * sa[2] * sw[13];
                wscaled_2[14] = accum[60] * sa[2] * sw[14];
                wscaled_2[15] = accum[61] * sa[2] * sw[15];
                unsigned int wscaled_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled_2[_lp*2 + 0], wscaled_2[_lp*2+1 + 0]));
                    wscaled_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words_3[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                {
                    const uint4* _vptr_18 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? wepi_gidx[6] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_18 = reinterpret_cast<uint4*>(&wgw[24]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_18[_blk] = _vptr_18[_blk];
                    }
                }
                {
                    const uint4* _vptr_19 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 24 < M) ? wrow_base + 24 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_19 = reinterpret_cast<uint4*>(&wrw[24]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_19[_blk] = _vptr_19[_blk];
                    }
                }
                {
                    const uint4* _vptr_20 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? wepi_gidx[7] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_20 = reinterpret_cast<uint4*>(&wgw[28]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_20[_blk] = _vptr_20[_blk];
                    }
                }
                {
                    const uint4* _vptr_21 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 28 < M) ? wrow_base + 28 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_21 = reinterpret_cast<uint4*>(&wrw[28]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_21[_blk] = _vptr_21[_blk];
                    }
                }
                float _unpack_bf16_20[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_3[_pair]) << 16);
                    (&_unpack_bf16_20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_3[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_21[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_21[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[16 + _pair]) << 16);
                    (&_unpack_bf16_21[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[16 + _pair]) & 0xffff0000u);
                }
                float wepi_p2_0[8];
                wepi_p2_0[0] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_21[0] : 0.0f) * _unpack_bf16_20[0];
                wepi_p2_0[1] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_21[1] : 0.0f) * _unpack_bf16_20[1];
                wepi_p2_0[2] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_21[2] : 0.0f) * _unpack_bf16_20[2];
                wepi_p2_0[3] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_21[3] : 0.0f) * _unpack_bf16_20[3];
                wepi_p2_0[4] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_21[4] : 0.0f) * _unpack_bf16_20[4];
                wepi_p2_0[5] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_21[5] : 0.0f) * _unpack_bf16_20[5];
                wepi_p2_0[6] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_21[6] : 0.0f) * _unpack_bf16_20[6];
                wepi_p2_0[7] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_21[7] : 0.0f) * _unpack_bf16_20[7];
                unsigned int wepi_p2_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p2_0[_lp*2 + 0], wepi_p2_0[_lp*2+1 + 0]));
                    wepi_p2_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_22[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_22[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p2_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_22[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p2_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_23[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_23[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[16 + _pair]) << 16);
                    (&_unpack_bf16_23[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[16 + _pair]) & 0xffff0000u);
                }
                float wepi_out2_0[8];
                wepi_out2_0[0] = _unpack_bf16_23[0] + _unpack_bf16_22[0];
                wepi_out2_0[1] = _unpack_bf16_23[1] + _unpack_bf16_22[1];
                wepi_out2_0[2] = _unpack_bf16_23[2] + _unpack_bf16_22[2];
                wepi_out2_0[3] = _unpack_bf16_23[3] + _unpack_bf16_22[3];
                wepi_out2_0[4] = _unpack_bf16_23[4] + _unpack_bf16_22[4];
                wepi_out2_0[5] = _unpack_bf16_23[5] + _unpack_bf16_22[5];
                wepi_out2_0[6] = _unpack_bf16_23[6] + _unpack_bf16_22[6];
                wepi_out2_0[7] = _unpack_bf16_23[7] + _unpack_bf16_22[7];
                unsigned int wepi_out2_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out2_0[_lp*2 + 0], wepi_out2_0[_lp*2+1 + 0]));
                    wepi_out2_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 16) * 5376 + wgcol) / 2)), "r"((wepi_out2_0_bf16 + 0)[(0) + 0]), "r"((wepi_out2_0_bf16 + 0)[(0) + 1]), "r"((wepi_out2_0_bf16 + 0)[(0) + 2]), "r"((wepi_out2_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 16 < M)) : "memory");
                float _unpack_bf16_24[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_24[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_3[4 + _pair]) << 16);
                    (&_unpack_bf16_24[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_3[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_25[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_25[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[20 + _pair]) << 16);
                    (&_unpack_bf16_25[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[20 + _pair]) & 0xffff0000u);
                }
                float wepi_p2_1[8];
                wepi_p2_1[0] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_25[0] : 0.0f) * _unpack_bf16_24[0];
                wepi_p2_1[1] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_25[1] : 0.0f) * _unpack_bf16_24[1];
                wepi_p2_1[2] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_25[2] : 0.0f) * _unpack_bf16_24[2];
                wepi_p2_1[3] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_25[3] : 0.0f) * _unpack_bf16_24[3];
                wepi_p2_1[4] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_25[4] : 0.0f) * _unpack_bf16_24[4];
                wepi_p2_1[5] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_25[5] : 0.0f) * _unpack_bf16_24[5];
                wepi_p2_1[6] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_25[6] : 0.0f) * _unpack_bf16_24[6];
                wepi_p2_1[7] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_25[7] : 0.0f) * _unpack_bf16_24[7];
                unsigned int wepi_p2_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p2_1[_lp*2 + 0], wepi_p2_1[_lp*2+1 + 0]));
                    wepi_p2_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_26[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_26[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p2_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_26[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p2_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_27[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_27[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[20 + _pair]) << 16);
                    (&_unpack_bf16_27[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[20 + _pair]) & 0xffff0000u);
                }
                float wepi_out2_1[8];
                wepi_out2_1[0] = _unpack_bf16_27[0] + _unpack_bf16_26[0];
                wepi_out2_1[1] = _unpack_bf16_27[1] + _unpack_bf16_26[1];
                wepi_out2_1[2] = _unpack_bf16_27[2] + _unpack_bf16_26[2];
                wepi_out2_1[3] = _unpack_bf16_27[3] + _unpack_bf16_26[3];
                wepi_out2_1[4] = _unpack_bf16_27[4] + _unpack_bf16_26[4];
                wepi_out2_1[5] = _unpack_bf16_27[5] + _unpack_bf16_26[5];
                wepi_out2_1[6] = _unpack_bf16_27[6] + _unpack_bf16_26[6];
                wepi_out2_1[7] = _unpack_bf16_27[7] + _unpack_bf16_26[7];
                unsigned int wepi_out2_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out2_1[_lp*2 + 0], wepi_out2_1[_lp*2+1 + 0]));
                    wepi_out2_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 20) * 5376 + wgcol) / 2)), "r"((wepi_out2_1_bf16 + 0)[(0) + 0]), "r"((wepi_out2_1_bf16 + 0)[(0) + 1]), "r"((wepi_out2_1_bf16 + 0)[(0) + 2]), "r"((wepi_out2_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 20 < M)) : "memory");
                float wscaled_4[16];
                wscaled_4[0] = accum[34] * sa[3] * sw[0];
                wscaled_4[1] = accum[35] * sa[3] * sw[1];
                wscaled_4[2] = accum[38] * sa[3] * sw[2];
                wscaled_4[3] = accum[39] * sa[3] * sw[3];
                wscaled_4[4] = accum[42] * sa[3] * sw[4];
                wscaled_4[5] = accum[43] * sa[3] * sw[5];
                wscaled_4[6] = accum[46] * sa[3] * sw[6];
                wscaled_4[7] = accum[47] * sa[3] * sw[7];
                wscaled_4[8] = accum[50] * sa[3] * sw[8];
                wscaled_4[9] = accum[51] * sa[3] * sw[9];
                wscaled_4[10] = accum[54] * sa[3] * sw[10];
                wscaled_4[11] = accum[55] * sa[3] * sw[11];
                wscaled_4[12] = accum[58] * sa[3] * sw[12];
                wscaled_4[13] = accum[59] * sa[3] * sw[13];
                wscaled_4[14] = accum[62] * sa[3] * sw[14];
                wscaled_4[15] = accum[63] * sa[3] * sw[15];
                unsigned int wscaled_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled_4[_lp*2 + 0], wscaled_4[_lp*2+1 + 0]));
                    wscaled_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words_5[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                {
                    const uint4* _vptr_24 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? wepi_gidx[8] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_24 = reinterpret_cast<uint4*>(&wgw[32]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_24[_blk] = _vptr_24[_blk];
                    }
                }
                {
                    const uint4* _vptr_25 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 32 < M) ? wrow_base + 32 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_25 = reinterpret_cast<uint4*>(&wrw[32]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_25[_blk] = _vptr_25[_blk];
                    }
                }
                {
                    const uint4* _vptr_26 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? wepi_gidx[9] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_26 = reinterpret_cast<uint4*>(&wgw[36]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_26[_blk] = _vptr_26[_blk];
                    }
                }
                {
                    const uint4* _vptr_27 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 36 < M) ? wrow_base + 36 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_27 = reinterpret_cast<uint4*>(&wrw[36]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_27[_blk] = _vptr_27[_blk];
                    }
                }
                float _unpack_bf16_28[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_28[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_5[_pair]) << 16);
                    (&_unpack_bf16_28[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_5[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_29[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_29[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[24 + _pair]) << 16);
                    (&_unpack_bf16_29[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[24 + _pair]) & 0xffff0000u);
                }
                float wepi_p3_0[8];
                wepi_p3_0[0] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_29[0] : 0.0f) * _unpack_bf16_28[0];
                wepi_p3_0[1] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_29[1] : 0.0f) * _unpack_bf16_28[1];
                wepi_p3_0[2] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_29[2] : 0.0f) * _unpack_bf16_28[2];
                wepi_p3_0[3] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_29[3] : 0.0f) * _unpack_bf16_28[3];
                wepi_p3_0[4] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_29[4] : 0.0f) * _unpack_bf16_28[4];
                wepi_p3_0[5] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_29[5] : 0.0f) * _unpack_bf16_28[5];
                wepi_p3_0[6] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_29[6] : 0.0f) * _unpack_bf16_28[6];
                wepi_p3_0[7] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_29[7] : 0.0f) * _unpack_bf16_28[7];
                unsigned int wepi_p3_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p3_0[_lp*2 + 0], wepi_p3_0[_lp*2+1 + 0]));
                    wepi_p3_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_30[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_30[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p3_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_30[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p3_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_31[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_31[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[24 + _pair]) << 16);
                    (&_unpack_bf16_31[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[24 + _pair]) & 0xffff0000u);
                }
                float wepi_out3_0[8];
                wepi_out3_0[0] = _unpack_bf16_31[0] + _unpack_bf16_30[0];
                wepi_out3_0[1] = _unpack_bf16_31[1] + _unpack_bf16_30[1];
                wepi_out3_0[2] = _unpack_bf16_31[2] + _unpack_bf16_30[2];
                wepi_out3_0[3] = _unpack_bf16_31[3] + _unpack_bf16_30[3];
                wepi_out3_0[4] = _unpack_bf16_31[4] + _unpack_bf16_30[4];
                wepi_out3_0[5] = _unpack_bf16_31[5] + _unpack_bf16_30[5];
                wepi_out3_0[6] = _unpack_bf16_31[6] + _unpack_bf16_30[6];
                wepi_out3_0[7] = _unpack_bf16_31[7] + _unpack_bf16_30[7];
                unsigned int wepi_out3_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out3_0[_lp*2 + 0], wepi_out3_0[_lp*2+1 + 0]));
                    wepi_out3_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 24) * 5376 + wgcol) / 2)), "r"((wepi_out3_0_bf16 + 0)[(0) + 0]), "r"((wepi_out3_0_bf16 + 0)[(0) + 1]), "r"((wepi_out3_0_bf16 + 0)[(0) + 2]), "r"((wepi_out3_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 24 < M)) : "memory");
                float _unpack_bf16_32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_5[4 + _pair]) << 16);
                    (&_unpack_bf16_32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_5[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_33[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_33[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[28 + _pair]) << 16);
                    (&_unpack_bf16_33[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[28 + _pair]) & 0xffff0000u);
                }
                float wepi_p3_1[8];
                wepi_p3_1[0] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_33[0] : 0.0f) * _unpack_bf16_32[0];
                wepi_p3_1[1] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_33[1] : 0.0f) * _unpack_bf16_32[1];
                wepi_p3_1[2] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_33[2] : 0.0f) * _unpack_bf16_32[2];
                wepi_p3_1[3] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_33[3] : 0.0f) * _unpack_bf16_32[3];
                wepi_p3_1[4] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_33[4] : 0.0f) * _unpack_bf16_32[4];
                wepi_p3_1[5] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_33[5] : 0.0f) * _unpack_bf16_32[5];
                wepi_p3_1[6] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_33[6] : 0.0f) * _unpack_bf16_32[6];
                wepi_p3_1[7] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_33[7] : 0.0f) * _unpack_bf16_32[7];
                unsigned int wepi_p3_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p3_1[_lp*2 + 0], wepi_p3_1[_lp*2+1 + 0]));
                    wepi_p3_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_34[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_34[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p3_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_34[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p3_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_35[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_35[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[28 + _pair]) << 16);
                    (&_unpack_bf16_35[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[28 + _pair]) & 0xffff0000u);
                }
                float wepi_out3_1[8];
                wepi_out3_1[0] = _unpack_bf16_35[0] + _unpack_bf16_34[0];
                wepi_out3_1[1] = _unpack_bf16_35[1] + _unpack_bf16_34[1];
                wepi_out3_1[2] = _unpack_bf16_35[2] + _unpack_bf16_34[2];
                wepi_out3_1[3] = _unpack_bf16_35[3] + _unpack_bf16_34[3];
                wepi_out3_1[4] = _unpack_bf16_35[4] + _unpack_bf16_34[4];
                wepi_out3_1[5] = _unpack_bf16_35[5] + _unpack_bf16_34[5];
                wepi_out3_1[6] = _unpack_bf16_35[6] + _unpack_bf16_34[6];
                wepi_out3_1[7] = _unpack_bf16_35[7] + _unpack_bf16_34[7];
                unsigned int wepi_out3_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out3_1[_lp*2 + 0], wepi_out3_1[_lp*2+1 + 0]));
                    wepi_out3_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 28) * 5376 + wgcol) / 2)), "r"((wepi_out3_1_bf16 + 0)[(0) + 0]), "r"((wepi_out3_1_bf16 + 0)[(0) + 1]), "r"((wepi_out3_1_bf16 + 0)[(0) + 2]), "r"((wepi_out3_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 28 < M)) : "memory");
                float wscaled_6[16];
                wscaled_6[0] = accum[64] * sa[4] * sw[0];
                wscaled_6[1] = accum[65] * sa[4] * sw[1];
                wscaled_6[2] = accum[68] * sa[4] * sw[2];
                wscaled_6[3] = accum[69] * sa[4] * sw[3];
                wscaled_6[4] = accum[72] * sa[4] * sw[4];
                wscaled_6[5] = accum[73] * sa[4] * sw[5];
                wscaled_6[6] = accum[76] * sa[4] * sw[6];
                wscaled_6[7] = accum[77] * sa[4] * sw[7];
                wscaled_6[8] = accum[80] * sa[4] * sw[8];
                wscaled_6[9] = accum[81] * sa[4] * sw[9];
                wscaled_6[10] = accum[84] * sa[4] * sw[10];
                wscaled_6[11] = accum[85] * sa[4] * sw[11];
                wscaled_6[12] = accum[88] * sa[4] * sw[12];
                wscaled_6[13] = accum[89] * sa[4] * sw[13];
                wscaled_6[14] = accum[92] * sa[4] * sw[14];
                wscaled_6[15] = accum[93] * sa[4] * sw[15];
                unsigned int wscaled_6_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled_6[_lp*2 + 0], wscaled_6[_lp*2+1 + 0]));
                    wscaled_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_28 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_29 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words_7[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                {
                    const uint4* _vptr_30 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? wepi_gidx[10] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_30 = reinterpret_cast<uint4*>(&wgw[40]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_30[_blk] = _vptr_30[_blk];
                    }
                }
                {
                    const uint4* _vptr_31 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 40 < M) ? wrow_base + 40 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_31 = reinterpret_cast<uint4*>(&wrw[40]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_31[_blk] = _vptr_31[_blk];
                    }
                }
                {
                    const uint4* _vptr_32 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? wepi_gidx[11] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_32 = reinterpret_cast<uint4*>(&wgw[44]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_32[_blk] = _vptr_32[_blk];
                    }
                }
                {
                    const uint4* _vptr_33 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 44 < M) ? wrow_base + 44 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_33 = reinterpret_cast<uint4*>(&wrw[44]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_33[_blk] = _vptr_33[_blk];
                    }
                }
                float _unpack_bf16_36[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_36[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_7[_pair]) << 16);
                    (&_unpack_bf16_36[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_7[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_37[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_37[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[32 + _pair]) << 16);
                    (&_unpack_bf16_37[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[32 + _pair]) & 0xffff0000u);
                }
                float wepi_p4_0[8];
                wepi_p4_0[0] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_37[0] : 0.0f) * _unpack_bf16_36[0];
                wepi_p4_0[1] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_37[1] : 0.0f) * _unpack_bf16_36[1];
                wepi_p4_0[2] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_37[2] : 0.0f) * _unpack_bf16_36[2];
                wepi_p4_0[3] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_37[3] : 0.0f) * _unpack_bf16_36[3];
                wepi_p4_0[4] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_37[4] : 0.0f) * _unpack_bf16_36[4];
                wepi_p4_0[5] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_37[5] : 0.0f) * _unpack_bf16_36[5];
                wepi_p4_0[6] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_37[6] : 0.0f) * _unpack_bf16_36[6];
                wepi_p4_0[7] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_37[7] : 0.0f) * _unpack_bf16_36[7];
                unsigned int wepi_p4_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p4_0[_lp*2 + 0], wepi_p4_0[_lp*2+1 + 0]));
                    wepi_p4_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_38[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_38[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p4_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_38[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p4_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_39[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_39[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[32 + _pair]) << 16);
                    (&_unpack_bf16_39[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[32 + _pair]) & 0xffff0000u);
                }
                float wepi_out4_0[8];
                wepi_out4_0[0] = _unpack_bf16_39[0] + _unpack_bf16_38[0];
                wepi_out4_0[1] = _unpack_bf16_39[1] + _unpack_bf16_38[1];
                wepi_out4_0[2] = _unpack_bf16_39[2] + _unpack_bf16_38[2];
                wepi_out4_0[3] = _unpack_bf16_39[3] + _unpack_bf16_38[3];
                wepi_out4_0[4] = _unpack_bf16_39[4] + _unpack_bf16_38[4];
                wepi_out4_0[5] = _unpack_bf16_39[5] + _unpack_bf16_38[5];
                wepi_out4_0[6] = _unpack_bf16_39[6] + _unpack_bf16_38[6];
                wepi_out4_0[7] = _unpack_bf16_39[7] + _unpack_bf16_38[7];
                unsigned int wepi_out4_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out4_0[_lp*2 + 0], wepi_out4_0[_lp*2+1 + 0]));
                    wepi_out4_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 32) * 5376 + wgcol) / 2)), "r"((wepi_out4_0_bf16 + 0)[(0) + 0]), "r"((wepi_out4_0_bf16 + 0)[(0) + 1]), "r"((wepi_out4_0_bf16 + 0)[(0) + 2]), "r"((wepi_out4_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 32 < M)) : "memory");
                float _unpack_bf16_40[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_40[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_7[4 + _pair]) << 16);
                    (&_unpack_bf16_40[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_7[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_41[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_41[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[36 + _pair]) << 16);
                    (&_unpack_bf16_41[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[36 + _pair]) & 0xffff0000u);
                }
                float wepi_p4_1[8];
                wepi_p4_1[0] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_41[0] : 0.0f) * _unpack_bf16_40[0];
                wepi_p4_1[1] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_41[1] : 0.0f) * _unpack_bf16_40[1];
                wepi_p4_1[2] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_41[2] : 0.0f) * _unpack_bf16_40[2];
                wepi_p4_1[3] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_41[3] : 0.0f) * _unpack_bf16_40[3];
                wepi_p4_1[4] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_41[4] : 0.0f) * _unpack_bf16_40[4];
                wepi_p4_1[5] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_41[5] : 0.0f) * _unpack_bf16_40[5];
                wepi_p4_1[6] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_41[6] : 0.0f) * _unpack_bf16_40[6];
                wepi_p4_1[7] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_41[7] : 0.0f) * _unpack_bf16_40[7];
                unsigned int wepi_p4_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p4_1[_lp*2 + 0], wepi_p4_1[_lp*2+1 + 0]));
                    wepi_p4_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_42[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_42[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p4_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_42[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p4_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_43[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_43[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[36 + _pair]) << 16);
                    (&_unpack_bf16_43[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[36 + _pair]) & 0xffff0000u);
                }
                float wepi_out4_1[8];
                wepi_out4_1[0] = _unpack_bf16_43[0] + _unpack_bf16_42[0];
                wepi_out4_1[1] = _unpack_bf16_43[1] + _unpack_bf16_42[1];
                wepi_out4_1[2] = _unpack_bf16_43[2] + _unpack_bf16_42[2];
                wepi_out4_1[3] = _unpack_bf16_43[3] + _unpack_bf16_42[3];
                wepi_out4_1[4] = _unpack_bf16_43[4] + _unpack_bf16_42[4];
                wepi_out4_1[5] = _unpack_bf16_43[5] + _unpack_bf16_42[5];
                wepi_out4_1[6] = _unpack_bf16_43[6] + _unpack_bf16_42[6];
                wepi_out4_1[7] = _unpack_bf16_43[7] + _unpack_bf16_42[7];
                unsigned int wepi_out4_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out4_1[_lp*2 + 0], wepi_out4_1[_lp*2+1 + 0]));
                    wepi_out4_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 36) * 5376 + wgcol) / 2)), "r"((wepi_out4_1_bf16 + 0)[(0) + 0]), "r"((wepi_out4_1_bf16 + 0)[(0) + 1]), "r"((wepi_out4_1_bf16 + 0)[(0) + 2]), "r"((wepi_out4_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 36 < M)) : "memory");
                float wscaled_8[16];
                wscaled_8[0] = accum[66] * sa[5] * sw[0];
                wscaled_8[1] = accum[67] * sa[5] * sw[1];
                wscaled_8[2] = accum[70] * sa[5] * sw[2];
                wscaled_8[3] = accum[71] * sa[5] * sw[3];
                wscaled_8[4] = accum[74] * sa[5] * sw[4];
                wscaled_8[5] = accum[75] * sa[5] * sw[5];
                wscaled_8[6] = accum[78] * sa[5] * sw[6];
                wscaled_8[7] = accum[79] * sa[5] * sw[7];
                wscaled_8[8] = accum[82] * sa[5] * sw[8];
                wscaled_8[9] = accum[83] * sa[5] * sw[9];
                wscaled_8[10] = accum[86] * sa[5] * sw[10];
                wscaled_8[11] = accum[87] * sa[5] * sw[11];
                wscaled_8[12] = accum[90] * sa[5] * sw[12];
                wscaled_8[13] = accum[91] * sa[5] * sw[13];
                wscaled_8[14] = accum[94] * sa[5] * sw[14];
                wscaled_8[15] = accum[95] * sa[5] * sw[15];
                unsigned int wscaled_8_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled_8[_lp*2 + 0], wscaled_8[_lp*2+1 + 0]));
                    wscaled_8_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words_9[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                {
                    const uint4* _vptr_36 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? wepi_gidx[12] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_36 = reinterpret_cast<uint4*>(&wgw[48]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_36[_blk] = _vptr_36[_blk];
                    }
                }
                {
                    const uint4* _vptr_37 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 48 < M) ? wrow_base + 48 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_37 = reinterpret_cast<uint4*>(&wrw[48]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_37[_blk] = _vptr_37[_blk];
                    }
                }
                {
                    const uint4* _vptr_38 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? wepi_gidx[13] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_38 = reinterpret_cast<uint4*>(&wgw[52]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_38[_blk] = _vptr_38[_blk];
                    }
                }
                {
                    const uint4* _vptr_39 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 52 < M) ? wrow_base + 52 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_39 = reinterpret_cast<uint4*>(&wrw[52]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_39[_blk] = _vptr_39[_blk];
                    }
                }
                float _unpack_bf16_44[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_44[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_9[_pair]) << 16);
                    (&_unpack_bf16_44[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_9[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_45[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_45[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[40 + _pair]) << 16);
                    (&_unpack_bf16_45[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[40 + _pair]) & 0xffff0000u);
                }
                float wepi_p5_0[8];
                wepi_p5_0[0] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_45[0] : 0.0f) * _unpack_bf16_44[0];
                wepi_p5_0[1] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_45[1] : 0.0f) * _unpack_bf16_44[1];
                wepi_p5_0[2] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_45[2] : 0.0f) * _unpack_bf16_44[2];
                wepi_p5_0[3] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_45[3] : 0.0f) * _unpack_bf16_44[3];
                wepi_p5_0[4] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_45[4] : 0.0f) * _unpack_bf16_44[4];
                wepi_p5_0[5] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_45[5] : 0.0f) * _unpack_bf16_44[5];
                wepi_p5_0[6] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_45[6] : 0.0f) * _unpack_bf16_44[6];
                wepi_p5_0[7] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_45[7] : 0.0f) * _unpack_bf16_44[7];
                unsigned int wepi_p5_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p5_0[_lp*2 + 0], wepi_p5_0[_lp*2+1 + 0]));
                    wepi_p5_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_46[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_46[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p5_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_46[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p5_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_47[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_47[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[40 + _pair]) << 16);
                    (&_unpack_bf16_47[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[40 + _pair]) & 0xffff0000u);
                }
                float wepi_out5_0[8];
                wepi_out5_0[0] = _unpack_bf16_47[0] + _unpack_bf16_46[0];
                wepi_out5_0[1] = _unpack_bf16_47[1] + _unpack_bf16_46[1];
                wepi_out5_0[2] = _unpack_bf16_47[2] + _unpack_bf16_46[2];
                wepi_out5_0[3] = _unpack_bf16_47[3] + _unpack_bf16_46[3];
                wepi_out5_0[4] = _unpack_bf16_47[4] + _unpack_bf16_46[4];
                wepi_out5_0[5] = _unpack_bf16_47[5] + _unpack_bf16_46[5];
                wepi_out5_0[6] = _unpack_bf16_47[6] + _unpack_bf16_46[6];
                wepi_out5_0[7] = _unpack_bf16_47[7] + _unpack_bf16_46[7];
                unsigned int wepi_out5_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out5_0[_lp*2 + 0], wepi_out5_0[_lp*2+1 + 0]));
                    wepi_out5_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 40) * 5376 + wgcol) / 2)), "r"((wepi_out5_0_bf16 + 0)[(0) + 0]), "r"((wepi_out5_0_bf16 + 0)[(0) + 1]), "r"((wepi_out5_0_bf16 + 0)[(0) + 2]), "r"((wepi_out5_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 40 < M)) : "memory");
                float _unpack_bf16_48[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_48[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_9[4 + _pair]) << 16);
                    (&_unpack_bf16_48[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_9[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_49[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_49[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[44 + _pair]) << 16);
                    (&_unpack_bf16_49[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[44 + _pair]) & 0xffff0000u);
                }
                float wepi_p5_1[8];
                wepi_p5_1[0] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_49[0] : 0.0f) * _unpack_bf16_48[0];
                wepi_p5_1[1] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_49[1] : 0.0f) * _unpack_bf16_48[1];
                wepi_p5_1[2] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_49[2] : 0.0f) * _unpack_bf16_48[2];
                wepi_p5_1[3] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_49[3] : 0.0f) * _unpack_bf16_48[3];
                wepi_p5_1[4] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_49[4] : 0.0f) * _unpack_bf16_48[4];
                wepi_p5_1[5] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_49[5] : 0.0f) * _unpack_bf16_48[5];
                wepi_p5_1[6] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_49[6] : 0.0f) * _unpack_bf16_48[6];
                wepi_p5_1[7] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_49[7] : 0.0f) * _unpack_bf16_48[7];
                unsigned int wepi_p5_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p5_1[_lp*2 + 0], wepi_p5_1[_lp*2+1 + 0]));
                    wepi_p5_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_50[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_50[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p5_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_50[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p5_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_51[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_51[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[44 + _pair]) << 16);
                    (&_unpack_bf16_51[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[44 + _pair]) & 0xffff0000u);
                }
                float wepi_out5_1[8];
                wepi_out5_1[0] = _unpack_bf16_51[0] + _unpack_bf16_50[0];
                wepi_out5_1[1] = _unpack_bf16_51[1] + _unpack_bf16_50[1];
                wepi_out5_1[2] = _unpack_bf16_51[2] + _unpack_bf16_50[2];
                wepi_out5_1[3] = _unpack_bf16_51[3] + _unpack_bf16_50[3];
                wepi_out5_1[4] = _unpack_bf16_51[4] + _unpack_bf16_50[4];
                wepi_out5_1[5] = _unpack_bf16_51[5] + _unpack_bf16_50[5];
                wepi_out5_1[6] = _unpack_bf16_51[6] + _unpack_bf16_50[6];
                wepi_out5_1[7] = _unpack_bf16_51[7] + _unpack_bf16_50[7];
                unsigned int wepi_out5_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out5_1[_lp*2 + 0], wepi_out5_1[_lp*2+1 + 0]));
                    wepi_out5_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 44) * 5376 + wgcol) / 2)), "r"((wepi_out5_1_bf16 + 0)[(0) + 0]), "r"((wepi_out5_1_bf16 + 0)[(0) + 1]), "r"((wepi_out5_1_bf16 + 0)[(0) + 2]), "r"((wepi_out5_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 44 < M)) : "memory");
                float wscaled_10[16];
                wscaled_10[0] = accum[96] * sa[6] * sw[0];
                wscaled_10[1] = accum[97] * sa[6] * sw[1];
                wscaled_10[2] = accum[100] * sa[6] * sw[2];
                wscaled_10[3] = accum[101] * sa[6] * sw[3];
                wscaled_10[4] = accum[104] * sa[6] * sw[4];
                wscaled_10[5] = accum[105] * sa[6] * sw[5];
                wscaled_10[6] = accum[108] * sa[6] * sw[6];
                wscaled_10[7] = accum[109] * sa[6] * sw[7];
                wscaled_10[8] = accum[112] * sa[6] * sw[8];
                wscaled_10[9] = accum[113] * sa[6] * sw[9];
                wscaled_10[10] = accum[116] * sa[6] * sw[10];
                wscaled_10[11] = accum[117] * sa[6] * sw[11];
                wscaled_10[12] = accum[120] * sa[6] * sw[12];
                wscaled_10[13] = accum[121] * sa[6] * sw[13];
                wscaled_10[14] = accum[124] * sa[6] * sw[14];
                wscaled_10[15] = accum[125] * sa[6] * sw[15];
                unsigned int wscaled_10_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled_10[_lp*2 + 0], wscaled_10[_lp*2+1 + 0]));
                    wscaled_10_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_40 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_40), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words_11[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                {
                    const uint4* _vptr_42 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? wepi_gidx[14] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_42 = reinterpret_cast<uint4*>(&wgw[56]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_42[_blk] = _vptr_42[_blk];
                    }
                }
                {
                    const uint4* _vptr_43 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 56 < M) ? wrow_base + 56 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_43 = reinterpret_cast<uint4*>(&wrw[56]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_43[_blk] = _vptr_43[_blk];
                    }
                }
                {
                    const uint4* _vptr_44 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? wepi_gidx[15] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_44 = reinterpret_cast<uint4*>(&wgw[60]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_44[_blk] = _vptr_44[_blk];
                    }
                }
                {
                    const uint4* _vptr_45 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 60 < M) ? wrow_base + 60 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_45 = reinterpret_cast<uint4*>(&wrw[60]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_45[_blk] = _vptr_45[_blk];
                    }
                }
                float _unpack_bf16_52[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_52[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_11[_pair]) << 16);
                    (&_unpack_bf16_52[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_11[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_53[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_53[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[48 + _pair]) << 16);
                    (&_unpack_bf16_53[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[48 + _pair]) & 0xffff0000u);
                }
                float wepi_p6_0[8];
                wepi_p6_0[0] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_53[0] : 0.0f) * _unpack_bf16_52[0];
                wepi_p6_0[1] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_53[1] : 0.0f) * _unpack_bf16_52[1];
                wepi_p6_0[2] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_53[2] : 0.0f) * _unpack_bf16_52[2];
                wepi_p6_0[3] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_53[3] : 0.0f) * _unpack_bf16_52[3];
                wepi_p6_0[4] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_53[4] : 0.0f) * _unpack_bf16_52[4];
                wepi_p6_0[5] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_53[5] : 0.0f) * _unpack_bf16_52[5];
                wepi_p6_0[6] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_53[6] : 0.0f) * _unpack_bf16_52[6];
                wepi_p6_0[7] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_53[7] : 0.0f) * _unpack_bf16_52[7];
                unsigned int wepi_p6_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p6_0[_lp*2 + 0], wepi_p6_0[_lp*2+1 + 0]));
                    wepi_p6_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_54[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_54[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p6_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_54[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p6_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_55[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_55[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[48 + _pair]) << 16);
                    (&_unpack_bf16_55[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[48 + _pair]) & 0xffff0000u);
                }
                float wepi_out6_0[8];
                wepi_out6_0[0] = _unpack_bf16_55[0] + _unpack_bf16_54[0];
                wepi_out6_0[1] = _unpack_bf16_55[1] + _unpack_bf16_54[1];
                wepi_out6_0[2] = _unpack_bf16_55[2] + _unpack_bf16_54[2];
                wepi_out6_0[3] = _unpack_bf16_55[3] + _unpack_bf16_54[3];
                wepi_out6_0[4] = _unpack_bf16_55[4] + _unpack_bf16_54[4];
                wepi_out6_0[5] = _unpack_bf16_55[5] + _unpack_bf16_54[5];
                wepi_out6_0[6] = _unpack_bf16_55[6] + _unpack_bf16_54[6];
                wepi_out6_0[7] = _unpack_bf16_55[7] + _unpack_bf16_54[7];
                unsigned int wepi_out6_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out6_0[_lp*2 + 0], wepi_out6_0[_lp*2+1 + 0]));
                    wepi_out6_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 48) * 5376 + wgcol) / 2)), "r"((wepi_out6_0_bf16 + 0)[(0) + 0]), "r"((wepi_out6_0_bf16 + 0)[(0) + 1]), "r"((wepi_out6_0_bf16 + 0)[(0) + 2]), "r"((wepi_out6_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 48 < M)) : "memory");
                float _unpack_bf16_56[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_56[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_11[4 + _pair]) << 16);
                    (&_unpack_bf16_56[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_11[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_57[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_57[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[52 + _pair]) << 16);
                    (&_unpack_bf16_57[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[52 + _pair]) & 0xffff0000u);
                }
                float wepi_p6_1[8];
                wepi_p6_1[0] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_57[0] : 0.0f) * _unpack_bf16_56[0];
                wepi_p6_1[1] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_57[1] : 0.0f) * _unpack_bf16_56[1];
                wepi_p6_1[2] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_57[2] : 0.0f) * _unpack_bf16_56[2];
                wepi_p6_1[3] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_57[3] : 0.0f) * _unpack_bf16_56[3];
                wepi_p6_1[4] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_57[4] : 0.0f) * _unpack_bf16_56[4];
                wepi_p6_1[5] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_57[5] : 0.0f) * _unpack_bf16_56[5];
                wepi_p6_1[6] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_57[6] : 0.0f) * _unpack_bf16_56[6];
                wepi_p6_1[7] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_57[7] : 0.0f) * _unpack_bf16_56[7];
                unsigned int wepi_p6_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p6_1[_lp*2 + 0], wepi_p6_1[_lp*2+1 + 0]));
                    wepi_p6_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_58[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_58[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p6_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_58[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p6_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_59[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_59[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[52 + _pair]) << 16);
                    (&_unpack_bf16_59[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[52 + _pair]) & 0xffff0000u);
                }
                float wepi_out6_1[8];
                wepi_out6_1[0] = _unpack_bf16_59[0] + _unpack_bf16_58[0];
                wepi_out6_1[1] = _unpack_bf16_59[1] + _unpack_bf16_58[1];
                wepi_out6_1[2] = _unpack_bf16_59[2] + _unpack_bf16_58[2];
                wepi_out6_1[3] = _unpack_bf16_59[3] + _unpack_bf16_58[3];
                wepi_out6_1[4] = _unpack_bf16_59[4] + _unpack_bf16_58[4];
                wepi_out6_1[5] = _unpack_bf16_59[5] + _unpack_bf16_58[5];
                wepi_out6_1[6] = _unpack_bf16_59[6] + _unpack_bf16_58[6];
                wepi_out6_1[7] = _unpack_bf16_59[7] + _unpack_bf16_58[7];
                unsigned int wepi_out6_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out6_1[_lp*2 + 0], wepi_out6_1[_lp*2+1 + 0]));
                    wepi_out6_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 52) * 5376 + wgcol) / 2)), "r"((wepi_out6_1_bf16 + 0)[(0) + 0]), "r"((wepi_out6_1_bf16 + 0)[(0) + 1]), "r"((wepi_out6_1_bf16 + 0)[(0) + 2]), "r"((wepi_out6_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 52 < M)) : "memory");
                float wscaled_12[16];
                wscaled_12[0] = accum[98] * sa[7] * sw[0];
                wscaled_12[1] = accum[99] * sa[7] * sw[1];
                wscaled_12[2] = accum[102] * sa[7] * sw[2];
                wscaled_12[3] = accum[103] * sa[7] * sw[3];
                wscaled_12[4] = accum[106] * sa[7] * sw[4];
                wscaled_12[5] = accum[107] * sa[7] * sw[5];
                wscaled_12[6] = accum[110] * sa[7] * sw[6];
                wscaled_12[7] = accum[111] * sa[7] * sw[7];
                wscaled_12[8] = accum[114] * sa[7] * sw[8];
                wscaled_12[9] = accum[115] * sa[7] * sw[9];
                wscaled_12[10] = accum[118] * sa[7] * sw[10];
                wscaled_12[11] = accum[119] * sa[7] * sw[11];
                wscaled_12[12] = accum[122] * sa[7] * sw[12];
                wscaled_12[13] = accum[123] * sa[7] * sw[13];
                wscaled_12[14] = accum[126] * sa[7] * sw[14];
                wscaled_12[15] = accum[127] * sa[7] * sw[15];
                unsigned int wscaled_12_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled_12[_lp*2 + 0], wscaled_12[_lp*2+1 + 0]));
                    wscaled_12_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_46 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_46), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_47 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_47), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words_13[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                float _unpack_bf16_60[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_60[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_13[_pair]) << 16);
                    (&_unpack_bf16_60[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_13[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_61[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_61[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[56 + _pair]) << 16);
                    (&_unpack_bf16_61[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[56 + _pair]) & 0xffff0000u);
                }
                float wepi_p7_0[8];
                wepi_p7_0[0] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_61[0] : 0.0f) * _unpack_bf16_60[0];
                wepi_p7_0[1] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_61[1] : 0.0f) * _unpack_bf16_60[1];
                wepi_p7_0[2] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_61[2] : 0.0f) * _unpack_bf16_60[2];
                wepi_p7_0[3] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_61[3] : 0.0f) * _unpack_bf16_60[3];
                wepi_p7_0[4] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_61[4] : 0.0f) * _unpack_bf16_60[4];
                wepi_p7_0[5] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_61[5] : 0.0f) * _unpack_bf16_60[5];
                wepi_p7_0[6] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_61[6] : 0.0f) * _unpack_bf16_60[6];
                wepi_p7_0[7] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_61[7] : 0.0f) * _unpack_bf16_60[7];
                unsigned int wepi_p7_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p7_0[_lp*2 + 0], wepi_p7_0[_lp*2+1 + 0]));
                    wepi_p7_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_62[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_62[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p7_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_62[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p7_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_63[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_63[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[56 + _pair]) << 16);
                    (&_unpack_bf16_63[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[56 + _pair]) & 0xffff0000u);
                }
                float wepi_out7_0[8];
                wepi_out7_0[0] = _unpack_bf16_63[0] + _unpack_bf16_62[0];
                wepi_out7_0[1] = _unpack_bf16_63[1] + _unpack_bf16_62[1];
                wepi_out7_0[2] = _unpack_bf16_63[2] + _unpack_bf16_62[2];
                wepi_out7_0[3] = _unpack_bf16_63[3] + _unpack_bf16_62[3];
                wepi_out7_0[4] = _unpack_bf16_63[4] + _unpack_bf16_62[4];
                wepi_out7_0[5] = _unpack_bf16_63[5] + _unpack_bf16_62[5];
                wepi_out7_0[6] = _unpack_bf16_63[6] + _unpack_bf16_62[6];
                wepi_out7_0[7] = _unpack_bf16_63[7] + _unpack_bf16_62[7];
                unsigned int wepi_out7_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out7_0[_lp*2 + 0], wepi_out7_0[_lp*2+1 + 0]));
                    wepi_out7_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 56) * 5376 + wgcol) / 2)), "r"((wepi_out7_0_bf16 + 0)[(0) + 0]), "r"((wepi_out7_0_bf16 + 0)[(0) + 1]), "r"((wepi_out7_0_bf16 + 0)[(0) + 2]), "r"((wepi_out7_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 56 < M)) : "memory");
                float _unpack_bf16_64[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_64[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_13[4 + _pair]) << 16);
                    (&_unpack_bf16_64[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_13[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_65[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_65[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[60 + _pair]) << 16);
                    (&_unpack_bf16_65[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[60 + _pair]) & 0xffff0000u);
                }
                float wepi_p7_1[8];
                wepi_p7_1[0] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_65[0] : 0.0f) * _unpack_bf16_64[0];
                wepi_p7_1[1] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_65[1] : 0.0f) * _unpack_bf16_64[1];
                wepi_p7_1[2] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_65[2] : 0.0f) * _unpack_bf16_64[2];
                wepi_p7_1[3] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_65[3] : 0.0f) * _unpack_bf16_64[3];
                wepi_p7_1[4] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_65[4] : 0.0f) * _unpack_bf16_64[4];
                wepi_p7_1[5] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_65[5] : 0.0f) * _unpack_bf16_64[5];
                wepi_p7_1[6] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_65[6] : 0.0f) * _unpack_bf16_64[6];
                wepi_p7_1[7] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_65[7] : 0.0f) * _unpack_bf16_64[7];
                unsigned int wepi_p7_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p7_1[_lp*2 + 0], wepi_p7_1[_lp*2+1 + 0]));
                    wepi_p7_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_66[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_66[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p7_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_66[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p7_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_67[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_67[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[60 + _pair]) << 16);
                    (&_unpack_bf16_67[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[60 + _pair]) & 0xffff0000u);
                }
                float wepi_out7_1[8];
                wepi_out7_1[0] = _unpack_bf16_67[0] + _unpack_bf16_66[0];
                wepi_out7_1[1] = _unpack_bf16_67[1] + _unpack_bf16_66[1];
                wepi_out7_1[2] = _unpack_bf16_67[2] + _unpack_bf16_66[2];
                wepi_out7_1[3] = _unpack_bf16_67[3] + _unpack_bf16_66[3];
                wepi_out7_1[4] = _unpack_bf16_67[4] + _unpack_bf16_66[4];
                wepi_out7_1[5] = _unpack_bf16_67[5] + _unpack_bf16_66[5];
                wepi_out7_1[6] = _unpack_bf16_67[6] + _unpack_bf16_66[6];
                wepi_out7_1[7] = _unpack_bf16_67[7] + _unpack_bf16_66[7];
                unsigned int wepi_out7_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out7_1[_lp*2 + 0], wepi_out7_1[_lp*2+1 + 0]));
                    wepi_out7_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 60) * 5376 + wgcol) / 2)), "r"((wepi_out7_1_bf16 + 0)[(0) + 0]), "r"((wepi_out7_1_bf16 + 0)[(0) + 1]), "r"((wepi_out7_1_bf16 + 0)[(0) + 2]), "r"((wepi_out7_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 60 < M)) : "memory");
            }
        }
    }

    // Cleanup
}

}  // namespace h3_out_proj_gemm_fp8_warp_epilogue_sm120a
#undef CAKE_INF
#undef GROUP_M
#undef LAUNCH_MIN_BLOCKS
#undef NUM_AB_PIPE_STAGES
#undef SMEM_A_STAGE_OFF
#undef SMEM_A_STAGE_STAGE_BYTES
#undef SMEM_A_STAGE_STRIDE
#undef SMEM_B_STAGE_OFF
#undef SMEM_B_STAGE_STAGE_BYTES
#undef SMEM_B_STAGE_STRIDE
#undef SMEM_SFA_SLOT_OFF
#undef SMEM_SFA_SLOT_STAGE_BYTES
#undef SMEM_SFA_SLOT_STRIDE
#undef SMEM_SFB_STAGE_OFF
#undef SMEM_SFB_STAGE_STAGE_BYTES
#undef SMEM_SFB_STAGE_STRIDE
#undef SMEM_STAGING_OFF
#undef SMEM_STAGING_STAGE_BYTES
#undef SMEM_STAGING_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef ab_empty_addr
#undef ab_full_addr

namespace h3_out_proj_gemm_nvfp4_warp_epilogue_sm120a {

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
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_AB_PIPE_STAGES 3
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 16384
#define SMEM_A_STAGE_STRIDE 16384
#define SMEM_B_STAGE_OFF 50176
#define SMEM_B_STAGE_STAGE_BYTES 8192
#define SMEM_B_STAGE_STRIDE 8192
#define SMEM_STAGING_OFF 74752
#define SMEM_STAGING_STAGE_BYTES 8192
#define SMEM_STAGING_STRIDE 8192
#define SMEM_SFA_SLOT_OFF 82944
#define SMEM_SFA_SLOT_STAGE_BYTES 4096
#define SMEM_SFA_SLOT_STRIDE 4096
#define SMEM_SFB_STAGE_OFF 91136
#define SMEM_SFB_STAGE_STAGE_BYTES 1024
#define SMEM_SFB_STAGE_STRIDE 1024
#define SMEM_TOTAL 94208
#define THREADS 384
#define GROUP_M 1
#define LAUNCH_MIN_BLOCKS 1


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





__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
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


__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_h3_out_proj_gemm_fused(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, unsigned int* __restrict__ attn_out_w, unsigned int* __restrict__ act_q_w, uint8_t* __restrict__ act_sf_w, float* __restrict__ act_scale, float* __restrict__ act_global_scale, float* __restrict__ w_scale, __nv_bfloat16* __restrict__ gate, int* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, unsigned int* __restrict__ out, unsigned int* __restrict__ flags, int M, int num_m_tiles, int total_tiles, float alpha)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_empty_addr (mbar_base + 24)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int B_stage_addr = smem + 50176;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int staging_addr = smem + 74752;
    unsigned int* SFA_slot = reinterpret_cast<unsigned int*>(smem_raw + 82944);
    const int SFA_slot_addr = smem + 82944;
    unsigned int* SFB_stage = reinterpret_cast<unsigned int*>(smem_raw + 91136);
    const int SFB_stage_addr = smem + 91136;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // ab_empty: 3 barriers, init_count=8
            mbarrier_init(smem + 24, 8);
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Warpgroup: 0 ----
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        // ---- Role: producer ----
        if (warp == 0) {
            { // producer_main
                unsigned int load_stage = 0;
                unsigned int _phase_ab_empty = 1;
                #pragma unroll 1
                for (int tile = bid; tile < total_tiles; tile += num_bids) {
                    int tile_m = tile / (GROUP_M * 42) * GROUP_M + (tile - tile / (GROUP_M * 42) * (GROUP_M * 42)) % ((num_m_tiles - tile / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 42) * GROUP_M : GROUP_M);
                    int tile_n = (tile - tile / (GROUP_M * 42) * (GROUP_M * 42)) / ((num_m_tiles - tile / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 42) * GROUP_M : GROUP_M);
                    int rows_left = M - tile_m * 256;
                    unsigned int rows_ready = ((rows_left < 256) ? rows_left : 256);
                    if (elect_sync()) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(flags) + (tile_m))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(rows_ready)) >= static_cast<unsigned int>(1));
                        }
                        asm volatile("fence.proxy.async;" ::: "memory");
                        #pragma unroll 1
                        for (int kp = 0; kp < 28; kp++) {
                            mbarrier_wait_hint(ab_empty_addr + (load_stage) * 8, _phase_ab_empty, 10000000);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 29696);
                            tma_2d_gmem2smem(A_stage_addr + load_stage * 16384, (&A), kp * 2 * 64, tile_m * 256, ab_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(B_stage_addr + load_stage * 8192, (&B), kp * 2 * 64, tile_n * 128, ab_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(SFB_stage_addr + load_stage * 1024, (&SFB), 0, tile_n * 224 + kp * 2 * 4, ab_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(SFA_slot_addr + (unsigned int)((kp * 2 >> 1 & 1) * 4096), (&SFA), (kp * 2 >> 1) * 16, tile_m * 256, ab_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_ab_empty ^= 1; }
                            mbarrier_wait_hint(ab_empty_addr + (load_stage) * 8, _phase_ab_empty, 10000000);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 25600);
                            tma_2d_gmem2smem(A_stage_addr + load_stage * 16384, (&A), (kp * 2 + 1) * 64, tile_m * 256, ab_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(B_stage_addr + load_stage * 8192, (&B), (kp * 2 + 1) * 64, tile_n * 128, ab_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(SFB_stage_addr + load_stage * 1024, (&SFB), 0, tile_n * 224 + (kp * 2 + 1) * 4, ab_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_ab_empty ^= 1; }
                        }
                    }
                }
            }
        // ---- Role: quant ----
        } else if (warp >= 1 && warp <= 3) {
            { // quant_main
                unsigned int qw[16];
                float qx[16];
                float qacc[4];
                unsigned int qwords[4];
                float global_scale = act_global_scale[0];
                int warp_id_in_role = warp - 1;
                int qrow0 = bid * 3 + warp_id_in_role;
                #pragma unroll 1
                for (int row = qrow0; row < M; row += num_bids * 3) {
                    qacc[0] = 0.0f;
                    #pragma unroll 2
                    for (int j = 0; j < 7; j++) {
                        {
                            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * j) * 16) + 0);
                            qw[0 + 0] = _uv4_0.x;
                            qw[0 + 1] = _uv4_0.y;
                            qw[0 + 2] = _uv4_0.z;
                            qw[0 + 3] = _uv4_0.w;
                        }
                        {
                            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * j) * 16 + 4) + 0);
                            qw[4 + 0] = _uv4_1.x;
                            qw[4 + 1] = _uv4_1.y;
                            qw[4 + 2] = _uv4_1.z;
                            qw[4 + 3] = _uv4_1.w;
                        }
                        {
                            uint4 _uv4_2 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * j) * 16 + 8) + 0);
                            qw[8 + 0] = _uv4_2.x;
                            qw[8 + 1] = _uv4_2.y;
                            qw[8 + 2] = _uv4_2.z;
                            qw[8 + 3] = _uv4_2.w;
                        }
                        {
                            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * j) * 16 + 12) + 0);
                            qw[12 + 0] = _uv4_3.x;
                            qw[12 + 1] = _uv4_3.y;
                            qw[12 + 2] = _uv4_3.z;
                            qw[12 + 3] = _uv4_3.w;
                        }
                        float _unpack_bf16_0[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) << 16);
                            (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) & 0xffff0000u);
                        }
                        qacc[0] = 0.0f;
                        qx[0] = _unpack_bf16_0[0];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[0]));
                        qx[1] = _unpack_bf16_0[1];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[1]));
                        qx[2] = _unpack_bf16_0[2];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[2]));
                        qx[3] = _unpack_bf16_0[3];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[3]));
                        qx[4] = _unpack_bf16_0[4];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[4]));
                        qx[5] = _unpack_bf16_0[5];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[5]));
                        qx[6] = _unpack_bf16_0[6];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[6]));
                        qx[7] = _unpack_bf16_0[7];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[7]));
                        qx[8] = _unpack_bf16_0[8];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[8]));
                        qx[9] = _unpack_bf16_0[9];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[9]));
                        qx[10] = _unpack_bf16_0[10];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[10]));
                        qx[11] = _unpack_bf16_0[11];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[11]));
                        qx[12] = _unpack_bf16_0[12];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[12]));
                        qx[13] = _unpack_bf16_0[13];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[13]));
                        qx[14] = _unpack_bf16_0[14];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[14]));
                        qx[15] = _unpack_bf16_0[15];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_0[15]));
                        qacc[1] = qacc[0] * 0.16666667f * global_scale;
                        uint16_t _cvt_satfinite_e4m3x2_f32_4;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_4) : "f"(0.0f), "f"(qacc[1]));
                        *reinterpret_cast<unsigned char*>(act_sf_w + (row * 448 + (lane + 32 * j) * 2)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_4 & 127));
                        uint16_t _cvt_satfinite_e4m3x2_f32_6;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_6) : "f"(0.0f), "f"(qacc[1]));
                        uint16_t _e4m3x2_decode_7 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_6 & 0xFFu);
                        uint32_t _f16x2_decode_7;
                        float _fp8_e4m3_decode_5;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_7) : "h"(_e4m3x2_decode_7));
                        uint16_t _f16_decode_7 = (uint16_t)_f16x2_decode_7;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_5) : "h"(_f16_decode_7));
                        qacc[2] = _fp8_e4m3_decode_5;
                        float _fdiv_8 = __fdiv_rn(global_scale, qacc[2]);
                        qacc[3] = ((qacc[2] != 0.0f) ? _fdiv_8 : 0.0f);
                        qx[0] = qx[0] * qacc[3];
                        qx[1] = qx[1] * qacc[3];
                        qx[2] = qx[2] * qacc[3];
                        qx[3] = qx[3] * qacc[3];
                        qx[4] = qx[4] * qacc[3];
                        qx[5] = qx[5] * qacc[3];
                        qx[6] = qx[6] * qacc[3];
                        qx[7] = qx[7] * qacc[3];
                        qx[8] = qx[8] * qacc[3];
                        qx[9] = qx[9] * qacc[3];
                        qx[10] = qx[10] * qacc[3];
                        qx[11] = qx[11] * qacc[3];
                        qx[12] = qx[12] * qacc[3];
                        qx[13] = qx[13] * qacc[3];
                        qx[14] = qx[14] * qacc[3];
                        qx[15] = qx[15] * qacc[3];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qwords[0]) : "f"(qx[0]), "f"(qx[1]), "f"(qx[2]), "f"(qx[3]), "f"(qx[4]), "f"(qx[5]), "f"(qx[6]), "f"(qx[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qwords[1]) : "f"(qx[8]), "f"(qx[9]), "f"(qx[10]), "f"(qx[11]), "f"(qx[12]), "f"(qx[13]), "f"(qx[14]), "f"(qx[15]));
                        float _unpack_bf16_1[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) << 16);
                            (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[8 + _pair]) & 0xffff0000u);
                        }
                        qacc[0] = 0.0f;
                        qx[0] = _unpack_bf16_1[0];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[0]));
                        qx[1] = _unpack_bf16_1[1];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[1]));
                        qx[2] = _unpack_bf16_1[2];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[2]));
                        qx[3] = _unpack_bf16_1[3];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[3]));
                        qx[4] = _unpack_bf16_1[4];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[4]));
                        qx[5] = _unpack_bf16_1[5];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[5]));
                        qx[6] = _unpack_bf16_1[6];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[6]));
                        qx[7] = _unpack_bf16_1[7];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[7]));
                        qx[8] = _unpack_bf16_1[8];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[8]));
                        qx[9] = _unpack_bf16_1[9];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[9]));
                        qx[10] = _unpack_bf16_1[10];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[10]));
                        qx[11] = _unpack_bf16_1[11];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[11]));
                        qx[12] = _unpack_bf16_1[12];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[12]));
                        qx[13] = _unpack_bf16_1[13];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[13]));
                        qx[14] = _unpack_bf16_1[14];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[14]));
                        qx[15] = _unpack_bf16_1[15];
                        qacc[0] = fmaxf(qacc[0], fabsf(_unpack_bf16_1[15]));
                        qacc[1] = qacc[0] * 0.16666667f * global_scale;
                        uint16_t _cvt_satfinite_e4m3x2_f32_9;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_9) : "f"(0.0f), "f"(qacc[1]));
                        *reinterpret_cast<unsigned char*>(act_sf_w + (row * 448 + (lane + 32 * j) * 2 + 1)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_9 & 127));
                        uint16_t _cvt_satfinite_e4m3x2_f32_11;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_11) : "f"(0.0f), "f"(qacc[1]));
                        uint16_t _e4m3x2_decode_12 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_11 & 0xFFu);
                        uint32_t _f16x2_decode_12;
                        float _fp8_e4m3_decode_10;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_12) : "h"(_e4m3x2_decode_12));
                        uint16_t _f16_decode_12 = (uint16_t)_f16x2_decode_12;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_10) : "h"(_f16_decode_12));
                        qacc[2] = _fp8_e4m3_decode_10;
                        float _fdiv_13 = __fdiv_rn(global_scale, qacc[2]);
                        qacc[3] = ((qacc[2] != 0.0f) ? _fdiv_13 : 0.0f);
                        qx[0] = qx[0] * qacc[3];
                        qx[1] = qx[1] * qacc[3];
                        qx[2] = qx[2] * qacc[3];
                        qx[3] = qx[3] * qacc[3];
                        qx[4] = qx[4] * qacc[3];
                        qx[5] = qx[5] * qacc[3];
                        qx[6] = qx[6] * qacc[3];
                        qx[7] = qx[7] * qacc[3];
                        qx[8] = qx[8] * qacc[3];
                        qx[9] = qx[9] * qacc[3];
                        qx[10] = qx[10] * qacc[3];
                        qx[11] = qx[11] * qacc[3];
                        qx[12] = qx[12] * qacc[3];
                        qx[13] = qx[13] * qacc[3];
                        qx[14] = qx[14] * qacc[3];
                        qx[15] = qx[15] * qacc[3];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qwords[2]) : "f"(qx[0]), "f"(qx[1]), "f"(qx[2]), "f"(qx[3]), "f"(qx[4]), "f"(qx[5]), "f"(qx[6]), "f"(qx[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qwords[3]) : "f"(qx[8]), "f"(qx[9]), "f"(qx[10]), "f"(qx[11]), "f"(qx[12]), "f"(qx[13]), "f"(qx[14]), "f"(qx[15]));
                        reinterpret_cast<int4*>(act_q_w + ((row * 3584 + (lane + 32 * j) * 16) / 4))[0] = reinterpret_cast<int4*>(qwords)[0];
                    }
                    __threadfence();
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(flags) + (row / 256))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // mma_main
            unsigned int mma_stage = 0;
            int warp_id_in_role_1 = warp - 4;
            int warp_m = warp_id_in_role_1 % 4;
            int warp_n = warp_id_in_role_1 / 4;
            int role_tid = warp_id_in_role_1 * 32 + lane;
            double alpha_den = (double)act_global_scale[0] * (double)w_scale[0];
            float alpha_dev = (float)((double)1.0f / alpha_den);
            float accum[128];
            unsigned int a_frag[16];
            unsigned int b_frag[16];
            unsigned int sfa_pair[16];
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / (GROUP_M * 42) * GROUP_M + (tile_1 - tile_1 / (GROUP_M * 42) * (GROUP_M * 42)) % ((num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M : GROUP_M);
                int tile_n_1 = (tile_1 - tile_1 / (GROUP_M * 42) * (GROUP_M * 42)) / ((num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M : GROUP_M);
                accum[0] = 0.0f;
                accum[1] = 0.0f;
                accum[2] = 0.0f;
                accum[3] = 0.0f;
                accum[4] = 0.0f;
                accum[5] = 0.0f;
                accum[6] = 0.0f;
                accum[7] = 0.0f;
                accum[8] = 0.0f;
                accum[9] = 0.0f;
                accum[10] = 0.0f;
                accum[11] = 0.0f;
                accum[12] = 0.0f;
                accum[13] = 0.0f;
                accum[14] = 0.0f;
                accum[15] = 0.0f;
                accum[16] = 0.0f;
                accum[17] = 0.0f;
                accum[18] = 0.0f;
                accum[19] = 0.0f;
                accum[20] = 0.0f;
                accum[21] = 0.0f;
                accum[22] = 0.0f;
                accum[23] = 0.0f;
                accum[24] = 0.0f;
                accum[25] = 0.0f;
                accum[26] = 0.0f;
                accum[27] = 0.0f;
                accum[28] = 0.0f;
                accum[29] = 0.0f;
                accum[30] = 0.0f;
                accum[31] = 0.0f;
                accum[32] = 0.0f;
                accum[33] = 0.0f;
                accum[34] = 0.0f;
                accum[35] = 0.0f;
                accum[36] = 0.0f;
                accum[37] = 0.0f;
                accum[38] = 0.0f;
                accum[39] = 0.0f;
                accum[40] = 0.0f;
                accum[41] = 0.0f;
                accum[42] = 0.0f;
                accum[43] = 0.0f;
                accum[44] = 0.0f;
                accum[45] = 0.0f;
                accum[46] = 0.0f;
                accum[47] = 0.0f;
                accum[48] = 0.0f;
                accum[49] = 0.0f;
                accum[50] = 0.0f;
                accum[51] = 0.0f;
                accum[52] = 0.0f;
                accum[53] = 0.0f;
                accum[54] = 0.0f;
                accum[55] = 0.0f;
                accum[56] = 0.0f;
                accum[57] = 0.0f;
                accum[58] = 0.0f;
                accum[59] = 0.0f;
                accum[60] = 0.0f;
                accum[61] = 0.0f;
                accum[62] = 0.0f;
                accum[63] = 0.0f;
                accum[64] = 0.0f;
                accum[65] = 0.0f;
                accum[66] = 0.0f;
                accum[67] = 0.0f;
                accum[68] = 0.0f;
                accum[69] = 0.0f;
                accum[70] = 0.0f;
                accum[71] = 0.0f;
                accum[72] = 0.0f;
                accum[73] = 0.0f;
                accum[74] = 0.0f;
                accum[75] = 0.0f;
                accum[76] = 0.0f;
                accum[77] = 0.0f;
                accum[78] = 0.0f;
                accum[79] = 0.0f;
                accum[80] = 0.0f;
                accum[81] = 0.0f;
                accum[82] = 0.0f;
                accum[83] = 0.0f;
                accum[84] = 0.0f;
                accum[85] = 0.0f;
                accum[86] = 0.0f;
                accum[87] = 0.0f;
                accum[88] = 0.0f;
                accum[89] = 0.0f;
                accum[90] = 0.0f;
                accum[91] = 0.0f;
                accum[92] = 0.0f;
                accum[93] = 0.0f;
                accum[94] = 0.0f;
                accum[95] = 0.0f;
                accum[96] = 0.0f;
                accum[97] = 0.0f;
                accum[98] = 0.0f;
                accum[99] = 0.0f;
                accum[100] = 0.0f;
                accum[101] = 0.0f;
                accum[102] = 0.0f;
                accum[103] = 0.0f;
                accum[104] = 0.0f;
                accum[105] = 0.0f;
                accum[106] = 0.0f;
                accum[107] = 0.0f;
                accum[108] = 0.0f;
                accum[109] = 0.0f;
                accum[110] = 0.0f;
                accum[111] = 0.0f;
                accum[112] = 0.0f;
                accum[113] = 0.0f;
                accum[114] = 0.0f;
                accum[115] = 0.0f;
                accum[116] = 0.0f;
                accum[117] = 0.0f;
                accum[118] = 0.0f;
                accum[119] = 0.0f;
                accum[120] = 0.0f;
                accum[121] = 0.0f;
                accum[122] = 0.0f;
                accum[123] = 0.0f;
                accum[124] = 0.0f;
                accum[125] = 0.0f;
                accum[126] = 0.0f;
                accum[127] = 0.0f;
                int col_base = tile_n_1 * 128 + warp_n * 64 + (lane & 3) * 2;
                float sw[16];
                sw[0] = alpha_dev;
                sw[1] = alpha_dev;
                sw[2] = alpha_dev;
                sw[3] = alpha_dev;
                sw[4] = alpha_dev;
                sw[5] = alpha_dev;
                sw[6] = alpha_dev;
                sw[7] = alpha_dev;
                sw[8] = alpha_dev;
                sw[9] = alpha_dev;
                sw[10] = alpha_dev;
                sw[11] = alpha_dev;
                sw[12] = alpha_dev;
                sw[13] = alpha_dev;
                sw[14] = alpha_dev;
                sw[15] = alpha_dev;
                float sa[8];
                int wepi_gidx[16];
                #pragma unroll 1
                for (int kp_1 = 0; kp_1 < 28; kp_1++) {
                    mbarrier_wait_hint(ab_full_addr + (mma_stage) * 8, _phase_ab_full, 10000000);
                    if (kp_1 == 24) {
                        int arow = tile_m_1 * 256 + warp_m * 64 + (lane >> 2);
                        sa[0] = 1.0f;
                        int arow_0 = tile_m_1 * 256 + warp_m * 64 + 8 + (lane >> 2);
                        sa[1] = 1.0f;
                        int arow_1 = tile_m_1 * 256 + warp_m * 64 + 16 + (lane >> 2);
                        sa[2] = 1.0f;
                        int arow_2 = tile_m_1 * 256 + warp_m * 64 + 16 + 8 + (lane >> 2);
                        sa[3] = 1.0f;
                        int arow_3 = tile_m_1 * 256 + warp_m * 64 + 32 + (lane >> 2);
                        sa[4] = 1.0f;
                        int arow_4 = tile_m_1 * 256 + warp_m * 64 + 32 + 8 + (lane >> 2);
                        sa[5] = 1.0f;
                        int arow_5 = tile_m_1 * 256 + warp_m * 64 + 48 + (lane >> 2);
                        sa[6] = 1.0f;
                        int arow_6 = tile_m_1 * 256 + warp_m * 64 + 48 + 8 + (lane >> 2);
                        sa[7] = 1.0f;
                        wepi_gidx[0] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) : 0)];
                        wepi_gidx[1] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 4 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 4 : 0)];
                        wepi_gidx[2] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 8 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 8 : 0)];
                        wepi_gidx[3] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 12 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 12 : 0)];
                        wepi_gidx[4] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 16 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 16 : 0)];
                        wepi_gidx[5] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 20 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 20 : 0)];
                        wepi_gidx[6] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 24 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 24 : 0)];
                        wepi_gidx[7] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 28 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 28 : 0)];
                        wepi_gidx[8] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 32 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 32 : 0)];
                        wepi_gidx[9] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 36 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 36 : 0)];
                        wepi_gidx[10] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 40 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 40 : 0)];
                        wepi_gidx[11] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 44 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 44 : 0)];
                        wepi_gidx[12] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 48 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 48 : 0)];
                        wepi_gidx[13] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 52 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 52 : 0)];
                        wepi_gidx[14] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 56 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 56 : 0)];
                        wepi_gidx[15] = gate_index[((tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 60 < M) ? tile_m_1 * 256 + warp_m * 64 + (lane >> 3) + 60 : 0)];
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    __int128_t _smem_raw_0;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_0) : "r"(SFA_slot_addr + (unsigned int)((kp_1 * 2 >> 1 & 1) * 4096) + (unsigned int)((warp_m * 64 + (lane & 1) * 8 + (lane >> 2)) * 16)));
                    sfa_pair[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[0];
                    sfa_pair[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[1];
                    sfa_pair[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[2];
                    sfa_pair[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[3];
                    __int128_t _smem_raw_1;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_1) : "r"(SFA_slot_addr + (unsigned int)((kp_1 * 2 >> 1 & 1) * 4096) + (unsigned int)((warp_m * 64 + 16 + (lane & 1) * 8 + (lane >> 2)) * 16)));
                    sfa_pair[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[0];
                    sfa_pair[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[1];
                    sfa_pair[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[2];
                    sfa_pair[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[3];
                    __int128_t _smem_raw_2;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_2) : "r"(SFA_slot_addr + (unsigned int)((kp_1 * 2 >> 1 & 1) * 4096) + (unsigned int)((warp_m * 64 + 32 + (lane & 1) * 8 + (lane >> 2)) * 16)));
                    sfa_pair[8] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[0];
                    sfa_pair[9] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[1];
                    sfa_pair[10] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[2];
                    sfa_pair[11] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[3];
                    __int128_t _smem_raw_3;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_3) : "r"(SFA_slot_addr + (unsigned int)((kp_1 * 2 >> 1 & 1) * 4096) + (unsigned int)((warp_m * 64 + 48 + (lane & 1) * 8 + (lane >> 2)) * 16)));
                    sfa_pair[12] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[0];
                    sfa_pair[13] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[1];
                    sfa_pair[14] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[2];
                    sfa_pair[15] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[3];
                    unsigned int smem_regs[1];
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++) {
                        smem_regs[_lr] = SFB_stage[mma_stage * 256 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr];
                    }
                    unsigned int smem_regs_0[1];
                    #pragma unroll
                    for (int _lr_1 = 0; _lr_1 < 1; _lr_1++) {
                        smem_regs_0[_lr_1] = SFB_stage[mma_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_1];
                    }
                    unsigned int smem_regs_1[1];
                    #pragma unroll
                    for (int _lr_2 = 0; _lr_2 < 1; _lr_2++) {
                        smem_regs_1[_lr_2] = SFB_stage[mma_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_2];
                    }
                    unsigned int smem_regs_2[1];
                    #pragma unroll
                    for (int _lr_3 = 0; _lr_3 < 1; _lr_3++) {
                        smem_regs_2[_lr_3] = SFB_stage[mma_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_3];
                    }
                    unsigned int smem_regs_3[1];
                    #pragma unroll
                    for (int _lr_4 = 0; _lr_4 < 1; _lr_4++) {
                        smem_regs_3[_lr_4] = SFB_stage[mma_stage * 256 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_4];
                    }
                    unsigned int smem_regs_4[1];
                    #pragma unroll
                    for (int _lr_5 = 0; _lr_5 < 1; _lr_5++) {
                        smem_regs_4[_lr_5] = SFB_stage[mma_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_5];
                    }
                    unsigned int smem_regs_5[1];
                    #pragma unroll
                    for (int _lr_6 = 0; _lr_6 < 1; _lr_6++) {
                        smem_regs_5[_lr_6] = SFB_stage[mma_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_6];
                    }
                    unsigned int smem_regs_6[1];
                    #pragma unroll
                    for (int _lr_7 = 0; _lr_7 < 1; _lr_7++) {
                        smem_regs_6[_lr_7] = SFB_stage[mma_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_7];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[4])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[4])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[4])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[4])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[4])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[4])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[4])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[4])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[8])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[8])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[8])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[8])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[8])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[8])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[8])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[8])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[12])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[12])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[12])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[12])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[12])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[12])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[12])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[12])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_7[1];
                    #pragma unroll
                    for (int _lr_8 = 0; _lr_8 < 1; _lr_8++) {
                        smem_regs_7[_lr_8] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_8];
                    }
                    unsigned int smem_regs_8[1];
                    #pragma unroll
                    for (int _lr_9 = 0; _lr_9 < 1; _lr_9++) {
                        smem_regs_8[_lr_9] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_9];
                    }
                    unsigned int smem_regs_9[1];
                    #pragma unroll
                    for (int _lr_10 = 0; _lr_10 < 1; _lr_10++) {
                        smem_regs_9[_lr_10] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_10];
                    }
                    unsigned int smem_regs_10[1];
                    #pragma unroll
                    for (int _lr_11 = 0; _lr_11 < 1; _lr_11++) {
                        smem_regs_10[_lr_11] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_11];
                    }
                    unsigned int smem_regs_11[1];
                    #pragma unroll
                    for (int _lr_12 = 0; _lr_12 < 1; _lr_12++) {
                        smem_regs_11[_lr_12] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_12];
                    }
                    unsigned int smem_regs_12[1];
                    #pragma unroll
                    for (int _lr_13 = 0; _lr_13 < 1; _lr_13++) {
                        smem_regs_12[_lr_13] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_13];
                    }
                    unsigned int smem_regs_13[1];
                    #pragma unroll
                    for (int _lr_14 = 0; _lr_14 < 1; _lr_14++) {
                        smem_regs_13[_lr_14] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_14];
                    }
                    unsigned int smem_regs_14[1];
                    #pragma unroll
                    for (int _lr_15 = 0; _lr_15 < 1; _lr_15++) {
                        smem_regs_14[_lr_15] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_15];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[5])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[5])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[5])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[5])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[5])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[5])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[5])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[5])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[9])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[9])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[9])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[9])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[9])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[9])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[9])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[9])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[13])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[13])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[13])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[13])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[13])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[13])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[13])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[13])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_ab_full ^= 1; }
                    mbarrier_wait_hint(ab_full_addr + (mma_stage) * 8, _phase_ab_full, 10000000);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_15[1];
                    #pragma unroll
                    for (int _lr_16 = 0; _lr_16 < 1; _lr_16++) {
                        smem_regs_15[_lr_16] = SFB_stage[mma_stage * 256 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_16];
                    }
                    unsigned int smem_regs_16[1];
                    #pragma unroll
                    for (int _lr_17 = 0; _lr_17 < 1; _lr_17++) {
                        smem_regs_16[_lr_17] = SFB_stage[mma_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_17];
                    }
                    unsigned int smem_regs_17[1];
                    #pragma unroll
                    for (int _lr_18 = 0; _lr_18 < 1; _lr_18++) {
                        smem_regs_17[_lr_18] = SFB_stage[mma_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_18];
                    }
                    unsigned int smem_regs_18[1];
                    #pragma unroll
                    for (int _lr_19 = 0; _lr_19 < 1; _lr_19++) {
                        smem_regs_18[_lr_19] = SFB_stage[mma_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_19];
                    }
                    unsigned int smem_regs_19[1];
                    #pragma unroll
                    for (int _lr_20 = 0; _lr_20 < 1; _lr_20++) {
                        smem_regs_19[_lr_20] = SFB_stage[mma_stage * 256 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_20];
                    }
                    unsigned int smem_regs_20[1];
                    #pragma unroll
                    for (int _lr_21 = 0; _lr_21 < 1; _lr_21++) {
                        smem_regs_20[_lr_21] = SFB_stage[mma_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_21];
                    }
                    unsigned int smem_regs_21[1];
                    #pragma unroll
                    for (int _lr_22 = 0; _lr_22 < 1; _lr_22++) {
                        smem_regs_21[_lr_22] = SFB_stage[mma_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_22];
                    }
                    unsigned int smem_regs_22[1];
                    #pragma unroll
                    for (int _lr_23 = 0; _lr_23 < 1; _lr_23++) {
                        smem_regs_22[_lr_23] = SFB_stage[mma_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_23];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[2])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[2])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[2])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[2])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[2])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[2])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[2])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[2])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[6])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[6])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[6])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[6])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[6])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[6])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[6])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[6])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[10])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[10])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[10])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[10])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[10])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[10])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[10])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[10])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[14])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[14])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[14])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[14])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[14])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[14])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[14])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[14])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_23[1];
                    #pragma unroll
                    for (int _lr_24 = 0; _lr_24 < 1; _lr_24++) {
                        smem_regs_23[_lr_24] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_24];
                    }
                    unsigned int smem_regs_24[1];
                    #pragma unroll
                    for (int _lr_25 = 0; _lr_25 < 1; _lr_25++) {
                        smem_regs_24[_lr_25] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_25];
                    }
                    unsigned int smem_regs_25[1];
                    #pragma unroll
                    for (int _lr_26 = 0; _lr_26 < 1; _lr_26++) {
                        smem_regs_25[_lr_26] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_26];
                    }
                    unsigned int smem_regs_26[1];
                    #pragma unroll
                    for (int _lr_27 = 0; _lr_27 < 1; _lr_27++) {
                        smem_regs_26[_lr_27] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2) + (unsigned int)_lr_27];
                    }
                    unsigned int smem_regs_27[1];
                    #pragma unroll
                    for (int _lr_28 = 0; _lr_28 < 1; _lr_28++) {
                        smem_regs_27[_lr_28] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_28];
                    }
                    unsigned int smem_regs_28[1];
                    #pragma unroll
                    for (int _lr_29 = 0; _lr_29 < 1; _lr_29++) {
                        smem_regs_28[_lr_29] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_29];
                    }
                    unsigned int smem_regs_29[1];
                    #pragma unroll
                    for (int _lr_30 = 0; _lr_30 < 1; _lr_30++) {
                        smem_regs_29[_lr_30] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_30];
                    }
                    unsigned int smem_regs_30[1];
                    #pragma unroll
                    for (int _lr_31 = 0; _lr_31 < 1; _lr_31++) {
                        smem_regs_30[_lr_31] = SFB_stage[mma_stage * 256 + 128 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1) + (unsigned int)_lr_31];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[3])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_23[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[3])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_24[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[3])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_25[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[3])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_26[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[3])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[3])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[3])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[3])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[7])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_23[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[7])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_24[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[7])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_25[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[7])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_26[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[7])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[7])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[7])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[7])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[11])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_23[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[11])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_24[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[11])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_25[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[11])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_26[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[11])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[11])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[11])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[11])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(sfa_pair[15])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_23[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(sfa_pair[15])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_24[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(sfa_pair[15])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_25[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(sfa_pair[15])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_26[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(sfa_pair[15])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(sfa_pair[15])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(sfa_pair[15])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(sfa_pair[15])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_ab_full ^= 1; }
                }
                if (tile_m_1 > 2147483647) {
                    float chk = 0.0f;
                    chk = chk + accum[0];
                    chk = chk + accum[1];
                    chk = chk + accum[2];
                    chk = chk + accum[3];
                    chk = chk + accum[4];
                    chk = chk + accum[5];
                    chk = chk + accum[6];
                    chk = chk + accum[7];
                    chk = chk + accum[8];
                    chk = chk + accum[9];
                    chk = chk + accum[10];
                    chk = chk + accum[11];
                    chk = chk + accum[12];
                    chk = chk + accum[13];
                    chk = chk + accum[14];
                    chk = chk + accum[15];
                    chk = chk + accum[16];
                    chk = chk + accum[17];
                    chk = chk + accum[18];
                    chk = chk + accum[19];
                    chk = chk + accum[20];
                    chk = chk + accum[21];
                    chk = chk + accum[22];
                    chk = chk + accum[23];
                    chk = chk + accum[24];
                    chk = chk + accum[25];
                    chk = chk + accum[26];
                    chk = chk + accum[27];
                    chk = chk + accum[28];
                    chk = chk + accum[29];
                    chk = chk + accum[30];
                    chk = chk + accum[31];
                    chk = chk + accum[32];
                    chk = chk + accum[33];
                    chk = chk + accum[34];
                    chk = chk + accum[35];
                    chk = chk + accum[36];
                    chk = chk + accum[37];
                    chk = chk + accum[38];
                    chk = chk + accum[39];
                    chk = chk + accum[40];
                    chk = chk + accum[41];
                    chk = chk + accum[42];
                    chk = chk + accum[43];
                    chk = chk + accum[44];
                    chk = chk + accum[45];
                    chk = chk + accum[46];
                    chk = chk + accum[47];
                    chk = chk + accum[48];
                    chk = chk + accum[49];
                    chk = chk + accum[50];
                    chk = chk + accum[51];
                    chk = chk + accum[52];
                    chk = chk + accum[53];
                    chk = chk + accum[54];
                    chk = chk + accum[55];
                    chk = chk + accum[56];
                    chk = chk + accum[57];
                    chk = chk + accum[58];
                    chk = chk + accum[59];
                    chk = chk + accum[60];
                    chk = chk + accum[61];
                    chk = chk + accum[62];
                    chk = chk + accum[63];
                    chk = chk + accum[64];
                    chk = chk + accum[65];
                    chk = chk + accum[66];
                    chk = chk + accum[67];
                    chk = chk + accum[68];
                    chk = chk + accum[69];
                    chk = chk + accum[70];
                    chk = chk + accum[71];
                    chk = chk + accum[72];
                    chk = chk + accum[73];
                    chk = chk + accum[74];
                    chk = chk + accum[75];
                    chk = chk + accum[76];
                    chk = chk + accum[77];
                    chk = chk + accum[78];
                    chk = chk + accum[79];
                    chk = chk + accum[80];
                    chk = chk + accum[81];
                    chk = chk + accum[82];
                    chk = chk + accum[83];
                    chk = chk + accum[84];
                    chk = chk + accum[85];
                    chk = chk + accum[86];
                    chk = chk + accum[87];
                    chk = chk + accum[88];
                    chk = chk + accum[89];
                    chk = chk + accum[90];
                    chk = chk + accum[91];
                    chk = chk + accum[92];
                    chk = chk + accum[93];
                    chk = chk + accum[94];
                    chk = chk + accum[95];
                    chk = chk + accum[96];
                    chk = chk + accum[97];
                    chk = chk + accum[98];
                    chk = chk + accum[99];
                    chk = chk + accum[100];
                    chk = chk + accum[101];
                    chk = chk + accum[102];
                    chk = chk + accum[103];
                    chk = chk + accum[104];
                    chk = chk + accum[105];
                    chk = chk + accum[106];
                    chk = chk + accum[107];
                    chk = chk + accum[108];
                    chk = chk + accum[109];
                    chk = chk + accum[110];
                    chk = chk + accum[111];
                    chk = chk + accum[112];
                    chk = chk + accum[113];
                    chk = chk + accum[114];
                    chk = chk + accum[115];
                    chk = chk + accum[116];
                    chk = chk + accum[117];
                    chk = chk + accum[118];
                    chk = chk + accum[119];
                    chk = chk + accum[120];
                    chk = chk + accum[121];
                    chk = chk + accum[122];
                    chk = chk + accum[123];
                    chk = chk + accum[124];
                    chk = chk + accum[125];
                    chk = chk + accum[126];
                    chk = chk + accum[127];
                    if (lane == 0) {
                        out[tile_1] = __as_u32(chk);
                    }
                }
                int wsmat = lane >> 3;
                int wrow_base = tile_m_1 * 256 + warp_m * 64 + (lane >> 3);
                int wsrow_st = warp_m * 8 + (lane & 7);
                int wcol = warp_n * 64 + (lane & 7) * 8;
                int wgcol = tile_n_1 * 128 + wcol;
                int wsrow0 = warp_m * 8 + wsmat;
                int wsrow1 = warp_m * 8 + 4 + wsmat;
                unsigned int wrw[64];
                unsigned int wgw[64];
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? wepi_gidx[0] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_4 = reinterpret_cast<uint4*>(&wgw[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_4[_blk] = _vptr_4[_blk];
                    }
                }
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(residual + (((wrow_base < M) ? wrow_base : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_5 = reinterpret_cast<uint4*>(&wrw[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_5[_blk] = _vptr_5[_blk];
                    }
                }
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? wepi_gidx[1] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_6 = reinterpret_cast<uint4*>(&wgw[4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_6[_blk] = _vptr_6[_blk];
                    }
                }
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 4 < M) ? wrow_base + 4 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_7 = reinterpret_cast<uint4*>(&wrw[4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_7[_blk] = _vptr_7[_blk];
                    }
                }
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? wepi_gidx[2] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_8 = reinterpret_cast<uint4*>(&wgw[8]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_8[_blk] = _vptr_8[_blk];
                    }
                }
                {
                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 8 < M) ? wrow_base + 8 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_9 = reinterpret_cast<uint4*>(&wrw[8]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_9[_blk] = _vptr_9[_blk];
                    }
                }
                {
                    const uint4* _vptr_10 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? wepi_gidx[3] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_10 = reinterpret_cast<uint4*>(&wgw[12]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_10[_blk] = _vptr_10[_blk];
                    }
                }
                {
                    const uint4* _vptr_11 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 12 < M) ? wrow_base + 12 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_11 = reinterpret_cast<uint4*>(&wrw[12]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_11[_blk] = _vptr_11[_blk];
                    }
                }
                float wscaled[16];
                wscaled[0] = accum[0] * sa[0] * sw[0];
                wscaled[1] = accum[1] * sa[0] * sw[1];
                wscaled[2] = accum[4] * sa[0] * sw[2];
                wscaled[3] = accum[5] * sa[0] * sw[3];
                wscaled[4] = accum[8] * sa[0] * sw[4];
                wscaled[5] = accum[9] * sa[0] * sw[5];
                wscaled[6] = accum[12] * sa[0] * sw[6];
                wscaled[7] = accum[13] * sa[0] * sw[7];
                wscaled[8] = accum[16] * sa[0] * sw[8];
                wscaled[9] = accum[17] * sa[0] * sw[9];
                wscaled[10] = accum[20] * sa[0] * sw[10];
                wscaled[11] = accum[21] * sa[0] * sw[11];
                wscaled[12] = accum[24] * sa[0] * sw[12];
                wscaled[13] = accum[25] * sa[0] * sw[13];
                wscaled[14] = accum[28] * sa[0] * sw[14];
                wscaled[15] = accum[29] * sa[0] * sw[15];
                unsigned int wscaled_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled[_lp*2 + 0], wscaled[_lp*2+1 + 0]));
                    wscaled_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                {
                    const uint4* _vptr_14 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? wepi_gidx[4] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_14 = reinterpret_cast<uint4*>(&wgw[16]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_14[_blk] = _vptr_14[_blk];
                    }
                }
                {
                    const uint4* _vptr_15 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 16 < M) ? wrow_base + 16 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_15 = reinterpret_cast<uint4*>(&wrw[16]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_15[_blk] = _vptr_15[_blk];
                    }
                }
                {
                    const uint4* _vptr_16 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? wepi_gidx[5] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_16 = reinterpret_cast<uint4*>(&wgw[20]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_16[_blk] = _vptr_16[_blk];
                    }
                }
                {
                    const uint4* _vptr_17 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 20 < M) ? wrow_base + 20 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_17 = reinterpret_cast<uint4*>(&wrw[20]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_17[_blk] = _vptr_17[_blk];
                    }
                }
                float _unpack_bf16_2[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words[_pair]) << 16);
                    (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_3[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[_pair]) << 16);
                    (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[_pair]) & 0xffff0000u);
                }
                float wepi_p0_0[8];
                wepi_p0_0[0] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_3[0] : 0.0f) * _unpack_bf16_2[0];
                wepi_p0_0[1] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_3[1] : 0.0f) * _unpack_bf16_2[1];
                wepi_p0_0[2] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_3[2] : 0.0f) * _unpack_bf16_2[2];
                wepi_p0_0[3] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_3[3] : 0.0f) * _unpack_bf16_2[3];
                wepi_p0_0[4] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_3[4] : 0.0f) * _unpack_bf16_2[4];
                wepi_p0_0[5] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_3[5] : 0.0f) * _unpack_bf16_2[5];
                wepi_p0_0[6] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_3[6] : 0.0f) * _unpack_bf16_2[6];
                wepi_p0_0[7] = ((wepi_gidx[0] >= 0 && wepi_gidx[0] < 9) ? _unpack_bf16_3[7] : 0.0f) * _unpack_bf16_2[7];
                unsigned int wepi_p0_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p0_0[_lp*2 + 0], wepi_p0_0[_lp*2+1 + 0]));
                    wepi_p0_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_4[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p0_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p0_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_5[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[_pair]) << 16);
                    (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[_pair]) & 0xffff0000u);
                }
                float wepi_out0_0[8];
                wepi_out0_0[0] = _unpack_bf16_5[0] + _unpack_bf16_4[0];
                wepi_out0_0[1] = _unpack_bf16_5[1] + _unpack_bf16_4[1];
                wepi_out0_0[2] = _unpack_bf16_5[2] + _unpack_bf16_4[2];
                wepi_out0_0[3] = _unpack_bf16_5[3] + _unpack_bf16_4[3];
                wepi_out0_0[4] = _unpack_bf16_5[4] + _unpack_bf16_4[4];
                wepi_out0_0[5] = _unpack_bf16_5[5] + _unpack_bf16_4[5];
                wepi_out0_0[6] = _unpack_bf16_5[6] + _unpack_bf16_4[6];
                wepi_out0_0[7] = _unpack_bf16_5[7] + _unpack_bf16_4[7];
                unsigned int wepi_out0_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out0_0[_lp*2 + 0], wepi_out0_0[_lp*2+1 + 0]));
                    wepi_out0_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + ((wrow_base * 5376 + wgcol) / 2)), "r"((wepi_out0_0_bf16 + 0)[(0) + 0]), "r"((wepi_out0_0_bf16 + 0)[(0) + 1]), "r"((wepi_out0_0_bf16 + 0)[(0) + 2]), "r"((wepi_out0_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base < M)) : "memory");
                float _unpack_bf16_6[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words[4 + _pair]) << 16);
                    (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_7[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[4 + _pair]) << 16);
                    (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[4 + _pair]) & 0xffff0000u);
                }
                float wepi_p0_1[8];
                wepi_p0_1[0] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_7[0] : 0.0f) * _unpack_bf16_6[0];
                wepi_p0_1[1] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_7[1] : 0.0f) * _unpack_bf16_6[1];
                wepi_p0_1[2] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_7[2] : 0.0f) * _unpack_bf16_6[2];
                wepi_p0_1[3] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_7[3] : 0.0f) * _unpack_bf16_6[3];
                wepi_p0_1[4] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_7[4] : 0.0f) * _unpack_bf16_6[4];
                wepi_p0_1[5] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_7[5] : 0.0f) * _unpack_bf16_6[5];
                wepi_p0_1[6] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_7[6] : 0.0f) * _unpack_bf16_6[6];
                wepi_p0_1[7] = ((wepi_gidx[1] >= 0 && wepi_gidx[1] < 9) ? _unpack_bf16_7[7] : 0.0f) * _unpack_bf16_6[7];
                unsigned int wepi_p0_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p0_1[_lp*2 + 0], wepi_p0_1[_lp*2+1 + 0]));
                    wepi_p0_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_8[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p0_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p0_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_9[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[4 + _pair]) << 16);
                    (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[4 + _pair]) & 0xffff0000u);
                }
                float wepi_out0_1[8];
                wepi_out0_1[0] = _unpack_bf16_9[0] + _unpack_bf16_8[0];
                wepi_out0_1[1] = _unpack_bf16_9[1] + _unpack_bf16_8[1];
                wepi_out0_1[2] = _unpack_bf16_9[2] + _unpack_bf16_8[2];
                wepi_out0_1[3] = _unpack_bf16_9[3] + _unpack_bf16_8[3];
                wepi_out0_1[4] = _unpack_bf16_9[4] + _unpack_bf16_8[4];
                wepi_out0_1[5] = _unpack_bf16_9[5] + _unpack_bf16_8[5];
                wepi_out0_1[6] = _unpack_bf16_9[6] + _unpack_bf16_8[6];
                wepi_out0_1[7] = _unpack_bf16_9[7] + _unpack_bf16_8[7];
                unsigned int wepi_out0_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out0_1[_lp*2 + 0], wepi_out0_1[_lp*2+1 + 0]));
                    wepi_out0_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 4) * 5376 + wgcol) / 2)), "r"((wepi_out0_1_bf16 + 0)[(0) + 0]), "r"((wepi_out0_1_bf16 + 0)[(0) + 1]), "r"((wepi_out0_1_bf16 + 0)[(0) + 2]), "r"((wepi_out0_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 4 < M)) : "memory");
                float wscaled_0[16];
                wscaled_0[0] = accum[2] * sa[1] * sw[0];
                wscaled_0[1] = accum[3] * sa[1] * sw[1];
                wscaled_0[2] = accum[6] * sa[1] * sw[2];
                wscaled_0[3] = accum[7] * sa[1] * sw[3];
                wscaled_0[4] = accum[10] * sa[1] * sw[4];
                wscaled_0[5] = accum[11] * sa[1] * sw[5];
                wscaled_0[6] = accum[14] * sa[1] * sw[6];
                wscaled_0[7] = accum[15] * sa[1] * sw[7];
                wscaled_0[8] = accum[18] * sa[1] * sw[8];
                wscaled_0[9] = accum[19] * sa[1] * sw[9];
                wscaled_0[10] = accum[22] * sa[1] * sw[10];
                wscaled_0[11] = accum[23] * sa[1] * sw[11];
                wscaled_0[12] = accum[26] * sa[1] * sw[12];
                wscaled_0[13] = accum[27] * sa[1] * sw[13];
                wscaled_0[14] = accum[30] * sa[1] * sw[14];
                wscaled_0[15] = accum[31] * sa[1] * sw[15];
                unsigned int wscaled_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled_0[_lp*2 + 0], wscaled_0[_lp*2+1 + 0]));
                    wscaled_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_18 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_0_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words_1[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_1[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                {
                    const uint4* _vptr_20 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? wepi_gidx[6] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_20 = reinterpret_cast<uint4*>(&wgw[24]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_20[_blk] = _vptr_20[_blk];
                    }
                }
                {
                    const uint4* _vptr_21 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 24 < M) ? wrow_base + 24 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_21 = reinterpret_cast<uint4*>(&wrw[24]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_21[_blk] = _vptr_21[_blk];
                    }
                }
                {
                    const uint4* _vptr_22 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? wepi_gidx[7] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_22 = reinterpret_cast<uint4*>(&wgw[28]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_22[_blk] = _vptr_22[_blk];
                    }
                }
                {
                    const uint4* _vptr_23 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 28 < M) ? wrow_base + 28 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_23 = reinterpret_cast<uint4*>(&wrw[28]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_23[_blk] = _vptr_23[_blk];
                    }
                }
                float _unpack_bf16_10[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_1[_pair]) << 16);
                    (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_1[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_11[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[8 + _pair]) << 16);
                    (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[8 + _pair]) & 0xffff0000u);
                }
                float wepi_p1_0[8];
                wepi_p1_0[0] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_11[0] : 0.0f) * _unpack_bf16_10[0];
                wepi_p1_0[1] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_11[1] : 0.0f) * _unpack_bf16_10[1];
                wepi_p1_0[2] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_11[2] : 0.0f) * _unpack_bf16_10[2];
                wepi_p1_0[3] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_11[3] : 0.0f) * _unpack_bf16_10[3];
                wepi_p1_0[4] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_11[4] : 0.0f) * _unpack_bf16_10[4];
                wepi_p1_0[5] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_11[5] : 0.0f) * _unpack_bf16_10[5];
                wepi_p1_0[6] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_11[6] : 0.0f) * _unpack_bf16_10[6];
                wepi_p1_0[7] = ((wepi_gidx[2] >= 0 && wepi_gidx[2] < 9) ? _unpack_bf16_11[7] : 0.0f) * _unpack_bf16_10[7];
                unsigned int wepi_p1_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p1_0[_lp*2 + 0], wepi_p1_0[_lp*2+1 + 0]));
                    wepi_p1_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_12[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p1_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p1_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_13[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[8 + _pair]) << 16);
                    (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[8 + _pair]) & 0xffff0000u);
                }
                float wepi_out1_0[8];
                wepi_out1_0[0] = _unpack_bf16_13[0] + _unpack_bf16_12[0];
                wepi_out1_0[1] = _unpack_bf16_13[1] + _unpack_bf16_12[1];
                wepi_out1_0[2] = _unpack_bf16_13[2] + _unpack_bf16_12[2];
                wepi_out1_0[3] = _unpack_bf16_13[3] + _unpack_bf16_12[3];
                wepi_out1_0[4] = _unpack_bf16_13[4] + _unpack_bf16_12[4];
                wepi_out1_0[5] = _unpack_bf16_13[5] + _unpack_bf16_12[5];
                wepi_out1_0[6] = _unpack_bf16_13[6] + _unpack_bf16_12[6];
                wepi_out1_0[7] = _unpack_bf16_13[7] + _unpack_bf16_12[7];
                unsigned int wepi_out1_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out1_0[_lp*2 + 0], wepi_out1_0[_lp*2+1 + 0]));
                    wepi_out1_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 8) * 5376 + wgcol) / 2)), "r"((wepi_out1_0_bf16 + 0)[(0) + 0]), "r"((wepi_out1_0_bf16 + 0)[(0) + 1]), "r"((wepi_out1_0_bf16 + 0)[(0) + 2]), "r"((wepi_out1_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 8 < M)) : "memory");
                float _unpack_bf16_14[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_1[4 + _pair]) << 16);
                    (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_1[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_15[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[12 + _pair]) << 16);
                    (&_unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[12 + _pair]) & 0xffff0000u);
                }
                float wepi_p1_1[8];
                wepi_p1_1[0] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_15[0] : 0.0f) * _unpack_bf16_14[0];
                wepi_p1_1[1] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_15[1] : 0.0f) * _unpack_bf16_14[1];
                wepi_p1_1[2] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_15[2] : 0.0f) * _unpack_bf16_14[2];
                wepi_p1_1[3] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_15[3] : 0.0f) * _unpack_bf16_14[3];
                wepi_p1_1[4] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_15[4] : 0.0f) * _unpack_bf16_14[4];
                wepi_p1_1[5] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_15[5] : 0.0f) * _unpack_bf16_14[5];
                wepi_p1_1[6] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_15[6] : 0.0f) * _unpack_bf16_14[6];
                wepi_p1_1[7] = ((wepi_gidx[3] >= 0 && wepi_gidx[3] < 9) ? _unpack_bf16_15[7] : 0.0f) * _unpack_bf16_14[7];
                unsigned int wepi_p1_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p1_1[_lp*2 + 0], wepi_p1_1[_lp*2+1 + 0]));
                    wepi_p1_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_16[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p1_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p1_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_17[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[12 + _pair]) << 16);
                    (&_unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[12 + _pair]) & 0xffff0000u);
                }
                float wepi_out1_1[8];
                wepi_out1_1[0] = _unpack_bf16_17[0] + _unpack_bf16_16[0];
                wepi_out1_1[1] = _unpack_bf16_17[1] + _unpack_bf16_16[1];
                wepi_out1_1[2] = _unpack_bf16_17[2] + _unpack_bf16_16[2];
                wepi_out1_1[3] = _unpack_bf16_17[3] + _unpack_bf16_16[3];
                wepi_out1_1[4] = _unpack_bf16_17[4] + _unpack_bf16_16[4];
                wepi_out1_1[5] = _unpack_bf16_17[5] + _unpack_bf16_16[5];
                wepi_out1_1[6] = _unpack_bf16_17[6] + _unpack_bf16_16[6];
                wepi_out1_1[7] = _unpack_bf16_17[7] + _unpack_bf16_16[7];
                unsigned int wepi_out1_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out1_1[_lp*2 + 0], wepi_out1_1[_lp*2+1 + 0]));
                    wepi_out1_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 12) * 5376 + wgcol) / 2)), "r"((wepi_out1_1_bf16 + 0)[(0) + 0]), "r"((wepi_out1_1_bf16 + 0)[(0) + 1]), "r"((wepi_out1_1_bf16 + 0)[(0) + 2]), "r"((wepi_out1_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 12 < M)) : "memory");
                float wscaled_2[16];
                wscaled_2[0] = accum[32] * sa[2] * sw[0];
                wscaled_2[1] = accum[33] * sa[2] * sw[1];
                wscaled_2[2] = accum[36] * sa[2] * sw[2];
                wscaled_2[3] = accum[37] * sa[2] * sw[3];
                wscaled_2[4] = accum[40] * sa[2] * sw[4];
                wscaled_2[5] = accum[41] * sa[2] * sw[5];
                wscaled_2[6] = accum[44] * sa[2] * sw[6];
                wscaled_2[7] = accum[45] * sa[2] * sw[7];
                wscaled_2[8] = accum[48] * sa[2] * sw[8];
                wscaled_2[9] = accum[49] * sa[2] * sw[9];
                wscaled_2[10] = accum[52] * sa[2] * sw[10];
                wscaled_2[11] = accum[53] * sa[2] * sw[11];
                wscaled_2[12] = accum[56] * sa[2] * sw[12];
                wscaled_2[13] = accum[57] * sa[2] * sw[13];
                wscaled_2[14] = accum[60] * sa[2] * sw[14];
                wscaled_2[15] = accum[61] * sa[2] * sw[15];
                unsigned int wscaled_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled_2[_lp*2 + 0], wscaled_2[_lp*2+1 + 0]));
                    wscaled_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_24 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_24), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_25 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_25), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_2_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words_3[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_3[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                {
                    const uint4* _vptr_26 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? wepi_gidx[8] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_26 = reinterpret_cast<uint4*>(&wgw[32]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_26[_blk] = _vptr_26[_blk];
                    }
                }
                {
                    const uint4* _vptr_27 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 32 < M) ? wrow_base + 32 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_27 = reinterpret_cast<uint4*>(&wrw[32]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_27[_blk] = _vptr_27[_blk];
                    }
                }
                {
                    const uint4* _vptr_28 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? wepi_gidx[9] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_28 = reinterpret_cast<uint4*>(&wgw[36]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_28[_blk] = _vptr_28[_blk];
                    }
                }
                {
                    const uint4* _vptr_29 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 36 < M) ? wrow_base + 36 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_29 = reinterpret_cast<uint4*>(&wrw[36]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_29[_blk] = _vptr_29[_blk];
                    }
                }
                float _unpack_bf16_18[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_3[_pair]) << 16);
                    (&_unpack_bf16_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_3[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_19[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[16 + _pair]) << 16);
                    (&_unpack_bf16_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[16 + _pair]) & 0xffff0000u);
                }
                float wepi_p2_0[8];
                wepi_p2_0[0] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_19[0] : 0.0f) * _unpack_bf16_18[0];
                wepi_p2_0[1] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_19[1] : 0.0f) * _unpack_bf16_18[1];
                wepi_p2_0[2] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_19[2] : 0.0f) * _unpack_bf16_18[2];
                wepi_p2_0[3] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_19[3] : 0.0f) * _unpack_bf16_18[3];
                wepi_p2_0[4] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_19[4] : 0.0f) * _unpack_bf16_18[4];
                wepi_p2_0[5] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_19[5] : 0.0f) * _unpack_bf16_18[5];
                wepi_p2_0[6] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_19[6] : 0.0f) * _unpack_bf16_18[6];
                wepi_p2_0[7] = ((wepi_gidx[4] >= 0 && wepi_gidx[4] < 9) ? _unpack_bf16_19[7] : 0.0f) * _unpack_bf16_18[7];
                unsigned int wepi_p2_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p2_0[_lp*2 + 0], wepi_p2_0[_lp*2+1 + 0]));
                    wepi_p2_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_20[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p2_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p2_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_21[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_21[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[16 + _pair]) << 16);
                    (&_unpack_bf16_21[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[16 + _pair]) & 0xffff0000u);
                }
                float wepi_out2_0[8];
                wepi_out2_0[0] = _unpack_bf16_21[0] + _unpack_bf16_20[0];
                wepi_out2_0[1] = _unpack_bf16_21[1] + _unpack_bf16_20[1];
                wepi_out2_0[2] = _unpack_bf16_21[2] + _unpack_bf16_20[2];
                wepi_out2_0[3] = _unpack_bf16_21[3] + _unpack_bf16_20[3];
                wepi_out2_0[4] = _unpack_bf16_21[4] + _unpack_bf16_20[4];
                wepi_out2_0[5] = _unpack_bf16_21[5] + _unpack_bf16_20[5];
                wepi_out2_0[6] = _unpack_bf16_21[6] + _unpack_bf16_20[6];
                wepi_out2_0[7] = _unpack_bf16_21[7] + _unpack_bf16_20[7];
                unsigned int wepi_out2_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out2_0[_lp*2 + 0], wepi_out2_0[_lp*2+1 + 0]));
                    wepi_out2_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 16) * 5376 + wgcol) / 2)), "r"((wepi_out2_0_bf16 + 0)[(0) + 0]), "r"((wepi_out2_0_bf16 + 0)[(0) + 1]), "r"((wepi_out2_0_bf16 + 0)[(0) + 2]), "r"((wepi_out2_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 16 < M)) : "memory");
                float _unpack_bf16_22[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_22[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_3[4 + _pair]) << 16);
                    (&_unpack_bf16_22[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_3[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_23[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_23[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[20 + _pair]) << 16);
                    (&_unpack_bf16_23[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[20 + _pair]) & 0xffff0000u);
                }
                float wepi_p2_1[8];
                wepi_p2_1[0] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_23[0] : 0.0f) * _unpack_bf16_22[0];
                wepi_p2_1[1] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_23[1] : 0.0f) * _unpack_bf16_22[1];
                wepi_p2_1[2] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_23[2] : 0.0f) * _unpack_bf16_22[2];
                wepi_p2_1[3] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_23[3] : 0.0f) * _unpack_bf16_22[3];
                wepi_p2_1[4] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_23[4] : 0.0f) * _unpack_bf16_22[4];
                wepi_p2_1[5] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_23[5] : 0.0f) * _unpack_bf16_22[5];
                wepi_p2_1[6] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_23[6] : 0.0f) * _unpack_bf16_22[6];
                wepi_p2_1[7] = ((wepi_gidx[5] >= 0 && wepi_gidx[5] < 9) ? _unpack_bf16_23[7] : 0.0f) * _unpack_bf16_22[7];
                unsigned int wepi_p2_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p2_1[_lp*2 + 0], wepi_p2_1[_lp*2+1 + 0]));
                    wepi_p2_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_24[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_24[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p2_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_24[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p2_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_25[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_25[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[20 + _pair]) << 16);
                    (&_unpack_bf16_25[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[20 + _pair]) & 0xffff0000u);
                }
                float wepi_out2_1[8];
                wepi_out2_1[0] = _unpack_bf16_25[0] + _unpack_bf16_24[0];
                wepi_out2_1[1] = _unpack_bf16_25[1] + _unpack_bf16_24[1];
                wepi_out2_1[2] = _unpack_bf16_25[2] + _unpack_bf16_24[2];
                wepi_out2_1[3] = _unpack_bf16_25[3] + _unpack_bf16_24[3];
                wepi_out2_1[4] = _unpack_bf16_25[4] + _unpack_bf16_24[4];
                wepi_out2_1[5] = _unpack_bf16_25[5] + _unpack_bf16_24[5];
                wepi_out2_1[6] = _unpack_bf16_25[6] + _unpack_bf16_24[6];
                wepi_out2_1[7] = _unpack_bf16_25[7] + _unpack_bf16_24[7];
                unsigned int wepi_out2_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out2_1[_lp*2 + 0], wepi_out2_1[_lp*2+1 + 0]));
                    wepi_out2_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 20) * 5376 + wgcol) / 2)), "r"((wepi_out2_1_bf16 + 0)[(0) + 0]), "r"((wepi_out2_1_bf16 + 0)[(0) + 1]), "r"((wepi_out2_1_bf16 + 0)[(0) + 2]), "r"((wepi_out2_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 20 < M)) : "memory");
                float wscaled_4[16];
                wscaled_4[0] = accum[34] * sa[3] * sw[0];
                wscaled_4[1] = accum[35] * sa[3] * sw[1];
                wscaled_4[2] = accum[38] * sa[3] * sw[2];
                wscaled_4[3] = accum[39] * sa[3] * sw[3];
                wscaled_4[4] = accum[42] * sa[3] * sw[4];
                wscaled_4[5] = accum[43] * sa[3] * sw[5];
                wscaled_4[6] = accum[46] * sa[3] * sw[6];
                wscaled_4[7] = accum[47] * sa[3] * sw[7];
                wscaled_4[8] = accum[50] * sa[3] * sw[8];
                wscaled_4[9] = accum[51] * sa[3] * sw[9];
                wscaled_4[10] = accum[54] * sa[3] * sw[10];
                wscaled_4[11] = accum[55] * sa[3] * sw[11];
                wscaled_4[12] = accum[58] * sa[3] * sw[12];
                wscaled_4[13] = accum[59] * sa[3] * sw[13];
                wscaled_4[14] = accum[62] * sa[3] * sw[14];
                wscaled_4[15] = accum[63] * sa[3] * sw[15];
                unsigned int wscaled_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled_4[_lp*2 + 0], wscaled_4[_lp*2+1 + 0]));
                    wscaled_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_30 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_30), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_31 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_31), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_4_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words_5[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_5[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                {
                    const uint4* _vptr_32 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? wepi_gidx[10] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_32 = reinterpret_cast<uint4*>(&wgw[40]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_32[_blk] = _vptr_32[_blk];
                    }
                }
                {
                    const uint4* _vptr_33 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 40 < M) ? wrow_base + 40 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_33 = reinterpret_cast<uint4*>(&wrw[40]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_33[_blk] = _vptr_33[_blk];
                    }
                }
                {
                    const uint4* _vptr_34 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? wepi_gidx[11] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_34 = reinterpret_cast<uint4*>(&wgw[44]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_34[_blk] = _vptr_34[_blk];
                    }
                }
                {
                    const uint4* _vptr_35 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 44 < M) ? wrow_base + 44 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_35 = reinterpret_cast<uint4*>(&wrw[44]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_35[_blk] = _vptr_35[_blk];
                    }
                }
                float _unpack_bf16_26[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_26[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_5[_pair]) << 16);
                    (&_unpack_bf16_26[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_5[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_27[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_27[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[24 + _pair]) << 16);
                    (&_unpack_bf16_27[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[24 + _pair]) & 0xffff0000u);
                }
                float wepi_p3_0[8];
                wepi_p3_0[0] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_27[0] : 0.0f) * _unpack_bf16_26[0];
                wepi_p3_0[1] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_27[1] : 0.0f) * _unpack_bf16_26[1];
                wepi_p3_0[2] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_27[2] : 0.0f) * _unpack_bf16_26[2];
                wepi_p3_0[3] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_27[3] : 0.0f) * _unpack_bf16_26[3];
                wepi_p3_0[4] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_27[4] : 0.0f) * _unpack_bf16_26[4];
                wepi_p3_0[5] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_27[5] : 0.0f) * _unpack_bf16_26[5];
                wepi_p3_0[6] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_27[6] : 0.0f) * _unpack_bf16_26[6];
                wepi_p3_0[7] = ((wepi_gidx[6] >= 0 && wepi_gidx[6] < 9) ? _unpack_bf16_27[7] : 0.0f) * _unpack_bf16_26[7];
                unsigned int wepi_p3_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p3_0[_lp*2 + 0], wepi_p3_0[_lp*2+1 + 0]));
                    wepi_p3_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_28[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_28[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p3_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_28[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p3_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_29[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_29[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[24 + _pair]) << 16);
                    (&_unpack_bf16_29[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[24 + _pair]) & 0xffff0000u);
                }
                float wepi_out3_0[8];
                wepi_out3_0[0] = _unpack_bf16_29[0] + _unpack_bf16_28[0];
                wepi_out3_0[1] = _unpack_bf16_29[1] + _unpack_bf16_28[1];
                wepi_out3_0[2] = _unpack_bf16_29[2] + _unpack_bf16_28[2];
                wepi_out3_0[3] = _unpack_bf16_29[3] + _unpack_bf16_28[3];
                wepi_out3_0[4] = _unpack_bf16_29[4] + _unpack_bf16_28[4];
                wepi_out3_0[5] = _unpack_bf16_29[5] + _unpack_bf16_28[5];
                wepi_out3_0[6] = _unpack_bf16_29[6] + _unpack_bf16_28[6];
                wepi_out3_0[7] = _unpack_bf16_29[7] + _unpack_bf16_28[7];
                unsigned int wepi_out3_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out3_0[_lp*2 + 0], wepi_out3_0[_lp*2+1 + 0]));
                    wepi_out3_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 24) * 5376 + wgcol) / 2)), "r"((wepi_out3_0_bf16 + 0)[(0) + 0]), "r"((wepi_out3_0_bf16 + 0)[(0) + 1]), "r"((wepi_out3_0_bf16 + 0)[(0) + 2]), "r"((wepi_out3_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 24 < M)) : "memory");
                float _unpack_bf16_30[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_30[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_5[4 + _pair]) << 16);
                    (&_unpack_bf16_30[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_5[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_31[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_31[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[28 + _pair]) << 16);
                    (&_unpack_bf16_31[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[28 + _pair]) & 0xffff0000u);
                }
                float wepi_p3_1[8];
                wepi_p3_1[0] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_31[0] : 0.0f) * _unpack_bf16_30[0];
                wepi_p3_1[1] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_31[1] : 0.0f) * _unpack_bf16_30[1];
                wepi_p3_1[2] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_31[2] : 0.0f) * _unpack_bf16_30[2];
                wepi_p3_1[3] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_31[3] : 0.0f) * _unpack_bf16_30[3];
                wepi_p3_1[4] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_31[4] : 0.0f) * _unpack_bf16_30[4];
                wepi_p3_1[5] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_31[5] : 0.0f) * _unpack_bf16_30[5];
                wepi_p3_1[6] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_31[6] : 0.0f) * _unpack_bf16_30[6];
                wepi_p3_1[7] = ((wepi_gidx[7] >= 0 && wepi_gidx[7] < 9) ? _unpack_bf16_31[7] : 0.0f) * _unpack_bf16_30[7];
                unsigned int wepi_p3_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p3_1[_lp*2 + 0], wepi_p3_1[_lp*2+1 + 0]));
                    wepi_p3_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p3_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p3_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_33[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_33[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[28 + _pair]) << 16);
                    (&_unpack_bf16_33[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[28 + _pair]) & 0xffff0000u);
                }
                float wepi_out3_1[8];
                wepi_out3_1[0] = _unpack_bf16_33[0] + _unpack_bf16_32[0];
                wepi_out3_1[1] = _unpack_bf16_33[1] + _unpack_bf16_32[1];
                wepi_out3_1[2] = _unpack_bf16_33[2] + _unpack_bf16_32[2];
                wepi_out3_1[3] = _unpack_bf16_33[3] + _unpack_bf16_32[3];
                wepi_out3_1[4] = _unpack_bf16_33[4] + _unpack_bf16_32[4];
                wepi_out3_1[5] = _unpack_bf16_33[5] + _unpack_bf16_32[5];
                wepi_out3_1[6] = _unpack_bf16_33[6] + _unpack_bf16_32[6];
                wepi_out3_1[7] = _unpack_bf16_33[7] + _unpack_bf16_32[7];
                unsigned int wepi_out3_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out3_1[_lp*2 + 0], wepi_out3_1[_lp*2+1 + 0]));
                    wepi_out3_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 28) * 5376 + wgcol) / 2)), "r"((wepi_out3_1_bf16 + 0)[(0) + 0]), "r"((wepi_out3_1_bf16 + 0)[(0) + 1]), "r"((wepi_out3_1_bf16 + 0)[(0) + 2]), "r"((wepi_out3_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 28 < M)) : "memory");
                float wscaled_6[16];
                wscaled_6[0] = accum[64] * sa[4] * sw[0];
                wscaled_6[1] = accum[65] * sa[4] * sw[1];
                wscaled_6[2] = accum[68] * sa[4] * sw[2];
                wscaled_6[3] = accum[69] * sa[4] * sw[3];
                wscaled_6[4] = accum[72] * sa[4] * sw[4];
                wscaled_6[5] = accum[73] * sa[4] * sw[5];
                wscaled_6[6] = accum[76] * sa[4] * sw[6];
                wscaled_6[7] = accum[77] * sa[4] * sw[7];
                wscaled_6[8] = accum[80] * sa[4] * sw[8];
                wscaled_6[9] = accum[81] * sa[4] * sw[9];
                wscaled_6[10] = accum[84] * sa[4] * sw[10];
                wscaled_6[11] = accum[85] * sa[4] * sw[11];
                wscaled_6[12] = accum[88] * sa[4] * sw[12];
                wscaled_6[13] = accum[89] * sa[4] * sw[13];
                wscaled_6[14] = accum[92] * sa[4] * sw[14];
                wscaled_6[15] = accum[93] * sa[4] * sw[15];
                unsigned int wscaled_6_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled_6[_lp*2 + 0], wscaled_6[_lp*2+1 + 0]));
                    wscaled_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_36 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_36), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_37 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_37), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_6_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words_7[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_7[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                {
                    const uint4* _vptr_38 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? wepi_gidx[12] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_38 = reinterpret_cast<uint4*>(&wgw[48]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_38[_blk] = _vptr_38[_blk];
                    }
                }
                {
                    const uint4* _vptr_39 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 48 < M) ? wrow_base + 48 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_39 = reinterpret_cast<uint4*>(&wrw[48]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_39[_blk] = _vptr_39[_blk];
                    }
                }
                {
                    const uint4* _vptr_40 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? wepi_gidx[13] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_40 = reinterpret_cast<uint4*>(&wgw[52]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_40[_blk] = _vptr_40[_blk];
                    }
                }
                {
                    const uint4* _vptr_41 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 52 < M) ? wrow_base + 52 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_41 = reinterpret_cast<uint4*>(&wrw[52]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_41[_blk] = _vptr_41[_blk];
                    }
                }
                float _unpack_bf16_34[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_34[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_7[_pair]) << 16);
                    (&_unpack_bf16_34[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_7[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_35[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_35[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[32 + _pair]) << 16);
                    (&_unpack_bf16_35[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[32 + _pair]) & 0xffff0000u);
                }
                float wepi_p4_0[8];
                wepi_p4_0[0] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_35[0] : 0.0f) * _unpack_bf16_34[0];
                wepi_p4_0[1] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_35[1] : 0.0f) * _unpack_bf16_34[1];
                wepi_p4_0[2] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_35[2] : 0.0f) * _unpack_bf16_34[2];
                wepi_p4_0[3] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_35[3] : 0.0f) * _unpack_bf16_34[3];
                wepi_p4_0[4] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_35[4] : 0.0f) * _unpack_bf16_34[4];
                wepi_p4_0[5] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_35[5] : 0.0f) * _unpack_bf16_34[5];
                wepi_p4_0[6] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_35[6] : 0.0f) * _unpack_bf16_34[6];
                wepi_p4_0[7] = ((wepi_gidx[8] >= 0 && wepi_gidx[8] < 9) ? _unpack_bf16_35[7] : 0.0f) * _unpack_bf16_34[7];
                unsigned int wepi_p4_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p4_0[_lp*2 + 0], wepi_p4_0[_lp*2+1 + 0]));
                    wepi_p4_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_36[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_36[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p4_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_36[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p4_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_37[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_37[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[32 + _pair]) << 16);
                    (&_unpack_bf16_37[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[32 + _pair]) & 0xffff0000u);
                }
                float wepi_out4_0[8];
                wepi_out4_0[0] = _unpack_bf16_37[0] + _unpack_bf16_36[0];
                wepi_out4_0[1] = _unpack_bf16_37[1] + _unpack_bf16_36[1];
                wepi_out4_0[2] = _unpack_bf16_37[2] + _unpack_bf16_36[2];
                wepi_out4_0[3] = _unpack_bf16_37[3] + _unpack_bf16_36[3];
                wepi_out4_0[4] = _unpack_bf16_37[4] + _unpack_bf16_36[4];
                wepi_out4_0[5] = _unpack_bf16_37[5] + _unpack_bf16_36[5];
                wepi_out4_0[6] = _unpack_bf16_37[6] + _unpack_bf16_36[6];
                wepi_out4_0[7] = _unpack_bf16_37[7] + _unpack_bf16_36[7];
                unsigned int wepi_out4_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out4_0[_lp*2 + 0], wepi_out4_0[_lp*2+1 + 0]));
                    wepi_out4_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 32) * 5376 + wgcol) / 2)), "r"((wepi_out4_0_bf16 + 0)[(0) + 0]), "r"((wepi_out4_0_bf16 + 0)[(0) + 1]), "r"((wepi_out4_0_bf16 + 0)[(0) + 2]), "r"((wepi_out4_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 32 < M)) : "memory");
                float _unpack_bf16_38[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_38[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_7[4 + _pair]) << 16);
                    (&_unpack_bf16_38[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_7[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_39[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_39[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[36 + _pair]) << 16);
                    (&_unpack_bf16_39[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[36 + _pair]) & 0xffff0000u);
                }
                float wepi_p4_1[8];
                wepi_p4_1[0] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_39[0] : 0.0f) * _unpack_bf16_38[0];
                wepi_p4_1[1] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_39[1] : 0.0f) * _unpack_bf16_38[1];
                wepi_p4_1[2] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_39[2] : 0.0f) * _unpack_bf16_38[2];
                wepi_p4_1[3] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_39[3] : 0.0f) * _unpack_bf16_38[3];
                wepi_p4_1[4] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_39[4] : 0.0f) * _unpack_bf16_38[4];
                wepi_p4_1[5] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_39[5] : 0.0f) * _unpack_bf16_38[5];
                wepi_p4_1[6] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_39[6] : 0.0f) * _unpack_bf16_38[6];
                wepi_p4_1[7] = ((wepi_gidx[9] >= 0 && wepi_gidx[9] < 9) ? _unpack_bf16_39[7] : 0.0f) * _unpack_bf16_38[7];
                unsigned int wepi_p4_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p4_1[_lp*2 + 0], wepi_p4_1[_lp*2+1 + 0]));
                    wepi_p4_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_40[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_40[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p4_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_40[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p4_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_41[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_41[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[36 + _pair]) << 16);
                    (&_unpack_bf16_41[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[36 + _pair]) & 0xffff0000u);
                }
                float wepi_out4_1[8];
                wepi_out4_1[0] = _unpack_bf16_41[0] + _unpack_bf16_40[0];
                wepi_out4_1[1] = _unpack_bf16_41[1] + _unpack_bf16_40[1];
                wepi_out4_1[2] = _unpack_bf16_41[2] + _unpack_bf16_40[2];
                wepi_out4_1[3] = _unpack_bf16_41[3] + _unpack_bf16_40[3];
                wepi_out4_1[4] = _unpack_bf16_41[4] + _unpack_bf16_40[4];
                wepi_out4_1[5] = _unpack_bf16_41[5] + _unpack_bf16_40[5];
                wepi_out4_1[6] = _unpack_bf16_41[6] + _unpack_bf16_40[6];
                wepi_out4_1[7] = _unpack_bf16_41[7] + _unpack_bf16_40[7];
                unsigned int wepi_out4_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out4_1[_lp*2 + 0], wepi_out4_1[_lp*2+1 + 0]));
                    wepi_out4_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 36) * 5376 + wgcol) / 2)), "r"((wepi_out4_1_bf16 + 0)[(0) + 0]), "r"((wepi_out4_1_bf16 + 0)[(0) + 1]), "r"((wepi_out4_1_bf16 + 0)[(0) + 2]), "r"((wepi_out4_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 36 < M)) : "memory");
                float wscaled_8[16];
                wscaled_8[0] = accum[66] * sa[5] * sw[0];
                wscaled_8[1] = accum[67] * sa[5] * sw[1];
                wscaled_8[2] = accum[70] * sa[5] * sw[2];
                wscaled_8[3] = accum[71] * sa[5] * sw[3];
                wscaled_8[4] = accum[74] * sa[5] * sw[4];
                wscaled_8[5] = accum[75] * sa[5] * sw[5];
                wscaled_8[6] = accum[78] * sa[5] * sw[6];
                wscaled_8[7] = accum[79] * sa[5] * sw[7];
                wscaled_8[8] = accum[82] * sa[5] * sw[8];
                wscaled_8[9] = accum[83] * sa[5] * sw[9];
                wscaled_8[10] = accum[86] * sa[5] * sw[10];
                wscaled_8[11] = accum[87] * sa[5] * sw[11];
                wscaled_8[12] = accum[90] * sa[5] * sw[12];
                wscaled_8[13] = accum[91] * sa[5] * sw[13];
                wscaled_8[14] = accum[94] * sa[5] * sw[14];
                wscaled_8[15] = accum[95] * sa[5] * sw[15];
                unsigned int wscaled_8_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled_8[_lp*2 + 0], wscaled_8[_lp*2+1 + 0]));
                    wscaled_8_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_42 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_42), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_43 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_43), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_8_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words_9[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_9[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                {
                    const uint4* _vptr_44 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? wepi_gidx[14] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_44 = reinterpret_cast<uint4*>(&wgw[56]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_44[_blk] = _vptr_44[_blk];
                    }
                }
                {
                    const uint4* _vptr_45 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 56 < M) ? wrow_base + 56 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_45 = reinterpret_cast<uint4*>(&wrw[56]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_45[_blk] = _vptr_45[_blk];
                    }
                }
                {
                    const uint4* _vptr_46 = reinterpret_cast<const uint4*>(gate + (((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? wepi_gidx[15] : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_46 = reinterpret_cast<uint4*>(&wgw[60]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_46[_blk] = _vptr_46[_blk];
                    }
                }
                {
                    const uint4* _vptr_47 = reinterpret_cast<const uint4*>(residual + (((wrow_base + 60 < M) ? wrow_base + 60 : 0) * 5376 + wgcol) + 0);
                    uint4* _vdst_47 = reinterpret_cast<uint4*>(&wrw[60]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_47[_blk] = _vptr_47[_blk];
                    }
                }
                float _unpack_bf16_42[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_42[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_9[_pair]) << 16);
                    (&_unpack_bf16_42[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_9[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_43[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_43[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[40 + _pair]) << 16);
                    (&_unpack_bf16_43[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[40 + _pair]) & 0xffff0000u);
                }
                float wepi_p5_0[8];
                wepi_p5_0[0] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_43[0] : 0.0f) * _unpack_bf16_42[0];
                wepi_p5_0[1] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_43[1] : 0.0f) * _unpack_bf16_42[1];
                wepi_p5_0[2] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_43[2] : 0.0f) * _unpack_bf16_42[2];
                wepi_p5_0[3] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_43[3] : 0.0f) * _unpack_bf16_42[3];
                wepi_p5_0[4] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_43[4] : 0.0f) * _unpack_bf16_42[4];
                wepi_p5_0[5] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_43[5] : 0.0f) * _unpack_bf16_42[5];
                wepi_p5_0[6] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_43[6] : 0.0f) * _unpack_bf16_42[6];
                wepi_p5_0[7] = ((wepi_gidx[10] >= 0 && wepi_gidx[10] < 9) ? _unpack_bf16_43[7] : 0.0f) * _unpack_bf16_42[7];
                unsigned int wepi_p5_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p5_0[_lp*2 + 0], wepi_p5_0[_lp*2+1 + 0]));
                    wepi_p5_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_44[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_44[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p5_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_44[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p5_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_45[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_45[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[40 + _pair]) << 16);
                    (&_unpack_bf16_45[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[40 + _pair]) & 0xffff0000u);
                }
                float wepi_out5_0[8];
                wepi_out5_0[0] = _unpack_bf16_45[0] + _unpack_bf16_44[0];
                wepi_out5_0[1] = _unpack_bf16_45[1] + _unpack_bf16_44[1];
                wepi_out5_0[2] = _unpack_bf16_45[2] + _unpack_bf16_44[2];
                wepi_out5_0[3] = _unpack_bf16_45[3] + _unpack_bf16_44[3];
                wepi_out5_0[4] = _unpack_bf16_45[4] + _unpack_bf16_44[4];
                wepi_out5_0[5] = _unpack_bf16_45[5] + _unpack_bf16_44[5];
                wepi_out5_0[6] = _unpack_bf16_45[6] + _unpack_bf16_44[6];
                wepi_out5_0[7] = _unpack_bf16_45[7] + _unpack_bf16_44[7];
                unsigned int wepi_out5_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out5_0[_lp*2 + 0], wepi_out5_0[_lp*2+1 + 0]));
                    wepi_out5_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 40) * 5376 + wgcol) / 2)), "r"((wepi_out5_0_bf16 + 0)[(0) + 0]), "r"((wepi_out5_0_bf16 + 0)[(0) + 1]), "r"((wepi_out5_0_bf16 + 0)[(0) + 2]), "r"((wepi_out5_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 40 < M)) : "memory");
                float _unpack_bf16_46[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_46[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_9[4 + _pair]) << 16);
                    (&_unpack_bf16_46[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_9[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_47[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_47[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[44 + _pair]) << 16);
                    (&_unpack_bf16_47[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[44 + _pair]) & 0xffff0000u);
                }
                float wepi_p5_1[8];
                wepi_p5_1[0] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_47[0] : 0.0f) * _unpack_bf16_46[0];
                wepi_p5_1[1] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_47[1] : 0.0f) * _unpack_bf16_46[1];
                wepi_p5_1[2] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_47[2] : 0.0f) * _unpack_bf16_46[2];
                wepi_p5_1[3] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_47[3] : 0.0f) * _unpack_bf16_46[3];
                wepi_p5_1[4] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_47[4] : 0.0f) * _unpack_bf16_46[4];
                wepi_p5_1[5] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_47[5] : 0.0f) * _unpack_bf16_46[5];
                wepi_p5_1[6] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_47[6] : 0.0f) * _unpack_bf16_46[6];
                wepi_p5_1[7] = ((wepi_gidx[11] >= 0 && wepi_gidx[11] < 9) ? _unpack_bf16_47[7] : 0.0f) * _unpack_bf16_46[7];
                unsigned int wepi_p5_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p5_1[_lp*2 + 0], wepi_p5_1[_lp*2+1 + 0]));
                    wepi_p5_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_48[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_48[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p5_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_48[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p5_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_49[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_49[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[44 + _pair]) << 16);
                    (&_unpack_bf16_49[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[44 + _pair]) & 0xffff0000u);
                }
                float wepi_out5_1[8];
                wepi_out5_1[0] = _unpack_bf16_49[0] + _unpack_bf16_48[0];
                wepi_out5_1[1] = _unpack_bf16_49[1] + _unpack_bf16_48[1];
                wepi_out5_1[2] = _unpack_bf16_49[2] + _unpack_bf16_48[2];
                wepi_out5_1[3] = _unpack_bf16_49[3] + _unpack_bf16_48[3];
                wepi_out5_1[4] = _unpack_bf16_49[4] + _unpack_bf16_48[4];
                wepi_out5_1[5] = _unpack_bf16_49[5] + _unpack_bf16_48[5];
                wepi_out5_1[6] = _unpack_bf16_49[6] + _unpack_bf16_48[6];
                wepi_out5_1[7] = _unpack_bf16_49[7] + _unpack_bf16_48[7];
                unsigned int wepi_out5_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out5_1[_lp*2 + 0], wepi_out5_1[_lp*2+1 + 0]));
                    wepi_out5_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 44) * 5376 + wgcol) / 2)), "r"((wepi_out5_1_bf16 + 0)[(0) + 0]), "r"((wepi_out5_1_bf16 + 0)[(0) + 1]), "r"((wepi_out5_1_bf16 + 0)[(0) + 2]), "r"((wepi_out5_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 44 < M)) : "memory");
                float wscaled_10[16];
                wscaled_10[0] = accum[96] * sa[6] * sw[0];
                wscaled_10[1] = accum[97] * sa[6] * sw[1];
                wscaled_10[2] = accum[100] * sa[6] * sw[2];
                wscaled_10[3] = accum[101] * sa[6] * sw[3];
                wscaled_10[4] = accum[104] * sa[6] * sw[4];
                wscaled_10[5] = accum[105] * sa[6] * sw[5];
                wscaled_10[6] = accum[108] * sa[6] * sw[6];
                wscaled_10[7] = accum[109] * sa[6] * sw[7];
                wscaled_10[8] = accum[112] * sa[6] * sw[8];
                wscaled_10[9] = accum[113] * sa[6] * sw[9];
                wscaled_10[10] = accum[116] * sa[6] * sw[10];
                wscaled_10[11] = accum[117] * sa[6] * sw[11];
                wscaled_10[12] = accum[120] * sa[6] * sw[12];
                wscaled_10[13] = accum[121] * sa[6] * sw[13];
                wscaled_10[14] = accum[124] * sa[6] * sw[14];
                wscaled_10[15] = accum[125] * sa[6] * sw[15];
                unsigned int wscaled_10_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled_10[_lp*2 + 0], wscaled_10[_lp*2+1 + 0]));
                    wscaled_10_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_48 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_48), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_49 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_49), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_10_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words_11[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_11[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                float _unpack_bf16_50[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_50[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_11[_pair]) << 16);
                    (&_unpack_bf16_50[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_11[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_51[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_51[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[48 + _pair]) << 16);
                    (&_unpack_bf16_51[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[48 + _pair]) & 0xffff0000u);
                }
                float wepi_p6_0[8];
                wepi_p6_0[0] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_51[0] : 0.0f) * _unpack_bf16_50[0];
                wepi_p6_0[1] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_51[1] : 0.0f) * _unpack_bf16_50[1];
                wepi_p6_0[2] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_51[2] : 0.0f) * _unpack_bf16_50[2];
                wepi_p6_0[3] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_51[3] : 0.0f) * _unpack_bf16_50[3];
                wepi_p6_0[4] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_51[4] : 0.0f) * _unpack_bf16_50[4];
                wepi_p6_0[5] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_51[5] : 0.0f) * _unpack_bf16_50[5];
                wepi_p6_0[6] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_51[6] : 0.0f) * _unpack_bf16_50[6];
                wepi_p6_0[7] = ((wepi_gidx[12] >= 0 && wepi_gidx[12] < 9) ? _unpack_bf16_51[7] : 0.0f) * _unpack_bf16_50[7];
                unsigned int wepi_p6_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p6_0[_lp*2 + 0], wepi_p6_0[_lp*2+1 + 0]));
                    wepi_p6_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_52[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_52[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p6_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_52[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p6_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_53[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_53[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[48 + _pair]) << 16);
                    (&_unpack_bf16_53[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[48 + _pair]) & 0xffff0000u);
                }
                float wepi_out6_0[8];
                wepi_out6_0[0] = _unpack_bf16_53[0] + _unpack_bf16_52[0];
                wepi_out6_0[1] = _unpack_bf16_53[1] + _unpack_bf16_52[1];
                wepi_out6_0[2] = _unpack_bf16_53[2] + _unpack_bf16_52[2];
                wepi_out6_0[3] = _unpack_bf16_53[3] + _unpack_bf16_52[3];
                wepi_out6_0[4] = _unpack_bf16_53[4] + _unpack_bf16_52[4];
                wepi_out6_0[5] = _unpack_bf16_53[5] + _unpack_bf16_52[5];
                wepi_out6_0[6] = _unpack_bf16_53[6] + _unpack_bf16_52[6];
                wepi_out6_0[7] = _unpack_bf16_53[7] + _unpack_bf16_52[7];
                unsigned int wepi_out6_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out6_0[_lp*2 + 0], wepi_out6_0[_lp*2+1 + 0]));
                    wepi_out6_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 48) * 5376 + wgcol) / 2)), "r"((wepi_out6_0_bf16 + 0)[(0) + 0]), "r"((wepi_out6_0_bf16 + 0)[(0) + 1]), "r"((wepi_out6_0_bf16 + 0)[(0) + 2]), "r"((wepi_out6_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 48 < M)) : "memory");
                float _unpack_bf16_54[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_54[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_11[4 + _pair]) << 16);
                    (&_unpack_bf16_54[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_11[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_55[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_55[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[52 + _pair]) << 16);
                    (&_unpack_bf16_55[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[52 + _pair]) & 0xffff0000u);
                }
                float wepi_p6_1[8];
                wepi_p6_1[0] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_55[0] : 0.0f) * _unpack_bf16_54[0];
                wepi_p6_1[1] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_55[1] : 0.0f) * _unpack_bf16_54[1];
                wepi_p6_1[2] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_55[2] : 0.0f) * _unpack_bf16_54[2];
                wepi_p6_1[3] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_55[3] : 0.0f) * _unpack_bf16_54[3];
                wepi_p6_1[4] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_55[4] : 0.0f) * _unpack_bf16_54[4];
                wepi_p6_1[5] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_55[5] : 0.0f) * _unpack_bf16_54[5];
                wepi_p6_1[6] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_55[6] : 0.0f) * _unpack_bf16_54[6];
                wepi_p6_1[7] = ((wepi_gidx[13] >= 0 && wepi_gidx[13] < 9) ? _unpack_bf16_55[7] : 0.0f) * _unpack_bf16_54[7];
                unsigned int wepi_p6_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p6_1[_lp*2 + 0], wepi_p6_1[_lp*2+1 + 0]));
                    wepi_p6_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_56[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_56[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p6_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_56[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p6_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_57[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_57[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[52 + _pair]) << 16);
                    (&_unpack_bf16_57[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[52 + _pair]) & 0xffff0000u);
                }
                float wepi_out6_1[8];
                wepi_out6_1[0] = _unpack_bf16_57[0] + _unpack_bf16_56[0];
                wepi_out6_1[1] = _unpack_bf16_57[1] + _unpack_bf16_56[1];
                wepi_out6_1[2] = _unpack_bf16_57[2] + _unpack_bf16_56[2];
                wepi_out6_1[3] = _unpack_bf16_57[3] + _unpack_bf16_56[3];
                wepi_out6_1[4] = _unpack_bf16_57[4] + _unpack_bf16_56[4];
                wepi_out6_1[5] = _unpack_bf16_57[5] + _unpack_bf16_56[5];
                wepi_out6_1[6] = _unpack_bf16_57[6] + _unpack_bf16_56[6];
                wepi_out6_1[7] = _unpack_bf16_57[7] + _unpack_bf16_56[7];
                unsigned int wepi_out6_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out6_1[_lp*2 + 0], wepi_out6_1[_lp*2+1 + 0]));
                    wepi_out6_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 52) * 5376 + wgcol) / 2)), "r"((wepi_out6_1_bf16 + 0)[(0) + 0]), "r"((wepi_out6_1_bf16 + 0)[(0) + 1]), "r"((wepi_out6_1_bf16 + 0)[(0) + 2]), "r"((wepi_out6_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 52 < M)) : "memory");
                float wscaled_12[16];
                wscaled_12[0] = accum[98] * sa[7] * sw[0];
                wscaled_12[1] = accum[99] * sa[7] * sw[1];
                wscaled_12[2] = accum[102] * sa[7] * sw[2];
                wscaled_12[3] = accum[103] * sa[7] * sw[3];
                wscaled_12[4] = accum[106] * sa[7] * sw[4];
                wscaled_12[5] = accum[107] * sa[7] * sw[5];
                wscaled_12[6] = accum[110] * sa[7] * sw[6];
                wscaled_12[7] = accum[111] * sa[7] * sw[7];
                wscaled_12[8] = accum[114] * sa[7] * sw[8];
                wscaled_12[9] = accum[115] * sa[7] * sw[9];
                wscaled_12[10] = accum[118] * sa[7] * sw[10];
                wscaled_12[11] = accum[119] * sa[7] * sw[11];
                wscaled_12[12] = accum[122] * sa[7] * sw[12];
                wscaled_12[13] = accum[123] * sa[7] * sw[13];
                wscaled_12[14] = accum[126] * sa[7] * sw[14];
                wscaled_12[15] = accum[127] * sa[7] * sw[15];
                unsigned int wscaled_12_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wscaled_12[_lp*2 + 0], wscaled_12[_lp*2+1 + 0]));
                    wscaled_12_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_50 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + wsmat * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_50), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_51 = static_cast<uint32_t>(staging_addr + (unsigned int)(wsrow_st * 256) + (unsigned int)((warp_n * 64 + (4 + wsmat) * 8) * 2 ^ (wsrow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_51), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wscaled_12_bf16[7]))
                    : "memory");
                __syncwarp();
                unsigned int wo_words_13[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow0 * 256) + (unsigned int)(wcol * 2 ^ (wsrow0 & 7) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wo_words_13[(4) + 3]))
                    : "r"(staging_addr + (unsigned int)(wsrow1 * 256) + (unsigned int)(wcol * 2 ^ (wsrow1 & 7) << 4)));
                __syncwarp();
                float _unpack_bf16_58[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_58[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_13[_pair]) << 16);
                    (&_unpack_bf16_58[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_13[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_59[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_59[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[56 + _pair]) << 16);
                    (&_unpack_bf16_59[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[56 + _pair]) & 0xffff0000u);
                }
                float wepi_p7_0[8];
                wepi_p7_0[0] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_59[0] : 0.0f) * _unpack_bf16_58[0];
                wepi_p7_0[1] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_59[1] : 0.0f) * _unpack_bf16_58[1];
                wepi_p7_0[2] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_59[2] : 0.0f) * _unpack_bf16_58[2];
                wepi_p7_0[3] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_59[3] : 0.0f) * _unpack_bf16_58[3];
                wepi_p7_0[4] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_59[4] : 0.0f) * _unpack_bf16_58[4];
                wepi_p7_0[5] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_59[5] : 0.0f) * _unpack_bf16_58[5];
                wepi_p7_0[6] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_59[6] : 0.0f) * _unpack_bf16_58[6];
                wepi_p7_0[7] = ((wepi_gidx[14] >= 0 && wepi_gidx[14] < 9) ? _unpack_bf16_59[7] : 0.0f) * _unpack_bf16_58[7];
                unsigned int wepi_p7_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p7_0[_lp*2 + 0], wepi_p7_0[_lp*2+1 + 0]));
                    wepi_p7_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_60[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_60[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p7_0_bf16[_pair]) << 16);
                    (&_unpack_bf16_60[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p7_0_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_61[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_61[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[56 + _pair]) << 16);
                    (&_unpack_bf16_61[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[56 + _pair]) & 0xffff0000u);
                }
                float wepi_out7_0[8];
                wepi_out7_0[0] = _unpack_bf16_61[0] + _unpack_bf16_60[0];
                wepi_out7_0[1] = _unpack_bf16_61[1] + _unpack_bf16_60[1];
                wepi_out7_0[2] = _unpack_bf16_61[2] + _unpack_bf16_60[2];
                wepi_out7_0[3] = _unpack_bf16_61[3] + _unpack_bf16_60[3];
                wepi_out7_0[4] = _unpack_bf16_61[4] + _unpack_bf16_60[4];
                wepi_out7_0[5] = _unpack_bf16_61[5] + _unpack_bf16_60[5];
                wepi_out7_0[6] = _unpack_bf16_61[6] + _unpack_bf16_60[6];
                wepi_out7_0[7] = _unpack_bf16_61[7] + _unpack_bf16_60[7];
                unsigned int wepi_out7_0_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out7_0[_lp*2 + 0], wepi_out7_0[_lp*2+1 + 0]));
                    wepi_out7_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 56) * 5376 + wgcol) / 2)), "r"((wepi_out7_0_bf16 + 0)[(0) + 0]), "r"((wepi_out7_0_bf16 + 0)[(0) + 1]), "r"((wepi_out7_0_bf16 + 0)[(0) + 2]), "r"((wepi_out7_0_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 56 < M)) : "memory");
                float _unpack_bf16_62[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_62[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wo_words_13[4 + _pair]) << 16);
                    (&_unpack_bf16_62[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wo_words_13[4 + _pair]) & 0xffff0000u);
                }
                float _unpack_bf16_63[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_63[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wgw[60 + _pair]) << 16);
                    (&_unpack_bf16_63[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wgw[60 + _pair]) & 0xffff0000u);
                }
                float wepi_p7_1[8];
                wepi_p7_1[0] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_63[0] : 0.0f) * _unpack_bf16_62[0];
                wepi_p7_1[1] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_63[1] : 0.0f) * _unpack_bf16_62[1];
                wepi_p7_1[2] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_63[2] : 0.0f) * _unpack_bf16_62[2];
                wepi_p7_1[3] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_63[3] : 0.0f) * _unpack_bf16_62[3];
                wepi_p7_1[4] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_63[4] : 0.0f) * _unpack_bf16_62[4];
                wepi_p7_1[5] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_63[5] : 0.0f) * _unpack_bf16_62[5];
                wepi_p7_1[6] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_63[6] : 0.0f) * _unpack_bf16_62[6];
                wepi_p7_1[7] = ((wepi_gidx[15] >= 0 && wepi_gidx[15] < 9) ? _unpack_bf16_63[7] : 0.0f) * _unpack_bf16_62[7];
                unsigned int wepi_p7_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_p7_1[_lp*2 + 0], wepi_p7_1[_lp*2+1 + 0]));
                    wepi_p7_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_64[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_64[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wepi_p7_1_bf16[_pair]) << 16);
                    (&_unpack_bf16_64[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wepi_p7_1_bf16[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_65[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_65[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(wrw[60 + _pair]) << 16);
                    (&_unpack_bf16_65[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(wrw[60 + _pair]) & 0xffff0000u);
                }
                float wepi_out7_1[8];
                wepi_out7_1[0] = _unpack_bf16_65[0] + _unpack_bf16_64[0];
                wepi_out7_1[1] = _unpack_bf16_65[1] + _unpack_bf16_64[1];
                wepi_out7_1[2] = _unpack_bf16_65[2] + _unpack_bf16_64[2];
                wepi_out7_1[3] = _unpack_bf16_65[3] + _unpack_bf16_64[3];
                wepi_out7_1[4] = _unpack_bf16_65[4] + _unpack_bf16_64[4];
                wepi_out7_1[5] = _unpack_bf16_65[5] + _unpack_bf16_64[5];
                wepi_out7_1[6] = _unpack_bf16_65[6] + _unpack_bf16_64[6];
                wepi_out7_1[7] = _unpack_bf16_65[7] + _unpack_bf16_64[7];
                unsigned int wepi_out7_1_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wepi_out7_1[_lp*2 + 0], wepi_out7_1[_lp*2+1 + 0]));
                    wepi_out7_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + (((wrow_base + 60) * 5376 + wgcol) / 2)), "r"((wepi_out7_1_bf16 + 0)[(0) + 0]), "r"((wepi_out7_1_bf16 + 0)[(0) + 1]), "r"((wepi_out7_1_bf16 + 0)[(0) + 2]), "r"((wepi_out7_1_bf16 + 0)[(0) + 3]), "r"((unsigned int)(wrow_base + 60 < M)) : "memory");
            }
        }
    }

    // Cleanup
}

}  // namespace h3_out_proj_gemm_nvfp4_warp_epilogue_sm120a
#undef CAKE_INF
#undef GROUP_M
#undef LAUNCH_MIN_BLOCKS
#undef NUM_AB_PIPE_STAGES
#undef SMEM_A_STAGE_OFF
#undef SMEM_A_STAGE_STAGE_BYTES
#undef SMEM_A_STAGE_STRIDE
#undef SMEM_B_STAGE_OFF
#undef SMEM_B_STAGE_STAGE_BYTES
#undef SMEM_B_STAGE_STRIDE
#undef SMEM_SFA_SLOT_OFF
#undef SMEM_SFA_SLOT_STAGE_BYTES
#undef SMEM_SFA_SLOT_STRIDE
#undef SMEM_SFB_STAGE_OFF
#undef SMEM_SFB_STAGE_STAGE_BYTES
#undef SMEM_SFB_STAGE_STRIDE
#undef SMEM_STAGING_OFF
#undef SMEM_STAGING_STAGE_BYTES
#undef SMEM_STAGING_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef ab_empty_addr
#undef ab_full_addr

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>

#include "tvm_ffi_utils.h"

#include "cake_minimax_h3_sm120_host.cuh"

namespace {

constexpr int64_t kHidden = 5376;   // output width N
constexpr int64_t kAttnDim = 7168;  // reduction K
constexpr int64_t kSfBlock = 16;
constexpr int64_t kAttnSf = kAttnDim / kSfBlock;  // 448 UE4M3 scales per activation row
// TMA coordinates are 32-bit; the persistent tile counter is int32.
constexpr int64_t kMaxRows = int64_t{1} << 24;
constexpr int kBlockM = 256;
constexpr int kBlockN = 128;
constexpr int kBlockK = 64;
constexpr int kNTiles = 42;
constexpr int kSfbRowsPerNTile = 224;
constexpr int kSfbRowsPerKTile = 4;
constexpr uint32_t kSfaBoxBytes = 16;  // one k256 pair of ring stages per activation-scale box
constexpr CUtensorMapSwizzle kOperandSwizzle = CU_TENSOR_MAP_SWIZZLE_64B;  // 64-byte operand rows per stage
constexpr int kGemmThreads = 384;

using GemmKernel = void (*)(CUtensorMap, CUtensorMap, CUtensorMap, CUtensorMap, unsigned int*, unsigned int*, uint8_t*,
                            float*, float*, float*, __nv_bfloat16*, int*, __nv_bfloat16*, unsigned int*, unsigned int*,
                            int, int, int, float);

using GemmVariant = minimax_h3_sm120::GemmVariant<GemmKernel>;

// [program][quant (0 = fp8, 1 = nvfp4)].  Program 1 stages the epilogue per MMA warp instead of per CTA, and its
// NVFP4 kernel reads each activation-scale row once per pair of ring stages (same arithmetic, identical output bits).
// It measured faster than program 0 on the 188-SM RTX PRO 6000 Blackwell at every tested M (33472 to 109952, both
// quants), and on the 170-SM RTX 5090 with NVFP4 only, so it runs on 188-SM devices and for NVFP4 on 170-SM devices.
const GemmVariant kGemmVariants[2][2] = {
    {{h3_out_proj_gemm_fp8_sm120a::kernel_h3_out_proj_gemm_fused, 94208},
     {h3_out_proj_gemm_nvfp4_sm120a::kernel_h3_out_proj_gemm_fused, 94208}},
    {{h3_out_proj_gemm_fp8_warp_epilogue_sm120a::kernel_h3_out_proj_gemm_fused, 94208},
     {h3_out_proj_gemm_nvfp4_warp_epilogue_sm120a::kernel_h3_out_proj_gemm_fused, 94208}},
};
constexpr int kWarpEpilogueSms = 188;       // both quants
constexpr int kWarpEpilogueNvfp4Sms = 170;  // NVFP4 (quant 1) only

// -1 routes by device and quant; 0 / 1 run that program on every device (set by the test hook below).
std::atomic<int> g_forced_program{-1};

int ProgramFor(int num_sms, int quant) {
  const int forced = g_forced_program.load(std::memory_order_relaxed);
  if (forced >= 0) return forced;
  return (num_sms == kWarpEpilogueSms || (num_sms == kWarpEpilogueNvfp4Sms && quant == 1)) ? 1 : 0;
}

void CheckTensor(const TensorView& tensor, const char* name, DLDevice device, DLDataType dtype,
                 std::initializer_list<int64_t> shape) {
  minimax_h3_sm120::CheckTensor(tensor, name, device, dtype, shape, "attn_out");
}

using minimax_h3_sm120::EncodeByteTile;

struct DeviceTag {};

int ConfigureKernels(int device_id) {
  return minimax_h3_sm120::DeviceConfig<DeviceTag, int>::Get(
      "MiniMax-H3 SM120 quantized output projection", device_id, [](const cudaDeviceProp& properties) {
        for (const auto& program : kGemmVariants) {
          for (const auto& variant : program) {
            minimax_h3_sm120::OptInDynamicSmem(variant.kernel, variant.dynamic_smem_bytes);
          }
        }
        return properties.multiProcessorCount;
      });
}

using LaunchPlan = minimax_h3_sm120::GemmLaunchPlan;

LaunchPlan MakeLaunchPlan(int64_t rows, int num_sms) {
  // Mirrors the Python launch_plan(): one persistent CTA per SM (ctas_per_sm = 1).  The in-kernel
  // quantization / TMA flag protocol needs every CTA resident, which a grid of at most one CTA per
  // SM (384 threads, ~92 KB dynamic SMEM) guarantees.  There is no separate quantization launch
  // (quant_grid is unused).
  return minimax_h3_sm120::MakeGemmLaunchPlan(rows, num_sms, kBlockM, kNTiles, 1);
}

struct CommonArgs {
  int64_t rows;
  DLDevice device;
  cudaStream_t stream;
  int num_sms;
  LaunchPlan plan;
  __nv_bfloat16* gate;
  int* gate_index;
  __nv_bfloat16* residual;
  unsigned int* out;
};

// Shared validation of the BF16 operands and the caller-owned output.
CommonArgs CheckCommon(const TensorView& attn_out, const TensorView& gate, const TensorView& gate_index,
                       const TensorView& residual, const TensorView& out) {
  TVM_FFI_CHECK(attn_out.ndim() == 2 && attn_out.size(0) >= 1 && attn_out.size(0) <= kMaxRows &&
                    attn_out.size(1) == kAttnDim, ValueError)
      << "attn_out must be [M, 7168] with 1 <= M <= " << kMaxRows;
  CommonArgs args{};
  args.rows = attn_out.size(0);
  args.device = attn_out.device();
  CheckTensor(attn_out, "attn_out", args.device, dl_bfloat16, {args.rows, kAttnDim});
  TVM_FFI_CHECK(gate.ndim() == 2 && gate.size(0) >= 1 && gate.size(1) == kHidden, ValueError)
      << "gate must be [rows, 5376]";
  CheckTensor(gate, "gate", args.device, dl_bfloat16, {gate.size(0), kHidden});
  CheckTensor(gate_index, "gate_index", args.device, dl_int32, {args.rows});
  CheckTensor(residual, "residual", args.device, dl_bfloat16, {args.rows, kHidden});
  CheckTensor(out, "out", args.device, dl_bfloat16, {args.rows, kHidden});
  TVM_FFI_CHECK(gate.size(0) == 9, ValueError)
      << "gate must have exactly 9 rows (the MiniMax-H3 AdaLN plan); indices outside the table contribute zero";
  args.gate = static_cast<__nv_bfloat16*>(gate.data_ptr());
  args.gate_index = static_cast<int*>(gate_index.data_ptr());
  args.residual = static_cast<__nv_bfloat16*>(residual.data_ptr());
  args.out = static_cast<unsigned int*>(out.data_ptr());
  args.num_sms = ConfigureKernels(args.device.device_id);
  args.stream = get_stream(args.device);
  args.plan = MakeLaunchPlan(args.rows, args.num_sms);
  return args;
}

struct QuantOutputs {
  unsigned int* attn_out;  // the BF16 [M, 7168] rows as packed 32-bit words
  unsigned int* act_q;
  uint8_t* act_sf;          // NVFP4 only (nullptr for FP8: never dereferenced)
  float* act_scale;         // FP8 only (a valid dummy for NVFP4: read for the epilogue row scale = 1.0)
  float* act_global_scale;  // NVFP4 only (a valid dummy for FP8: read once)
};

void LaunchGemm(int quant, const CommonArgs& args, const CUtensorMap& a_map, const CUtensorMap& b_map,
                const CUtensorMap& sfa_map, const CUtensorMap& sfb_map, const QuantOutputs& q, float* w_scale,
                double alpha, const char* what) {
  const GemmVariant variant = kGemmVariants[ProgramFor(args.num_sms, quant)][quant];
  // Per-M-tile ready counters (rows quantized so far), zeroed per call: a stream-ordered scratch
  // allocation so the public API needs no caller workspace.
  unsigned int* flags = nullptr;
  const size_t flag_bytes = static_cast<size_t>(args.plan.num_m_tiles) * sizeof(unsigned int);
  cudaError_t status = cudaMallocAsync(reinterpret_cast<void**>(&flags), flag_bytes, args.stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << what << " flag workspace allocation failed: " << cudaGetErrorString(status);
  status = cudaMemsetAsync(flags, 0, flag_bytes, args.stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << what << " flag workspace memset failed: " << cudaGetErrorString(status);
  variant.kernel<<<dim3(args.plan.gemm_grid), dim3(kGemmThreads), variant.dynamic_smem_bytes, args.stream>>>(
      a_map, b_map, sfa_map, sfb_map, q.attn_out, q.act_q, q.act_sf, q.act_scale, q.act_global_scale, w_scale,
      args.gate, args.gate_index, args.residual, args.out, flags, static_cast<int>(args.rows), args.plan.num_m_tiles,
      args.plan.total_tiles, static_cast<float>(alpha));
  status = cudaGetLastError();
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << what << " GEMM launch failed: " << cudaGetErrorString(status);
  status = cudaFreeAsync(flags, args.stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << what << " flag workspace release failed: " << cudaGetErrorString(status);
}

// Single-element FP32 constant 1.0 per device (dummy row scale / global scale of the other route).
float* OnesScalar(DLDevice device, cudaStream_t stream) {
  static std::mutex mutex;
  static std::vector<std::pair<int, float*>> per_device;
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto& entry : per_device) {
    if (entry.first == device.device_id) return entry.second;
  }
  float* ptr = nullptr;
  cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&ptr), 16 * sizeof(float));
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << "constant allocation failed: " << cudaGetErrorString(status);
  const float ones[16] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  status = cudaMemcpyAsync(ptr, ones, sizeof(ones), cudaMemcpyHostToDevice, stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << "constant upload failed: " << cudaGetErrorString(status);
  status = cudaStreamSynchronize(stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << "constant upload sync failed: " << cudaGetErrorString(status);
  per_device.emplace_back(device.device_id, ptr);
  return ptr;
}

}  // namespace

// FP8 W8A8 route.  attn_out [M, 7168] BF16 -> act_q [M, 7168] E4M3 (per-token scale act_scale [M] FP32 =
// RN(amax / 448)) -> o = BF16(act_q @ o_weight_q^T * act_scale[m] * o_weight_scale[n])
// -> out [M, 5376] BF16 = BF16(residual + BF16(gate[gate_index[m]] * o)).
// o_weight_q: E4M3 [5376, 7168] (per-output-channel), o_weight_scale: FP32 [5376].
void minimax_h3_sm120_fp8_out_proj(TensorView attn_out, TensorView o_weight_q, TensorView o_weight_scale,
                                   TensorView gate, TensorView gate_index, TensorView residual, TensorView act_q,
                                   TensorView act_scale, TensorView out) {
  const CommonArgs args = CheckCommon(attn_out, gate, gate_index, residual, out);
  CheckTensor(o_weight_q, "o_weight_q", args.device, dl_float8_e4m3fn, {kHidden, kAttnDim});
  CheckTensor(o_weight_scale, "o_weight_scale", args.device, dl_float32, {kHidden});
  CheckTensor(act_q, "act_q", args.device, dl_float8_e4m3fn, {args.rows, kAttnDim});
  CheckTensor(act_scale, "act_scale", args.device, dl_float32, {args.rows});
  ffi::CUDADeviceGuard device_guard(args.device.device_id);

  const CUtensorMap a_map = EncodeByteTile(act_q.data_ptr(), kAttnDim, args.rows, kBlockK, kBlockM,
                                           kOperandSwizzle, "act_q");
  const CUtensorMap b_map = EncodeByteTile(o_weight_q.data_ptr(), kAttnDim, kHidden, kBlockK, kBlockN,
                                           kOperandSwizzle, "o_weight_q");
  // The FP8 kernel never issues scale-tile loads; its descriptors only need valid parameter storage.
  const CUtensorMap unused_map{};
  QuantOutputs q{};
  q.attn_out = static_cast<unsigned int*>(attn_out.data_ptr());
  q.act_q = static_cast<unsigned int*>(act_q.data_ptr());
  q.act_sf = nullptr;
  q.act_scale = static_cast<float*>(act_scale.data_ptr());
  q.act_global_scale = OnesScalar(args.device, args.stream);
  LaunchGemm(0, args, a_map, b_map, unused_map, unused_map, q, static_cast<float*>(o_weight_scale.data_ptr()), 1.0,
             "MiniMax-H3 FP8");
}

// NVFP4 route (FlashInfer conventions).  attn_out -> act_q [M, 3584] u8 (E2M1x2) + act_sf [M, 448] UE4M3 u8
// (row-major, block 16) with the caller's activation global scale act_global_scale [1] FP32
// (448 * 6 / amax).  o_weight_q: u8 [5376, 3584], o_weight_sf: u8 with 5376 * 448 entries in the
// FlashInfer 128x4 swizzled layout (fp4_quantize(..., is_sf_swizzled_layout=True)).
// The kernel rescales the block-scaled accumulator by alpha = FP32(1 / (FP64(act_global_scale) *
// FP64(o_weight_global_scale))), computed on the device from the two FP32 [1] global scales.
void minimax_h3_sm120_nvfp4_out_proj(TensorView attn_out, TensorView o_weight_q, TensorView o_weight_sf,
                                     TensorView act_global_scale, TensorView gate, TensorView gate_index,
                                     TensorView residual, TensorView act_q, TensorView act_sf, TensorView out,
                                     TensorView o_weight_global_scale) {
  const CommonArgs args = CheckCommon(attn_out, gate, gate_index, residual, out);
  CheckTensor(o_weight_q, "o_weight_q", args.device, dl_uint8, {kHidden, kAttnDim / 2});
  TVM_FFI_CHECK(o_weight_sf.device().device_type == kDLCUDA &&
                    o_weight_sf.device().device_id == args.device.device_id, ValueError)
      << "o_weight_sf must be a CUDA tensor on the same device as attn_out";
  TVM_FFI_CHECK(encode_dlpack_dtype(o_weight_sf.dtype()) == encode_dlpack_dtype(dl_uint8), ValueError)
      << "o_weight_sf must be uint8";
  int64_t sf_numel = 1;
  for (int dim = 0; dim < o_weight_sf.ndim(); ++dim) sf_numel *= o_weight_sf.size(dim);
  TVM_FFI_CHECK(sf_numel == kHidden * kAttnSf && o_weight_sf.IsContiguous(), ValueError)
      << "o_weight_sf must be a contiguous uint8 tensor with 5376 * 448 entries (128x4 swizzled layout)";
  CheckTensor(act_global_scale, "act_global_scale", args.device, dl_float32, {1});
  CheckTensor(o_weight_global_scale, "o_weight_global_scale", args.device, dl_float32, {1});
  CheckTensor(act_q, "act_q", args.device, dl_uint8, {args.rows, kAttnDim / 2});
  CheckTensor(act_sf, "act_sf", args.device, dl_uint8, {args.rows, kAttnSf});
  ffi::CUDADeviceGuard device_guard(args.device.device_id);

  const CUtensorMap a_map = EncodeByteTile(act_q.data_ptr(), kAttnDim / 2, args.rows, kBlockK, kBlockM,
                                           kOperandSwizzle, "act_q");
  const CUtensorMap b_map = EncodeByteTile(o_weight_q.data_ptr(), kAttnDim / 2, kHidden, kBlockK, kBlockN,
                                           kOperandSwizzle, "o_weight_q");
  const CUtensorMap sfa_map = EncodeByteTile(act_sf.data_ptr(), kAttnSf, args.rows, kSfaBoxBytes,
                                             kBlockM, CU_TENSOR_MAP_SWIZZLE_NONE, "act_sf");
  // The 128x4 swizzled weight scales are addressed as [42 N tiles x 224 rows, 256 bytes].
  const CUtensorMap sfb_map = EncodeByteTile(o_weight_sf.data_ptr(), 256, int64_t{kNTiles} * kSfbRowsPerNTile,
                                             256, kSfbRowsPerKTile, CU_TENSOR_MAP_SWIZZLE_NONE, "o_weight_sf");
  QuantOutputs q{};
  q.attn_out = static_cast<unsigned int*>(attn_out.data_ptr());
  q.act_q = static_cast<unsigned int*>(act_q.data_ptr());
  q.act_sf = static_cast<uint8_t*>(act_sf.data_ptr());
  q.act_scale = OnesScalar(args.device, args.stream);
  q.act_global_scale = static_cast<float*>(act_global_scale.data_ptr());
  // The weight global scale is the kernel's w_scale operand; its alpha argument is unused.
  LaunchGemm(1, args, a_map, b_map, sfa_map, sfb_map, q, static_cast<float*>(o_weight_global_scale.data_ptr()),
             0.0, "MiniMax-H3 NVFP4");
}

// Test hooks: run program 0 or 1 on every device (-1 restores the routing by device and quant), and report the
// program a call of quant (0 = FP8, 1 = NVFP4) on device_id runs.
void minimax_h3_sm120_out_proj_force_program(int64_t program) {
  TVM_FFI_CHECK(program >= -1 && program <= 1, ValueError) << "program must be -1 (route by device), 0 or 1";
  g_forced_program.store(static_cast<int>(program), std::memory_order_relaxed);
}

int64_t minimax_h3_sm120_out_proj_program(int64_t device_id, int64_t quant) {
  TVM_FFI_CHECK(quant == 0 || quant == 1, ValueError) << "quant must be 0 (FP8) or 1 (NVFP4)";
  return ProgramFor(ConfigureKernels(static_cast<int>(device_id)), static_cast<int>(quant));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_sm120_fp8_out_proj, minimax_h3_sm120_fp8_out_proj);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_sm120_nvfp4_out_proj, minimax_h3_sm120_nvfp4_out_proj);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_sm120_out_proj_force_program, minimax_h3_sm120_out_proj_force_program);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_sm120_out_proj_program, minimax_h3_sm120_out_proj_program);
// clang-format on
