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
#define NUM_MAIN_1_STAGES 1
#define SMEM_Q_STAGE_OFF 0
#define SMEM_Q_STAGE_STAGE_BYTES 32768
#define SMEM_Q_STAGE_STRIDE 32768
#define SMEM_V_STAGE_OFF 42656
#define SMEM_V_STAGE_STAGE_BYTES 32768
#define SMEM_V_STAGE_STRIDE 32768
#define SMEM_RED_OFF 33792
#define SMEM_RED_STAGE_BYTES 8192
#define SMEM_RED_STRIDE 8192
#define SMEM_QM_SMEM_OFF 41984
#define SMEM_QM_SMEM_STAGE_BYTES 512
#define SMEM_QM_SMEM_STRIDE 512
#define SMEM_QM_CODES_OFF 42496
#define SMEM_QM_CODES_STAGE_BYTES 128
#define SMEM_QM_CODES_STRIDE 128
#define SMEM_QM_SF_OFF 42624
#define SMEM_QM_SF_STAGE_BYTES 32
#define SMEM_QM_SF_STRIDE 32
#define SMEM_TOTAL 75520
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 3

#include <math_constants.h>


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

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_nvfp4_attention_f0a6fedc1ec9c401610b(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, float* __restrict__ mean_k, uint8_t* __restrict__ q4, uint8_t* __restrict__ sfq, uint8_t* __restrict__ qm4, uint8_t* __restrict__ sfqm, uint8_t* __restrict__ k4, uint8_t* __restrict__ sfk, uint8_t* __restrict__ vt4, uint8_t* __restrict__ sfv_lo, uint8_t* __restrict__ sfv_hi, int seqlen, int q_stride, int kv_stride, int heads, int num_q_blocks, int total_bh, long long q_stride_s, long long q_stride_h, long long q_stride_b, long long k_stride_s, long long k_stride_h, long long k_stride_b, long long v_stride_s, long long v_stride_h, long long v_stride_b, uint8_t* __restrict__ q4l, uint8_t* __restrict__ sfq_l, uint8_t* __restrict__ k4l, uint8_t* __restrict__ sfk_l)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* q_stage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int q_stage_addr = smem + 0;
    __nv_bfloat16* v_stage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 42656);
    const int v_stage_addr = smem + 42656;
    float* red = reinterpret_cast<float*>(smem_raw + 33792);
    const int red_addr = smem + 33792;
    float* qm_smem = reinterpret_cast<float*>(smem_raw + 41984);
    const int qm_smem_addr = smem + 41984;
    unsigned int* qm_codes = reinterpret_cast<unsigned int*>(smem_raw + 42496);
    const int qm_codes_addr = smem + 42496;
    float* qm_sf = reinterpret_cast<float*>(smem_raw + 42624);
    const int qm_sf_addr = smem + 42624;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int col = (tid & 15) * 8;
    int rgrp = tid >> 4;
    int vrow = tid >> 3;
    int group = tid & 7;
    int group_pair = group >> 2;
    int group_lane = group & 3;
    float chunk_vals[8];
    unsigned int chunk_words[4];
    unsigned int row_words[4];
    unsigned int qv_lo[4];
    unsigned int qv_hi[4];
    float acc[8];
    float row_vals[8];
    float qv[16];
    float x[16];
    unsigned int q_codes0[2];
    unsigned int q_codes1[2];
    unsigned int q_codes2[2];
    unsigned int q_codes3[2];
    float q_sf[4];
    float sf_tmp[1];
    float qm_slice[16];
    float km_slice[16];
    unsigned int codes[2];
    unsigned int q_codes_l0[2];
    unsigned int q_codes_l1[2];
    unsigned int q_codes_l2[2];
    unsigned int q_codes_l3[2];
    float q_sf_l[4];
    float xt[4];
    #pragma unroll 1
    for (int work = bid; work < total_bh * num_q_blocks; work += num_bids) {
        int bh = work / num_q_blocks;
        int block = work - bh * num_q_blocks;
        int batch = bh / heads;
        int head = bh - batch * heads;
        int token_base = block * 128;
        int n_real = seqlen - token_base;
        n_real = ((n_real > 128) ? 128 : n_real);
        n_real = ((n_real < 0) ? 0 : n_real);
        long long q_base = (long long)batch * q_stride_b + (long long)head * q_stride_h + (long long)token_base * q_stride_s;
        long long k_base = (long long)batch * k_stride_b + (long long)head * k_stride_h + (long long)token_base * k_stride_s;
        long long v_base = (long long)batch * v_stride_b + (long long)head * v_stride_h + (long long)token_base * v_stride_s;
        long long q_row0 = (long long)bh * (long long)q_stride + (long long)token_base;
        long long q_tile = (long long)work * 1024;
        int kv_blocks = kv_stride >> 7;
        int has_kv = ((block < kv_blocks) ? 1 : 0);
        long long kv_row0 = (long long)bh * (long long)kv_stride + (long long)token_base;
        long long kv_tile = (long long)bh * (long long)kv_blocks + (long long)block;
        int qcrow0 = tid >> 4;
        int qcchunk0 = tid & 15;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(q_stage_addr + (unsigned int)((qcrow0 * 128 + qcchunk0 * 8) * 2)), "l"(Q + (q_base + (long long)qcrow0 * q_stride_s + (long long)(qcchunk0 * 8))), "r"((qcrow0 < n_real) ? 16 : 0));
        int qcrow1 = tid + 256 >> 4;
        int qcchunk1 = tid + 256 & 15;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(q_stage_addr + (unsigned int)((qcrow1 * 128 + qcchunk1 * 8) * 2)), "l"(Q + (q_base + (long long)qcrow1 * q_stride_s + (long long)(qcchunk1 * 8))), "r"((qcrow1 < n_real) ? 16 : 0));
        int qcrow2 = tid + 512 >> 4;
        int qcchunk2 = tid + 512 & 15;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(q_stage_addr + (unsigned int)((qcrow2 * 128 + qcchunk2 * 8) * 2)), "l"(Q + (q_base + (long long)qcrow2 * q_stride_s + (long long)(qcchunk2 * 8))), "r"((qcrow2 < n_real) ? 16 : 0));
        int qcrow3 = tid + 768 >> 4;
        int qcchunk3 = tid + 768 & 15;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(q_stage_addr + (unsigned int)((qcrow3 * 128 + qcchunk3 * 8) * 2)), "l"(Q + (q_base + (long long)qcrow3 * q_stride_s + (long long)(qcchunk3 * 8))), "r"((qcrow3 < n_real) ? 16 : 0));
        int qcrow4 = tid + 1024 >> 4;
        int qcchunk4 = tid + 1024 & 15;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(q_stage_addr + (unsigned int)((qcrow4 * 128 + qcchunk4 * 8) * 2)), "l"(Q + (q_base + (long long)qcrow4 * q_stride_s + (long long)(qcchunk4 * 8))), "r"((qcrow4 < n_real) ? 16 : 0));
        int qcrow5 = tid + 1280 >> 4;
        int qcchunk5 = tid + 1280 & 15;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(q_stage_addr + (unsigned int)((qcrow5 * 128 + qcchunk5 * 8) * 2)), "l"(Q + (q_base + (long long)qcrow5 * q_stride_s + (long long)(qcchunk5 * 8))), "r"((qcrow5 < n_real) ? 16 : 0));
        int qcrow6 = tid + 1536 >> 4;
        int qcchunk6 = tid + 1536 & 15;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(q_stage_addr + (unsigned int)((qcrow6 * 128 + qcchunk6 * 8) * 2)), "l"(Q + (q_base + (long long)qcrow6 * q_stride_s + (long long)(qcchunk6 * 8))), "r"((qcrow6 < n_real) ? 16 : 0));
        int qcrow7 = tid + 1792 >> 4;
        int qcchunk7 = tid + 1792 & 15;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(q_stage_addr + (unsigned int)((qcrow7 * 128 + qcchunk7 * 8) * 2)), "l"(Q + (q_base + (long long)qcrow7 * q_stride_s + (long long)(qcchunk7 * 8))), "r"((qcrow7 < n_real) ? 16 : 0));
        asm volatile("cp.async.commit_group;");
        int vcrow0 = tid >> 4;
        int vcchunk0 = tid & 15;
        int vccol0 = vcchunk0 * 8 ^ vcrow0 >> 4 << 3;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(v_stage_addr + (unsigned int)((vcrow0 * 128 + vccol0) * 2)), "l"(V + (v_base + (long long)vcrow0 * v_stride_s + (long long)(vcchunk0 * 8))), "r"((vcrow0 < n_real && has_kv == 1) ? 16 : 0));
        int vcrow1 = tid + 256 >> 4;
        int vcchunk1 = tid + 256 & 15;
        int vccol1 = vcchunk1 * 8 ^ vcrow1 >> 4 << 3;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(v_stage_addr + (unsigned int)((vcrow1 * 128 + vccol1) * 2)), "l"(V + (v_base + (long long)vcrow1 * v_stride_s + (long long)(vcchunk1 * 8))), "r"((vcrow1 < n_real && has_kv == 1) ? 16 : 0));
        int vcrow2 = tid + 512 >> 4;
        int vcchunk2 = tid + 512 & 15;
        int vccol2 = vcchunk2 * 8 ^ vcrow2 >> 4 << 3;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(v_stage_addr + (unsigned int)((vcrow2 * 128 + vccol2) * 2)), "l"(V + (v_base + (long long)vcrow2 * v_stride_s + (long long)(vcchunk2 * 8))), "r"((vcrow2 < n_real && has_kv == 1) ? 16 : 0));
        int vcrow3 = tid + 768 >> 4;
        int vcchunk3 = tid + 768 & 15;
        int vccol3 = vcchunk3 * 8 ^ vcrow3 >> 4 << 3;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(v_stage_addr + (unsigned int)((vcrow3 * 128 + vccol3) * 2)), "l"(V + (v_base + (long long)vcrow3 * v_stride_s + (long long)(vcchunk3 * 8))), "r"((vcrow3 < n_real && has_kv == 1) ? 16 : 0));
        int vcrow4 = tid + 1024 >> 4;
        int vcchunk4 = tid + 1024 & 15;
        int vccol4 = vcchunk4 * 8 ^ vcrow4 >> 4 << 3;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(v_stage_addr + (unsigned int)((vcrow4 * 128 + vccol4) * 2)), "l"(V + (v_base + (long long)vcrow4 * v_stride_s + (long long)(vcchunk4 * 8))), "r"((vcrow4 < n_real && has_kv == 1) ? 16 : 0));
        int vcrow5 = tid + 1280 >> 4;
        int vcchunk5 = tid + 1280 & 15;
        int vccol5 = vcchunk5 * 8 ^ vcrow5 >> 4 << 3;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(v_stage_addr + (unsigned int)((vcrow5 * 128 + vccol5) * 2)), "l"(V + (v_base + (long long)vcrow5 * v_stride_s + (long long)(vcchunk5 * 8))), "r"((vcrow5 < n_real && has_kv == 1) ? 16 : 0));
        int vcrow6 = tid + 1536 >> 4;
        int vcchunk6 = tid + 1536 & 15;
        int vccol6 = vcchunk6 * 8 ^ vcrow6 >> 4 << 3;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(v_stage_addr + (unsigned int)((vcrow6 * 128 + vccol6) * 2)), "l"(V + (v_base + (long long)vcrow6 * v_stride_s + (long long)(vcchunk6 * 8))), "r"((vcrow6 < n_real && has_kv == 1) ? 16 : 0));
        int vcrow7 = tid + 1792 >> 4;
        int vcchunk7 = tid + 1792 & 15;
        int vccol7 = vcchunk7 * 8 ^ vcrow7 >> 4 << 3;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(v_stage_addr + (unsigned int)((vcrow7 * 128 + vccol7) * 2)), "l"(V + (v_base + (long long)vcrow7 * v_stride_s + (long long)(vcchunk7 * 8))), "r"((vcrow7 < n_real && has_kv == 1) ? 16 : 0));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&row_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 3]))
            : "r"(q_stage_addr + (unsigned int)((rgrp * 128 + col) * 2)));
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&row_vals[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) << 16);
            (&row_vals[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) & 0xffff0000u);
        }
        acc[0] = acc[0] + row_vals[0];
        acc[1] = acc[1] + row_vals[1];
        acc[2] = acc[2] + row_vals[2];
        acc[3] = acc[3] + row_vals[3];
        acc[4] = acc[4] + row_vals[4];
        acc[5] = acc[5] + row_vals[5];
        acc[6] = acc[6] + row_vals[6];
        acc[7] = acc[7] + row_vals[7];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&row_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 3]))
            : "r"(q_stage_addr + (unsigned int)(((16 + rgrp) * 128 + col) * 2)));
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&row_vals[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) << 16);
            (&row_vals[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) & 0xffff0000u);
        }
        acc[0] = acc[0] + row_vals[0];
        acc[1] = acc[1] + row_vals[1];
        acc[2] = acc[2] + row_vals[2];
        acc[3] = acc[3] + row_vals[3];
        acc[4] = acc[4] + row_vals[4];
        acc[5] = acc[5] + row_vals[5];
        acc[6] = acc[6] + row_vals[6];
        acc[7] = acc[7] + row_vals[7];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&row_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 3]))
            : "r"(q_stage_addr + (unsigned int)(((32 + rgrp) * 128 + col) * 2)));
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&row_vals[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) << 16);
            (&row_vals[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) & 0xffff0000u);
        }
        acc[0] = acc[0] + row_vals[0];
        acc[1] = acc[1] + row_vals[1];
        acc[2] = acc[2] + row_vals[2];
        acc[3] = acc[3] + row_vals[3];
        acc[4] = acc[4] + row_vals[4];
        acc[5] = acc[5] + row_vals[5];
        acc[6] = acc[6] + row_vals[6];
        acc[7] = acc[7] + row_vals[7];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&row_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 3]))
            : "r"(q_stage_addr + (unsigned int)(((48 + rgrp) * 128 + col) * 2)));
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&row_vals[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) << 16);
            (&row_vals[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) & 0xffff0000u);
        }
        acc[0] = acc[0] + row_vals[0];
        acc[1] = acc[1] + row_vals[1];
        acc[2] = acc[2] + row_vals[2];
        acc[3] = acc[3] + row_vals[3];
        acc[4] = acc[4] + row_vals[4];
        acc[5] = acc[5] + row_vals[5];
        acc[6] = acc[6] + row_vals[6];
        acc[7] = acc[7] + row_vals[7];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&row_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 3]))
            : "r"(q_stage_addr + (unsigned int)(((64 + rgrp) * 128 + col) * 2)));
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&row_vals[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) << 16);
            (&row_vals[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) & 0xffff0000u);
        }
        acc[0] = acc[0] + row_vals[0];
        acc[1] = acc[1] + row_vals[1];
        acc[2] = acc[2] + row_vals[2];
        acc[3] = acc[3] + row_vals[3];
        acc[4] = acc[4] + row_vals[4];
        acc[5] = acc[5] + row_vals[5];
        acc[6] = acc[6] + row_vals[6];
        acc[7] = acc[7] + row_vals[7];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&row_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 3]))
            : "r"(q_stage_addr + (unsigned int)(((80 + rgrp) * 128 + col) * 2)));
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&row_vals[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) << 16);
            (&row_vals[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) & 0xffff0000u);
        }
        acc[0] = acc[0] + row_vals[0];
        acc[1] = acc[1] + row_vals[1];
        acc[2] = acc[2] + row_vals[2];
        acc[3] = acc[3] + row_vals[3];
        acc[4] = acc[4] + row_vals[4];
        acc[5] = acc[5] + row_vals[5];
        acc[6] = acc[6] + row_vals[6];
        acc[7] = acc[7] + row_vals[7];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&row_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 3]))
            : "r"(q_stage_addr + (unsigned int)(((96 + rgrp) * 128 + col) * 2)));
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&row_vals[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) << 16);
            (&row_vals[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) & 0xffff0000u);
        }
        acc[0] = acc[0] + row_vals[0];
        acc[1] = acc[1] + row_vals[1];
        acc[2] = acc[2] + row_vals[2];
        acc[3] = acc[3] + row_vals[3];
        acc[4] = acc[4] + row_vals[4];
        acc[5] = acc[5] + row_vals[5];
        acc[6] = acc[6] + row_vals[6];
        acc[7] = acc[7] + row_vals[7];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&row_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 3]))
            : "r"(q_stage_addr + (unsigned int)(((112 + rgrp) * 128 + col) * 2)));
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&row_vals[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) << 16);
            (&row_vals[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(row_words[_pair]) & 0xffff0000u);
        }
        acc[0] = acc[0] + row_vals[0];
        acc[1] = acc[1] + row_vals[1];
        acc[2] = acc[2] + row_vals[2];
        acc[3] = acc[3] + row_vals[3];
        acc[4] = acc[4] + row_vals[4];
        acc[5] = acc[5] + row_vals[5];
        acc[6] = acc[6] + row_vals[6];
        acc[7] = acc[7] + row_vals[7];
        red[rgrp * 128 + col] = acc[0];
        red[rgrp * 128 + col + 1] = acc[1];
        red[rgrp * 128 + col + 2] = acc[2];
        red[rgrp * 128 + col + 3] = acc[3];
        red[rgrp * 128 + col + 4] = acc[4];
        red[rgrp * 128 + col + 5] = acc[5];
        red[rgrp * 128 + col + 6] = acc[6];
        red[rgrp * 128 + col + 7] = acc[7];
        __syncthreads();
        if (tid < 128) {
            float total = 0.0f;
            #pragma unroll
            for (int g = 0; g < 16; g++) {
                total = total + red[g * 128 + tid];
            }
            float den = (float)(((n_real > 0) ? n_real : 1));
            float _fdiv_0 = __fdiv_rn(total, den);
            qm_smem[tid] = _fdiv_0;
        }
        __syncthreads();
        qm_slice[0] = qm_smem[group * 16];
        qm_slice[1] = qm_smem[group * 16 + 1];
        qm_slice[2] = qm_smem[group * 16 + 2];
        qm_slice[3] = qm_smem[group * 16 + 3];
        qm_slice[4] = qm_smem[group * 16 + 4];
        qm_slice[5] = qm_smem[group * 16 + 5];
        qm_slice[6] = qm_smem[group * 16 + 6];
        qm_slice[7] = qm_smem[group * 16 + 7];
        qm_slice[8] = qm_smem[group * 16 + 8];
        qm_slice[9] = qm_smem[group * 16 + 9];
        qm_slice[10] = qm_smem[group * 16 + 10];
        qm_slice[11] = qm_smem[group * 16 + 11];
        qm_slice[12] = qm_smem[group * 16 + 12];
        qm_slice[13] = qm_smem[group * 16 + 13];
        qm_slice[14] = qm_smem[group * 16 + 14];
        qm_slice[15] = qm_smem[group * 16 + 15];
        unsigned int q_addr0 = q_stage_addr + (unsigned int)((vrow * 128 + group * 16) * 2);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[(0) + 3]))
            : "r"(q_addr0));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[(0) + 3]))
            : "r"(q_addr0 + 16));
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&qv[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qv_lo[_pair]) << 16);
            (&qv[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qv_lo[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&qv[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qv_hi[_pair]) << 16);
            (&qv[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qv_hi[_pair]) & 0xffff0000u);
        }
        x[0] = ((n_real > vrow) ? qv[0] - qm_slice[0] : 0.0f);
        x[1] = ((n_real > vrow) ? qv[1] - qm_slice[1] : 0.0f);
        x[2] = ((n_real > vrow) ? qv[2] - qm_slice[2] : 0.0f);
        x[3] = ((n_real > vrow) ? qv[3] - qm_slice[3] : 0.0f);
        x[4] = ((n_real > vrow) ? qv[4] - qm_slice[4] : 0.0f);
        x[5] = ((n_real > vrow) ? qv[5] - qm_slice[5] : 0.0f);
        x[6] = ((n_real > vrow) ? qv[6] - qm_slice[6] : 0.0f);
        x[7] = ((n_real > vrow) ? qv[7] - qm_slice[7] : 0.0f);
        x[8] = ((n_real > vrow) ? qv[8] - qm_slice[8] : 0.0f);
        x[9] = ((n_real > vrow) ? qv[9] - qm_slice[9] : 0.0f);
        x[10] = ((n_real > vrow) ? qv[10] - qm_slice[10] : 0.0f);
        x[11] = ((n_real > vrow) ? qv[11] - qm_slice[11] : 0.0f);
        x[12] = ((n_real > vrow) ? qv[12] - qm_slice[12] : 0.0f);
        x[13] = ((n_real > vrow) ? qv[13] - qm_slice[13] : 0.0f);
        x[14] = ((n_real > vrow) ? qv[14] - qm_slice[14] : 0.0f);
        x[15] = ((n_real > vrow) ? qv[15] - qm_slice[15] : 0.0f);
        float _max_0 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
        uint16_t _e4m3x2_3;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_3) : "f"(0.0f), "f"(_max_0));
        uint8_t _cvt_satfinite_e4m3_f32_2 = (uint8_t)_e4m3x2_3;
        uint32_t _f16x2_decode_4;
        float _fp8_e4m3_decode_1;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_4) : "h"(_e4m3x2_3));
        uint16_t _f16_decode_4 = (uint16_t)_f16x2_decode_4;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_1) : "h"(_f16_decode_4));
        float _fdiv_rn_0 = __fdiv_rn(1.0f, _fp8_e4m3_decode_1);
        x[0] = x[0] * _fdiv_rn_0;
        x[1] = x[1] * _fdiv_rn_0;
        x[2] = x[2] * _fdiv_rn_0;
        x[3] = x[3] * _fdiv_rn_0;
        x[4] = x[4] * _fdiv_rn_0;
        x[5] = x[5] * _fdiv_rn_0;
        x[6] = x[6] * _fdiv_rn_0;
        x[7] = x[7] * _fdiv_rn_0;
        x[8] = x[8] * _fdiv_rn_0;
        x[9] = x[9] * _fdiv_rn_0;
        x[10] = x[10] * _fdiv_rn_0;
        x[11] = x[11] * _fdiv_rn_0;
        x[12] = x[12] * _fdiv_rn_0;
        x[13] = x[13] * _fdiv_rn_0;
        x[14] = x[14] * _fdiv_rn_0;
        x[15] = x[15] * _fdiv_rn_0;
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes0[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes0[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
        sf_tmp[0] = _max_0;
        q_sf[0] = sf_tmp[0];
        x[0] = ((n_real > vrow) ? qv[0] - qm_slice[0] : 0.0f);
        x[1] = ((n_real > vrow) ? qv[1] - qm_slice[1] : 0.0f);
        x[2] = ((n_real > vrow) ? qv[2] - qm_slice[2] : 0.0f);
        x[3] = ((n_real > vrow) ? qv[3] - qm_slice[3] : 0.0f);
        x[4] = ((n_real > vrow) ? qv[4] - qm_slice[4] : 0.0f);
        x[5] = ((n_real > vrow) ? qv[5] - qm_slice[5] : 0.0f);
        x[6] = ((n_real > vrow) ? qv[6] - qm_slice[6] : 0.0f);
        x[7] = ((n_real > vrow) ? qv[7] - qm_slice[7] : 0.0f);
        x[8] = ((n_real > vrow) ? qv[8] - qm_slice[8] : 0.0f);
        x[9] = ((n_real > vrow) ? qv[9] - qm_slice[9] : 0.0f);
        x[10] = ((n_real > vrow) ? qv[10] - qm_slice[10] : 0.0f);
        x[11] = ((n_real > vrow) ? qv[11] - qm_slice[11] : 0.0f);
        x[12] = ((n_real > vrow) ? qv[12] - qm_slice[12] : 0.0f);
        x[13] = ((n_real > vrow) ? qv[13] - qm_slice[13] : 0.0f);
        x[14] = ((n_real > vrow) ? qv[14] - qm_slice[14] : 0.0f);
        x[15] = ((n_real > vrow) ? qv[15] - qm_slice[15] : 0.0f);
        uint16_t _e4m3x2_6;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_6) : "f"(0.0f), "f"(sf_tmp[0]));
        uint8_t _cvt_satfinite_e4m3_f32_5 = (uint8_t)_e4m3x2_6;
        uint32_t _f16x2_decode_7;
        float _fp8_decode_0;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_7) : "h"(_e4m3x2_6));
        uint16_t _f16_decode_7 = (uint16_t)_f16x2_decode_7;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_0) : "h"(_f16_decode_7));
        float _unpack_fp4_0[16];
        {
            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
            uint32_t _fp4_decode_pairs[4];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_codes0[0]));
            (&_unpack_fp4_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
            (&_unpack_fp4_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
            (&_unpack_fp4_0[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
            (&_unpack_fp4_0[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
            (&_unpack_fp4_0[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
            (&_unpack_fp4_0[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
            (&_unpack_fp4_0[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
            (&_unpack_fp4_0[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
            #else
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_0[0]), "=f"(_unpack_fp4_0[1]), "=f"(_unpack_fp4_0[2]), "=f"(_unpack_fp4_0[3]), "=f"(_unpack_fp4_0[4]), "=f"(_unpack_fp4_0[5]), "=f"(_unpack_fp4_0[6]), "=f"(_unpack_fp4_0[7]) : "r"(q_codes0[0]));
            #endif
        }
        {
            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
            uint32_t _fp4_decode_pairs[4];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_codes0[1]));
            (&_unpack_fp4_0[8])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
            (&_unpack_fp4_0[8])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
            (&_unpack_fp4_0[10])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
            (&_unpack_fp4_0[10])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
            (&_unpack_fp4_0[12])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
            (&_unpack_fp4_0[12])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
            (&_unpack_fp4_0[14])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
            (&_unpack_fp4_0[14])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
            #else
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_0[8]), "=f"(_unpack_fp4_0[9]), "=f"(_unpack_fp4_0[10]), "=f"(_unpack_fp4_0[11]), "=f"(_unpack_fp4_0[12]), "=f"(_unpack_fp4_0[13]), "=f"(_unpack_fp4_0[14]), "=f"(_unpack_fp4_0[15]) : "r"(q_codes0[1]));
            #endif
        }
        x[0] = __fmaf_rn(-_unpack_fp4_0[0], _fp8_decode_0, x[0]);
        x[1] = __fmaf_rn(-_unpack_fp4_0[1], _fp8_decode_0, x[1]);
        x[2] = __fmaf_rn(-_unpack_fp4_0[2], _fp8_decode_0, x[2]);
        x[3] = __fmaf_rn(-_unpack_fp4_0[3], _fp8_decode_0, x[3]);
        x[4] = __fmaf_rn(-_unpack_fp4_0[4], _fp8_decode_0, x[4]);
        x[5] = __fmaf_rn(-_unpack_fp4_0[5], _fp8_decode_0, x[5]);
        x[6] = __fmaf_rn(-_unpack_fp4_0[6], _fp8_decode_0, x[6]);
        x[7] = __fmaf_rn(-_unpack_fp4_0[7], _fp8_decode_0, x[7]);
        x[8] = __fmaf_rn(-_unpack_fp4_0[8], _fp8_decode_0, x[8]);
        x[9] = __fmaf_rn(-_unpack_fp4_0[9], _fp8_decode_0, x[9]);
        x[10] = __fmaf_rn(-_unpack_fp4_0[10], _fp8_decode_0, x[10]);
        x[11] = __fmaf_rn(-_unpack_fp4_0[11], _fp8_decode_0, x[11]);
        x[12] = __fmaf_rn(-_unpack_fp4_0[12], _fp8_decode_0, x[12]);
        x[13] = __fmaf_rn(-_unpack_fp4_0[13], _fp8_decode_0, x[13]);
        x[14] = __fmaf_rn(-_unpack_fp4_0[14], _fp8_decode_0, x[14]);
        x[15] = __fmaf_rn(-_unpack_fp4_0[15], _fp8_decode_0, x[15]);
        float _max_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
        uint16_t _e4m3x2_10;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_10) : "f"(0.0f), "f"(_max_1));
        uint8_t _cvt_satfinite_e4m3_f32_9 = (uint8_t)_e4m3x2_10;
        uint32_t _f16x2_decode_11;
        float _fp8_e4m3_decode_8;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_11) : "h"(_e4m3x2_10));
        uint16_t _f16_decode_11 = (uint16_t)_f16x2_decode_11;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_8) : "h"(_f16_decode_11));
        float _fdiv_rn_1 = __fdiv_rn(1.0f, _fp8_e4m3_decode_8);
        x[0] = x[0] * _fdiv_rn_1;
        x[1] = x[1] * _fdiv_rn_1;
        x[2] = x[2] * _fdiv_rn_1;
        x[3] = x[3] * _fdiv_rn_1;
        x[4] = x[4] * _fdiv_rn_1;
        x[5] = x[5] * _fdiv_rn_1;
        x[6] = x[6] * _fdiv_rn_1;
        x[7] = x[7] * _fdiv_rn_1;
        x[8] = x[8] * _fdiv_rn_1;
        x[9] = x[9] * _fdiv_rn_1;
        x[10] = x[10] * _fdiv_rn_1;
        x[11] = x[11] * _fdiv_rn_1;
        x[12] = x[12] * _fdiv_rn_1;
        x[13] = x[13] * _fdiv_rn_1;
        x[14] = x[14] * _fdiv_rn_1;
        x[15] = x[15] * _fdiv_rn_1;
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes_l0[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes_l0[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
        sf_tmp[0] = _max_1;
        q_sf_l[0] = sf_tmp[0];
        unsigned int q_addr1 = q_stage_addr + (unsigned int)(((vrow + 32) * 128 + group * 16) * 2);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[(0) + 3]))
            : "r"(q_addr1));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[(0) + 3]))
            : "r"(q_addr1 + 16));
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&qv[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qv_lo[_pair]) << 16);
            (&qv[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qv_lo[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&qv[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qv_hi[_pair]) << 16);
            (&qv[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qv_hi[_pair]) & 0xffff0000u);
        }
        x[0] = ((n_real > vrow + 32) ? qv[0] - qm_slice[0] : 0.0f);
        x[1] = ((n_real > vrow + 32) ? qv[1] - qm_slice[1] : 0.0f);
        x[2] = ((n_real > vrow + 32) ? qv[2] - qm_slice[2] : 0.0f);
        x[3] = ((n_real > vrow + 32) ? qv[3] - qm_slice[3] : 0.0f);
        x[4] = ((n_real > vrow + 32) ? qv[4] - qm_slice[4] : 0.0f);
        x[5] = ((n_real > vrow + 32) ? qv[5] - qm_slice[5] : 0.0f);
        x[6] = ((n_real > vrow + 32) ? qv[6] - qm_slice[6] : 0.0f);
        x[7] = ((n_real > vrow + 32) ? qv[7] - qm_slice[7] : 0.0f);
        x[8] = ((n_real > vrow + 32) ? qv[8] - qm_slice[8] : 0.0f);
        x[9] = ((n_real > vrow + 32) ? qv[9] - qm_slice[9] : 0.0f);
        x[10] = ((n_real > vrow + 32) ? qv[10] - qm_slice[10] : 0.0f);
        x[11] = ((n_real > vrow + 32) ? qv[11] - qm_slice[11] : 0.0f);
        x[12] = ((n_real > vrow + 32) ? qv[12] - qm_slice[12] : 0.0f);
        x[13] = ((n_real > vrow + 32) ? qv[13] - qm_slice[13] : 0.0f);
        x[14] = ((n_real > vrow + 32) ? qv[14] - qm_slice[14] : 0.0f);
        x[15] = ((n_real > vrow + 32) ? qv[15] - qm_slice[15] : 0.0f);
        float _max_2 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
        uint16_t _e4m3x2_14;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_14) : "f"(0.0f), "f"(_max_2));
        uint8_t _cvt_satfinite_e4m3_f32_13 = (uint8_t)_e4m3x2_14;
        uint32_t _f16x2_decode_15;
        float _fp8_e4m3_decode_12;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_15) : "h"(_e4m3x2_14));
        uint16_t _f16_decode_15 = (uint16_t)_f16x2_decode_15;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_12) : "h"(_f16_decode_15));
        float _fdiv_rn_2 = __fdiv_rn(1.0f, _fp8_e4m3_decode_12);
        x[0] = x[0] * _fdiv_rn_2;
        x[1] = x[1] * _fdiv_rn_2;
        x[2] = x[2] * _fdiv_rn_2;
        x[3] = x[3] * _fdiv_rn_2;
        x[4] = x[4] * _fdiv_rn_2;
        x[5] = x[5] * _fdiv_rn_2;
        x[6] = x[6] * _fdiv_rn_2;
        x[7] = x[7] * _fdiv_rn_2;
        x[8] = x[8] * _fdiv_rn_2;
        x[9] = x[9] * _fdiv_rn_2;
        x[10] = x[10] * _fdiv_rn_2;
        x[11] = x[11] * _fdiv_rn_2;
        x[12] = x[12] * _fdiv_rn_2;
        x[13] = x[13] * _fdiv_rn_2;
        x[14] = x[14] * _fdiv_rn_2;
        x[15] = x[15] * _fdiv_rn_2;
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes1[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes1[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
        sf_tmp[0] = _max_2;
        q_sf[1] = sf_tmp[0];
        x[0] = ((n_real > vrow + 32) ? qv[0] - qm_slice[0] : 0.0f);
        x[1] = ((n_real > vrow + 32) ? qv[1] - qm_slice[1] : 0.0f);
        x[2] = ((n_real > vrow + 32) ? qv[2] - qm_slice[2] : 0.0f);
        x[3] = ((n_real > vrow + 32) ? qv[3] - qm_slice[3] : 0.0f);
        x[4] = ((n_real > vrow + 32) ? qv[4] - qm_slice[4] : 0.0f);
        x[5] = ((n_real > vrow + 32) ? qv[5] - qm_slice[5] : 0.0f);
        x[6] = ((n_real > vrow + 32) ? qv[6] - qm_slice[6] : 0.0f);
        x[7] = ((n_real > vrow + 32) ? qv[7] - qm_slice[7] : 0.0f);
        x[8] = ((n_real > vrow + 32) ? qv[8] - qm_slice[8] : 0.0f);
        x[9] = ((n_real > vrow + 32) ? qv[9] - qm_slice[9] : 0.0f);
        x[10] = ((n_real > vrow + 32) ? qv[10] - qm_slice[10] : 0.0f);
        x[11] = ((n_real > vrow + 32) ? qv[11] - qm_slice[11] : 0.0f);
        x[12] = ((n_real > vrow + 32) ? qv[12] - qm_slice[12] : 0.0f);
        x[13] = ((n_real > vrow + 32) ? qv[13] - qm_slice[13] : 0.0f);
        x[14] = ((n_real > vrow + 32) ? qv[14] - qm_slice[14] : 0.0f);
        x[15] = ((n_real > vrow + 32) ? qv[15] - qm_slice[15] : 0.0f);
        uint16_t _e4m3x2_17;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_17) : "f"(0.0f), "f"(sf_tmp[0]));
        uint8_t _cvt_satfinite_e4m3_f32_16 = (uint8_t)_e4m3x2_17;
        uint32_t _f16x2_decode_18;
        float _fp8_decode_1;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_18) : "h"(_e4m3x2_17));
        uint16_t _f16_decode_18 = (uint16_t)_f16x2_decode_18;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_1) : "h"(_f16_decode_18));
        float _unpack_fp4_1[16];
        {
            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
            uint32_t _fp4_decode_pairs[4];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_codes1[0]));
            (&_unpack_fp4_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
            (&_unpack_fp4_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
            (&_unpack_fp4_1[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
            (&_unpack_fp4_1[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
            (&_unpack_fp4_1[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
            (&_unpack_fp4_1[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
            (&_unpack_fp4_1[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
            (&_unpack_fp4_1[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
            #else
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_1[0]), "=f"(_unpack_fp4_1[1]), "=f"(_unpack_fp4_1[2]), "=f"(_unpack_fp4_1[3]), "=f"(_unpack_fp4_1[4]), "=f"(_unpack_fp4_1[5]), "=f"(_unpack_fp4_1[6]), "=f"(_unpack_fp4_1[7]) : "r"(q_codes1[0]));
            #endif
        }
        {
            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
            uint32_t _fp4_decode_pairs[4];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_codes1[1]));
            (&_unpack_fp4_1[8])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
            (&_unpack_fp4_1[8])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
            (&_unpack_fp4_1[10])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
            (&_unpack_fp4_1[10])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
            (&_unpack_fp4_1[12])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
            (&_unpack_fp4_1[12])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
            (&_unpack_fp4_1[14])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
            (&_unpack_fp4_1[14])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
            #else
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_1[8]), "=f"(_unpack_fp4_1[9]), "=f"(_unpack_fp4_1[10]), "=f"(_unpack_fp4_1[11]), "=f"(_unpack_fp4_1[12]), "=f"(_unpack_fp4_1[13]), "=f"(_unpack_fp4_1[14]), "=f"(_unpack_fp4_1[15]) : "r"(q_codes1[1]));
            #endif
        }
        x[0] = __fmaf_rn(-_unpack_fp4_1[0], _fp8_decode_1, x[0]);
        x[1] = __fmaf_rn(-_unpack_fp4_1[1], _fp8_decode_1, x[1]);
        x[2] = __fmaf_rn(-_unpack_fp4_1[2], _fp8_decode_1, x[2]);
        x[3] = __fmaf_rn(-_unpack_fp4_1[3], _fp8_decode_1, x[3]);
        x[4] = __fmaf_rn(-_unpack_fp4_1[4], _fp8_decode_1, x[4]);
        x[5] = __fmaf_rn(-_unpack_fp4_1[5], _fp8_decode_1, x[5]);
        x[6] = __fmaf_rn(-_unpack_fp4_1[6], _fp8_decode_1, x[6]);
        x[7] = __fmaf_rn(-_unpack_fp4_1[7], _fp8_decode_1, x[7]);
        x[8] = __fmaf_rn(-_unpack_fp4_1[8], _fp8_decode_1, x[8]);
        x[9] = __fmaf_rn(-_unpack_fp4_1[9], _fp8_decode_1, x[9]);
        x[10] = __fmaf_rn(-_unpack_fp4_1[10], _fp8_decode_1, x[10]);
        x[11] = __fmaf_rn(-_unpack_fp4_1[11], _fp8_decode_1, x[11]);
        x[12] = __fmaf_rn(-_unpack_fp4_1[12], _fp8_decode_1, x[12]);
        x[13] = __fmaf_rn(-_unpack_fp4_1[13], _fp8_decode_1, x[13]);
        x[14] = __fmaf_rn(-_unpack_fp4_1[14], _fp8_decode_1, x[14]);
        x[15] = __fmaf_rn(-_unpack_fp4_1[15], _fp8_decode_1, x[15]);
        float _max_3 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
        uint16_t _e4m3x2_21;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_21) : "f"(0.0f), "f"(_max_3));
        uint8_t _cvt_satfinite_e4m3_f32_20 = (uint8_t)_e4m3x2_21;
        uint32_t _f16x2_decode_22;
        float _fp8_e4m3_decode_19;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_22) : "h"(_e4m3x2_21));
        uint16_t _f16_decode_22 = (uint16_t)_f16x2_decode_22;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_19) : "h"(_f16_decode_22));
        float _fdiv_rn_3 = __fdiv_rn(1.0f, _fp8_e4m3_decode_19);
        x[0] = x[0] * _fdiv_rn_3;
        x[1] = x[1] * _fdiv_rn_3;
        x[2] = x[2] * _fdiv_rn_3;
        x[3] = x[3] * _fdiv_rn_3;
        x[4] = x[4] * _fdiv_rn_3;
        x[5] = x[5] * _fdiv_rn_3;
        x[6] = x[6] * _fdiv_rn_3;
        x[7] = x[7] * _fdiv_rn_3;
        x[8] = x[8] * _fdiv_rn_3;
        x[9] = x[9] * _fdiv_rn_3;
        x[10] = x[10] * _fdiv_rn_3;
        x[11] = x[11] * _fdiv_rn_3;
        x[12] = x[12] * _fdiv_rn_3;
        x[13] = x[13] * _fdiv_rn_3;
        x[14] = x[14] * _fdiv_rn_3;
        x[15] = x[15] * _fdiv_rn_3;
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes_l1[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes_l1[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
        sf_tmp[0] = _max_3;
        q_sf_l[1] = sf_tmp[0];
        unsigned int q_addr2 = q_stage_addr + (unsigned int)(((vrow + 64) * 128 + group * 16) * 2);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[(0) + 3]))
            : "r"(q_addr2));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[(0) + 3]))
            : "r"(q_addr2 + 16));
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&qv[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qv_lo[_pair]) << 16);
            (&qv[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qv_lo[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&qv[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qv_hi[_pair]) << 16);
            (&qv[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qv_hi[_pair]) & 0xffff0000u);
        }
        x[0] = ((n_real > vrow + 64) ? qv[0] - qm_slice[0] : 0.0f);
        x[1] = ((n_real > vrow + 64) ? qv[1] - qm_slice[1] : 0.0f);
        x[2] = ((n_real > vrow + 64) ? qv[2] - qm_slice[2] : 0.0f);
        x[3] = ((n_real > vrow + 64) ? qv[3] - qm_slice[3] : 0.0f);
        x[4] = ((n_real > vrow + 64) ? qv[4] - qm_slice[4] : 0.0f);
        x[5] = ((n_real > vrow + 64) ? qv[5] - qm_slice[5] : 0.0f);
        x[6] = ((n_real > vrow + 64) ? qv[6] - qm_slice[6] : 0.0f);
        x[7] = ((n_real > vrow + 64) ? qv[7] - qm_slice[7] : 0.0f);
        x[8] = ((n_real > vrow + 64) ? qv[8] - qm_slice[8] : 0.0f);
        x[9] = ((n_real > vrow + 64) ? qv[9] - qm_slice[9] : 0.0f);
        x[10] = ((n_real > vrow + 64) ? qv[10] - qm_slice[10] : 0.0f);
        x[11] = ((n_real > vrow + 64) ? qv[11] - qm_slice[11] : 0.0f);
        x[12] = ((n_real > vrow + 64) ? qv[12] - qm_slice[12] : 0.0f);
        x[13] = ((n_real > vrow + 64) ? qv[13] - qm_slice[13] : 0.0f);
        x[14] = ((n_real > vrow + 64) ? qv[14] - qm_slice[14] : 0.0f);
        x[15] = ((n_real > vrow + 64) ? qv[15] - qm_slice[15] : 0.0f);
        float _max_4 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
        uint16_t _e4m3x2_25;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_25) : "f"(0.0f), "f"(_max_4));
        uint8_t _cvt_satfinite_e4m3_f32_24 = (uint8_t)_e4m3x2_25;
        uint32_t _f16x2_decode_26;
        float _fp8_e4m3_decode_23;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_26) : "h"(_e4m3x2_25));
        uint16_t _f16_decode_26 = (uint16_t)_f16x2_decode_26;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_23) : "h"(_f16_decode_26));
        float _fdiv_rn_4 = __fdiv_rn(1.0f, _fp8_e4m3_decode_23);
        x[0] = x[0] * _fdiv_rn_4;
        x[1] = x[1] * _fdiv_rn_4;
        x[2] = x[2] * _fdiv_rn_4;
        x[3] = x[3] * _fdiv_rn_4;
        x[4] = x[4] * _fdiv_rn_4;
        x[5] = x[5] * _fdiv_rn_4;
        x[6] = x[6] * _fdiv_rn_4;
        x[7] = x[7] * _fdiv_rn_4;
        x[8] = x[8] * _fdiv_rn_4;
        x[9] = x[9] * _fdiv_rn_4;
        x[10] = x[10] * _fdiv_rn_4;
        x[11] = x[11] * _fdiv_rn_4;
        x[12] = x[12] * _fdiv_rn_4;
        x[13] = x[13] * _fdiv_rn_4;
        x[14] = x[14] * _fdiv_rn_4;
        x[15] = x[15] * _fdiv_rn_4;
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes2[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes2[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
        sf_tmp[0] = _max_4;
        q_sf[2] = sf_tmp[0];
        x[0] = ((n_real > vrow + 64) ? qv[0] - qm_slice[0] : 0.0f);
        x[1] = ((n_real > vrow + 64) ? qv[1] - qm_slice[1] : 0.0f);
        x[2] = ((n_real > vrow + 64) ? qv[2] - qm_slice[2] : 0.0f);
        x[3] = ((n_real > vrow + 64) ? qv[3] - qm_slice[3] : 0.0f);
        x[4] = ((n_real > vrow + 64) ? qv[4] - qm_slice[4] : 0.0f);
        x[5] = ((n_real > vrow + 64) ? qv[5] - qm_slice[5] : 0.0f);
        x[6] = ((n_real > vrow + 64) ? qv[6] - qm_slice[6] : 0.0f);
        x[7] = ((n_real > vrow + 64) ? qv[7] - qm_slice[7] : 0.0f);
        x[8] = ((n_real > vrow + 64) ? qv[8] - qm_slice[8] : 0.0f);
        x[9] = ((n_real > vrow + 64) ? qv[9] - qm_slice[9] : 0.0f);
        x[10] = ((n_real > vrow + 64) ? qv[10] - qm_slice[10] : 0.0f);
        x[11] = ((n_real > vrow + 64) ? qv[11] - qm_slice[11] : 0.0f);
        x[12] = ((n_real > vrow + 64) ? qv[12] - qm_slice[12] : 0.0f);
        x[13] = ((n_real > vrow + 64) ? qv[13] - qm_slice[13] : 0.0f);
        x[14] = ((n_real > vrow + 64) ? qv[14] - qm_slice[14] : 0.0f);
        x[15] = ((n_real > vrow + 64) ? qv[15] - qm_slice[15] : 0.0f);
        uint16_t _e4m3x2_28;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_28) : "f"(0.0f), "f"(sf_tmp[0]));
        uint8_t _cvt_satfinite_e4m3_f32_27 = (uint8_t)_e4m3x2_28;
        uint32_t _f16x2_decode_29;
        float _fp8_decode_2;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_29) : "h"(_e4m3x2_28));
        uint16_t _f16_decode_29 = (uint16_t)_f16x2_decode_29;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_2) : "h"(_f16_decode_29));
        float _unpack_fp4_2[16];
        {
            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
            uint32_t _fp4_decode_pairs[4];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_codes2[0]));
            (&_unpack_fp4_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
            (&_unpack_fp4_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
            (&_unpack_fp4_2[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
            (&_unpack_fp4_2[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
            (&_unpack_fp4_2[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
            (&_unpack_fp4_2[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
            (&_unpack_fp4_2[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
            (&_unpack_fp4_2[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
            #else
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_2[0]), "=f"(_unpack_fp4_2[1]), "=f"(_unpack_fp4_2[2]), "=f"(_unpack_fp4_2[3]), "=f"(_unpack_fp4_2[4]), "=f"(_unpack_fp4_2[5]), "=f"(_unpack_fp4_2[6]), "=f"(_unpack_fp4_2[7]) : "r"(q_codes2[0]));
            #endif
        }
        {
            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
            uint32_t _fp4_decode_pairs[4];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_codes2[1]));
            (&_unpack_fp4_2[8])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
            (&_unpack_fp4_2[8])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
            (&_unpack_fp4_2[10])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
            (&_unpack_fp4_2[10])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
            (&_unpack_fp4_2[12])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
            (&_unpack_fp4_2[12])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
            (&_unpack_fp4_2[14])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
            (&_unpack_fp4_2[14])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
            #else
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_2[8]), "=f"(_unpack_fp4_2[9]), "=f"(_unpack_fp4_2[10]), "=f"(_unpack_fp4_2[11]), "=f"(_unpack_fp4_2[12]), "=f"(_unpack_fp4_2[13]), "=f"(_unpack_fp4_2[14]), "=f"(_unpack_fp4_2[15]) : "r"(q_codes2[1]));
            #endif
        }
        x[0] = __fmaf_rn(-_unpack_fp4_2[0], _fp8_decode_2, x[0]);
        x[1] = __fmaf_rn(-_unpack_fp4_2[1], _fp8_decode_2, x[1]);
        x[2] = __fmaf_rn(-_unpack_fp4_2[2], _fp8_decode_2, x[2]);
        x[3] = __fmaf_rn(-_unpack_fp4_2[3], _fp8_decode_2, x[3]);
        x[4] = __fmaf_rn(-_unpack_fp4_2[4], _fp8_decode_2, x[4]);
        x[5] = __fmaf_rn(-_unpack_fp4_2[5], _fp8_decode_2, x[5]);
        x[6] = __fmaf_rn(-_unpack_fp4_2[6], _fp8_decode_2, x[6]);
        x[7] = __fmaf_rn(-_unpack_fp4_2[7], _fp8_decode_2, x[7]);
        x[8] = __fmaf_rn(-_unpack_fp4_2[8], _fp8_decode_2, x[8]);
        x[9] = __fmaf_rn(-_unpack_fp4_2[9], _fp8_decode_2, x[9]);
        x[10] = __fmaf_rn(-_unpack_fp4_2[10], _fp8_decode_2, x[10]);
        x[11] = __fmaf_rn(-_unpack_fp4_2[11], _fp8_decode_2, x[11]);
        x[12] = __fmaf_rn(-_unpack_fp4_2[12], _fp8_decode_2, x[12]);
        x[13] = __fmaf_rn(-_unpack_fp4_2[13], _fp8_decode_2, x[13]);
        x[14] = __fmaf_rn(-_unpack_fp4_2[14], _fp8_decode_2, x[14]);
        x[15] = __fmaf_rn(-_unpack_fp4_2[15], _fp8_decode_2, x[15]);
        float _max_5 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
        uint16_t _e4m3x2_32;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_32) : "f"(0.0f), "f"(_max_5));
        uint8_t _cvt_satfinite_e4m3_f32_31 = (uint8_t)_e4m3x2_32;
        uint32_t _f16x2_decode_33;
        float _fp8_e4m3_decode_30;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_33) : "h"(_e4m3x2_32));
        uint16_t _f16_decode_33 = (uint16_t)_f16x2_decode_33;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_30) : "h"(_f16_decode_33));
        float _fdiv_rn_5 = __fdiv_rn(1.0f, _fp8_e4m3_decode_30);
        x[0] = x[0] * _fdiv_rn_5;
        x[1] = x[1] * _fdiv_rn_5;
        x[2] = x[2] * _fdiv_rn_5;
        x[3] = x[3] * _fdiv_rn_5;
        x[4] = x[4] * _fdiv_rn_5;
        x[5] = x[5] * _fdiv_rn_5;
        x[6] = x[6] * _fdiv_rn_5;
        x[7] = x[7] * _fdiv_rn_5;
        x[8] = x[8] * _fdiv_rn_5;
        x[9] = x[9] * _fdiv_rn_5;
        x[10] = x[10] * _fdiv_rn_5;
        x[11] = x[11] * _fdiv_rn_5;
        x[12] = x[12] * _fdiv_rn_5;
        x[13] = x[13] * _fdiv_rn_5;
        x[14] = x[14] * _fdiv_rn_5;
        x[15] = x[15] * _fdiv_rn_5;
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes_l2[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes_l2[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
        sf_tmp[0] = _max_5;
        q_sf_l[2] = sf_tmp[0];
        unsigned int q_addr3 = q_stage_addr + (unsigned int)(((vrow + 96) * 128 + group * 16) * 2);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qv_lo[(0) + 3]))
            : "r"(q_addr3));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qv_hi[(0) + 3]))
            : "r"(q_addr3 + 16));
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&qv[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qv_lo[_pair]) << 16);
            (&qv[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qv_lo[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&qv[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qv_hi[_pair]) << 16);
            (&qv[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qv_hi[_pair]) & 0xffff0000u);
        }
        x[0] = ((n_real > vrow + 96) ? qv[0] - qm_slice[0] : 0.0f);
        x[1] = ((n_real > vrow + 96) ? qv[1] - qm_slice[1] : 0.0f);
        x[2] = ((n_real > vrow + 96) ? qv[2] - qm_slice[2] : 0.0f);
        x[3] = ((n_real > vrow + 96) ? qv[3] - qm_slice[3] : 0.0f);
        x[4] = ((n_real > vrow + 96) ? qv[4] - qm_slice[4] : 0.0f);
        x[5] = ((n_real > vrow + 96) ? qv[5] - qm_slice[5] : 0.0f);
        x[6] = ((n_real > vrow + 96) ? qv[6] - qm_slice[6] : 0.0f);
        x[7] = ((n_real > vrow + 96) ? qv[7] - qm_slice[7] : 0.0f);
        x[8] = ((n_real > vrow + 96) ? qv[8] - qm_slice[8] : 0.0f);
        x[9] = ((n_real > vrow + 96) ? qv[9] - qm_slice[9] : 0.0f);
        x[10] = ((n_real > vrow + 96) ? qv[10] - qm_slice[10] : 0.0f);
        x[11] = ((n_real > vrow + 96) ? qv[11] - qm_slice[11] : 0.0f);
        x[12] = ((n_real > vrow + 96) ? qv[12] - qm_slice[12] : 0.0f);
        x[13] = ((n_real > vrow + 96) ? qv[13] - qm_slice[13] : 0.0f);
        x[14] = ((n_real > vrow + 96) ? qv[14] - qm_slice[14] : 0.0f);
        x[15] = ((n_real > vrow + 96) ? qv[15] - qm_slice[15] : 0.0f);
        float _max_6 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
        uint16_t _e4m3x2_36;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_36) : "f"(0.0f), "f"(_max_6));
        uint8_t _cvt_satfinite_e4m3_f32_35 = (uint8_t)_e4m3x2_36;
        uint32_t _f16x2_decode_37;
        float _fp8_e4m3_decode_34;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_37) : "h"(_e4m3x2_36));
        uint16_t _f16_decode_37 = (uint16_t)_f16x2_decode_37;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_34) : "h"(_f16_decode_37));
        float _fdiv_rn_6 = __fdiv_rn(1.0f, _fp8_e4m3_decode_34);
        x[0] = x[0] * _fdiv_rn_6;
        x[1] = x[1] * _fdiv_rn_6;
        x[2] = x[2] * _fdiv_rn_6;
        x[3] = x[3] * _fdiv_rn_6;
        x[4] = x[4] * _fdiv_rn_6;
        x[5] = x[5] * _fdiv_rn_6;
        x[6] = x[6] * _fdiv_rn_6;
        x[7] = x[7] * _fdiv_rn_6;
        x[8] = x[8] * _fdiv_rn_6;
        x[9] = x[9] * _fdiv_rn_6;
        x[10] = x[10] * _fdiv_rn_6;
        x[11] = x[11] * _fdiv_rn_6;
        x[12] = x[12] * _fdiv_rn_6;
        x[13] = x[13] * _fdiv_rn_6;
        x[14] = x[14] * _fdiv_rn_6;
        x[15] = x[15] * _fdiv_rn_6;
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes3[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes3[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
        sf_tmp[0] = _max_6;
        q_sf[3] = sf_tmp[0];
        x[0] = ((n_real > vrow + 96) ? qv[0] - qm_slice[0] : 0.0f);
        x[1] = ((n_real > vrow + 96) ? qv[1] - qm_slice[1] : 0.0f);
        x[2] = ((n_real > vrow + 96) ? qv[2] - qm_slice[2] : 0.0f);
        x[3] = ((n_real > vrow + 96) ? qv[3] - qm_slice[3] : 0.0f);
        x[4] = ((n_real > vrow + 96) ? qv[4] - qm_slice[4] : 0.0f);
        x[5] = ((n_real > vrow + 96) ? qv[5] - qm_slice[5] : 0.0f);
        x[6] = ((n_real > vrow + 96) ? qv[6] - qm_slice[6] : 0.0f);
        x[7] = ((n_real > vrow + 96) ? qv[7] - qm_slice[7] : 0.0f);
        x[8] = ((n_real > vrow + 96) ? qv[8] - qm_slice[8] : 0.0f);
        x[9] = ((n_real > vrow + 96) ? qv[9] - qm_slice[9] : 0.0f);
        x[10] = ((n_real > vrow + 96) ? qv[10] - qm_slice[10] : 0.0f);
        x[11] = ((n_real > vrow + 96) ? qv[11] - qm_slice[11] : 0.0f);
        x[12] = ((n_real > vrow + 96) ? qv[12] - qm_slice[12] : 0.0f);
        x[13] = ((n_real > vrow + 96) ? qv[13] - qm_slice[13] : 0.0f);
        x[14] = ((n_real > vrow + 96) ? qv[14] - qm_slice[14] : 0.0f);
        x[15] = ((n_real > vrow + 96) ? qv[15] - qm_slice[15] : 0.0f);
        uint16_t _e4m3x2_39;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_39) : "f"(0.0f), "f"(sf_tmp[0]));
        uint8_t _cvt_satfinite_e4m3_f32_38 = (uint8_t)_e4m3x2_39;
        uint32_t _f16x2_decode_40;
        float _fp8_decode_3;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_40) : "h"(_e4m3x2_39));
        uint16_t _f16_decode_40 = (uint16_t)_f16x2_decode_40;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_3) : "h"(_f16_decode_40));
        float _unpack_fp4_3[16];
        {
            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
            uint32_t _fp4_decode_pairs[4];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_codes3[0]));
            (&_unpack_fp4_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
            (&_unpack_fp4_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
            (&_unpack_fp4_3[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
            (&_unpack_fp4_3[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
            (&_unpack_fp4_3[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
            (&_unpack_fp4_3[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
            (&_unpack_fp4_3[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
            (&_unpack_fp4_3[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
            #else
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_3[0]), "=f"(_unpack_fp4_3[1]), "=f"(_unpack_fp4_3[2]), "=f"(_unpack_fp4_3[3]), "=f"(_unpack_fp4_3[4]), "=f"(_unpack_fp4_3[5]), "=f"(_unpack_fp4_3[6]), "=f"(_unpack_fp4_3[7]) : "r"(q_codes3[0]));
            #endif
        }
        {
            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
            uint32_t _fp4_decode_pairs[4];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_codes3[1]));
            (&_unpack_fp4_3[8])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
            (&_unpack_fp4_3[8])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
            (&_unpack_fp4_3[10])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
            (&_unpack_fp4_3[10])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
            (&_unpack_fp4_3[12])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
            (&_unpack_fp4_3[12])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
            (&_unpack_fp4_3[14])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
            (&_unpack_fp4_3[14])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
            #else
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_3[8]), "=f"(_unpack_fp4_3[9]), "=f"(_unpack_fp4_3[10]), "=f"(_unpack_fp4_3[11]), "=f"(_unpack_fp4_3[12]), "=f"(_unpack_fp4_3[13]), "=f"(_unpack_fp4_3[14]), "=f"(_unpack_fp4_3[15]) : "r"(q_codes3[1]));
            #endif
        }
        x[0] = __fmaf_rn(-_unpack_fp4_3[0], _fp8_decode_3, x[0]);
        x[1] = __fmaf_rn(-_unpack_fp4_3[1], _fp8_decode_3, x[1]);
        x[2] = __fmaf_rn(-_unpack_fp4_3[2], _fp8_decode_3, x[2]);
        x[3] = __fmaf_rn(-_unpack_fp4_3[3], _fp8_decode_3, x[3]);
        x[4] = __fmaf_rn(-_unpack_fp4_3[4], _fp8_decode_3, x[4]);
        x[5] = __fmaf_rn(-_unpack_fp4_3[5], _fp8_decode_3, x[5]);
        x[6] = __fmaf_rn(-_unpack_fp4_3[6], _fp8_decode_3, x[6]);
        x[7] = __fmaf_rn(-_unpack_fp4_3[7], _fp8_decode_3, x[7]);
        x[8] = __fmaf_rn(-_unpack_fp4_3[8], _fp8_decode_3, x[8]);
        x[9] = __fmaf_rn(-_unpack_fp4_3[9], _fp8_decode_3, x[9]);
        x[10] = __fmaf_rn(-_unpack_fp4_3[10], _fp8_decode_3, x[10]);
        x[11] = __fmaf_rn(-_unpack_fp4_3[11], _fp8_decode_3, x[11]);
        x[12] = __fmaf_rn(-_unpack_fp4_3[12], _fp8_decode_3, x[12]);
        x[13] = __fmaf_rn(-_unpack_fp4_3[13], _fp8_decode_3, x[13]);
        x[14] = __fmaf_rn(-_unpack_fp4_3[14], _fp8_decode_3, x[14]);
        x[15] = __fmaf_rn(-_unpack_fp4_3[15], _fp8_decode_3, x[15]);
        float _max_7 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
        uint16_t _e4m3x2_43;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_43) : "f"(0.0f), "f"(_max_7));
        uint8_t _cvt_satfinite_e4m3_f32_42 = (uint8_t)_e4m3x2_43;
        uint32_t _f16x2_decode_44;
        float _fp8_e4m3_decode_41;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_44) : "h"(_e4m3x2_43));
        uint16_t _f16_decode_44 = (uint16_t)_f16x2_decode_44;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_41) : "h"(_f16_decode_44));
        float _fdiv_rn_7 = __fdiv_rn(1.0f, _fp8_e4m3_decode_41);
        x[0] = x[0] * _fdiv_rn_7;
        x[1] = x[1] * _fdiv_rn_7;
        x[2] = x[2] * _fdiv_rn_7;
        x[3] = x[3] * _fdiv_rn_7;
        x[4] = x[4] * _fdiv_rn_7;
        x[5] = x[5] * _fdiv_rn_7;
        x[6] = x[6] * _fdiv_rn_7;
        x[7] = x[7] * _fdiv_rn_7;
        x[8] = x[8] * _fdiv_rn_7;
        x[9] = x[9] * _fdiv_rn_7;
        x[10] = x[10] * _fdiv_rn_7;
        x[11] = x[11] * _fdiv_rn_7;
        x[12] = x[12] * _fdiv_rn_7;
        x[13] = x[13] * _fdiv_rn_7;
        x[14] = x[14] * _fdiv_rn_7;
        x[15] = x[15] * _fdiv_rn_7;
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes_l3[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(q_codes_l3[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
        sf_tmp[0] = _max_7;
        q_sf_l[3] = sf_tmp[0];
        __syncthreads();
        asm volatile("griddepcontrol.wait;" ::: "memory");
        long long q_off0 = (q_row0 + (long long)vrow) * 64 + (long long)(group * 8);
        *(reinterpret_cast<int*>(q4 + q_off0) + (0)) = q_codes0[0];
        *(reinterpret_cast<int*>(q4 + (q_off0 + 4)) + (0)) = q_codes0[1];
        uint16_t _cvt_satfinite_e4m3x2_f32_45;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_45) : "f"(0.0f), "f"(q_sf[0]));
        *reinterpret_cast<unsigned char*>(sfq + (q_tile + (long long)((((((vrow & 31) >> 3) * 2 + group_pair) * 8 + (vrow & 7)) * 4 + (vrow >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_45 & 127));
        *(reinterpret_cast<int*>(q4l + q_off0) + (0)) = q_codes_l0[0];
        *(reinterpret_cast<int*>(q4l + (q_off0 + 4)) + (0)) = q_codes_l0[1];
        uint16_t _cvt_satfinite_e4m3x2_f32_46;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_46) : "f"(0.0f), "f"(q_sf_l[0]));
        *reinterpret_cast<unsigned char*>(sfq_l + (q_tile + (long long)((((((vrow & 31) >> 3) * 2 + group_pair) * 8 + (vrow & 7)) * 4 + (vrow >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_46 & 127));
        long long q_off1 = (q_row0 + (long long)(vrow + 32)) * 64 + (long long)(group * 8);
        *(reinterpret_cast<int*>(q4 + q_off1) + (0)) = q_codes1[0];
        *(reinterpret_cast<int*>(q4 + (q_off1 + 4)) + (0)) = q_codes1[1];
        uint16_t _cvt_satfinite_e4m3x2_f32_47;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_47) : "f"(0.0f), "f"(q_sf[1]));
        *reinterpret_cast<unsigned char*>(sfq + (q_tile + (long long)((((((vrow + 32 & 31) >> 3) * 2 + group_pair) * 8 + (vrow + 32 & 7)) * 4 + (vrow + 32 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_47 & 127));
        *(reinterpret_cast<int*>(q4l + q_off1) + (0)) = q_codes_l1[0];
        *(reinterpret_cast<int*>(q4l + (q_off1 + 4)) + (0)) = q_codes_l1[1];
        uint16_t _cvt_satfinite_e4m3x2_f32_48;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_48) : "f"(0.0f), "f"(q_sf_l[1]));
        *reinterpret_cast<unsigned char*>(sfq_l + (q_tile + (long long)((((((vrow + 32 & 31) >> 3) * 2 + group_pair) * 8 + (vrow + 32 & 7)) * 4 + (vrow + 32 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_48 & 127));
        long long q_off2 = (q_row0 + (long long)(vrow + 64)) * 64 + (long long)(group * 8);
        *(reinterpret_cast<int*>(q4 + q_off2) + (0)) = q_codes2[0];
        *(reinterpret_cast<int*>(q4 + (q_off2 + 4)) + (0)) = q_codes2[1];
        uint16_t _cvt_satfinite_e4m3x2_f32_49;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_49) : "f"(0.0f), "f"(q_sf[2]));
        *reinterpret_cast<unsigned char*>(sfq + (q_tile + (long long)((((((vrow + 64 & 31) >> 3) * 2 + group_pair) * 8 + (vrow + 64 & 7)) * 4 + (vrow + 64 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_49 & 127));
        *(reinterpret_cast<int*>(q4l + q_off2) + (0)) = q_codes_l2[0];
        *(reinterpret_cast<int*>(q4l + (q_off2 + 4)) + (0)) = q_codes_l2[1];
        uint16_t _cvt_satfinite_e4m3x2_f32_50;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_50) : "f"(0.0f), "f"(q_sf_l[2]));
        *reinterpret_cast<unsigned char*>(sfq_l + (q_tile + (long long)((((((vrow + 64 & 31) >> 3) * 2 + group_pair) * 8 + (vrow + 64 & 7)) * 4 + (vrow + 64 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_50 & 127));
        long long q_off3 = (q_row0 + (long long)(vrow + 96)) * 64 + (long long)(group * 8);
        *(reinterpret_cast<int*>(q4 + q_off3) + (0)) = q_codes3[0];
        *(reinterpret_cast<int*>(q4 + (q_off3 + 4)) + (0)) = q_codes3[1];
        uint16_t _cvt_satfinite_e4m3x2_f32_51;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_51) : "f"(0.0f), "f"(q_sf[3]));
        *reinterpret_cast<unsigned char*>(sfq + (q_tile + (long long)((((((vrow + 96 & 31) >> 3) * 2 + group_pair) * 8 + (vrow + 96 & 7)) * 4 + (vrow + 96 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_51 & 127));
        *(reinterpret_cast<int*>(q4l + q_off3) + (0)) = q_codes_l3[0];
        *(reinterpret_cast<int*>(q4l + (q_off3 + 4)) + (0)) = q_codes_l3[1];
        uint16_t _cvt_satfinite_e4m3x2_f32_52;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_52) : "f"(0.0f), "f"(q_sf_l[3]));
        *reinterpret_cast<unsigned char*>(sfq_l + (q_tile + (long long)((((((vrow + 96 & 31) >> 3) * 2 + group_pair) * 8 + (vrow + 96 & 7)) * 4 + (vrow + 96 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_52 & 127));
        if (tid < 128) {
            *(reinterpret_cast<float*>(qm4 + (((long long)work * 128 + (long long)tid) * 4)) + (0)) = __uint_as_float(__as_u32(qm_smem[tid]) + 4096 >> 13 << 13);
        }
        if (has_kv == 1) {
            {
                unsigned _ldv8_53_0;
                unsigned _ldv8_53_1;
                unsigned _ldv8_53_2;
                unsigned _ldv8_53_3;
                unsigned _ldv8_53_4;
                unsigned _ldv8_53_5;
                unsigned _ldv8_53_6;
                unsigned _ldv8_53_7;
                asm volatile(
                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_53_0), "=r"(_ldv8_53_1), "=r"(_ldv8_53_2), "=r"(_ldv8_53_3), "=r"(_ldv8_53_4), "=r"(_ldv8_53_5), "=r"(_ldv8_53_6), "=r"(_ldv8_53_7) : "l"((const void*)(mean_k + ((long long)(bh * 128 + group * 16)))) : "memory");
                km_slice[0 + 0] = __uint_as_float(_ldv8_53_0);
                km_slice[0 + 1] = __uint_as_float(_ldv8_53_1);
                km_slice[0 + 2] = __uint_as_float(_ldv8_53_2);
                km_slice[0 + 3] = __uint_as_float(_ldv8_53_3);
                km_slice[0 + 4] = __uint_as_float(_ldv8_53_4);
                km_slice[0 + 5] = __uint_as_float(_ldv8_53_5);
                km_slice[0 + 6] = __uint_as_float(_ldv8_53_6);
                km_slice[0 + 7] = __uint_as_float(_ldv8_53_7);
            }
            {
                unsigned _ldv8_54_0;
                unsigned _ldv8_54_1;
                unsigned _ldv8_54_2;
                unsigned _ldv8_54_3;
                unsigned _ldv8_54_4;
                unsigned _ldv8_54_5;
                unsigned _ldv8_54_6;
                unsigned _ldv8_54_7;
                asm volatile(
                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_54_0), "=r"(_ldv8_54_1), "=r"(_ldv8_54_2), "=r"(_ldv8_54_3), "=r"(_ldv8_54_4), "=r"(_ldv8_54_5), "=r"(_ldv8_54_6), "=r"(_ldv8_54_7) : "l"((const void*)(mean_k + ((long long)(bh * 128 + group * 16 + 8)))) : "memory");
                km_slice[8 + 0] = __uint_as_float(_ldv8_54_0);
                km_slice[8 + 1] = __uint_as_float(_ldv8_54_1);
                km_slice[8 + 2] = __uint_as_float(_ldv8_54_2);
                km_slice[8 + 3] = __uint_as_float(_ldv8_54_3);
                km_slice[8 + 4] = __uint_as_float(_ldv8_54_4);
                km_slice[8 + 5] = __uint_as_float(_ldv8_54_5);
                km_slice[8 + 6] = __uint_as_float(_ldv8_54_6);
                km_slice[8 + 7] = __uint_as_float(_ldv8_54_7);
            }
            if (n_real > vrow) {
                {
                    const uint4* _vptr_55 = reinterpret_cast<const uint4*>(K + k_base + (long long)vrow * k_stride_s + (long long)(group * 16));
                    uint4 _vld_55[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
                        _vld_55[_blk] = _vptr_55[_blk];
                        uint32_t* _vpairs_55 = reinterpret_cast<uint32_t*>(&_vld_55[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) << 16);
                            (&qv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) & 0xffff0000u);
                        }
                    }
                }
                x[0] = qv[0] - km_slice[0];
                x[1] = qv[1] - km_slice[1];
                x[2] = qv[2] - km_slice[2];
                x[3] = qv[3] - km_slice[3];
                x[4] = qv[4] - km_slice[4];
                x[5] = qv[5] - km_slice[5];
                x[6] = qv[6] - km_slice[6];
                x[7] = qv[7] - km_slice[7];
                x[8] = qv[8] - km_slice[8];
                x[9] = qv[9] - km_slice[9];
                x[10] = qv[10] - km_slice[10];
                x[11] = qv[11] - km_slice[11];
                x[12] = qv[12] - km_slice[12];
                x[13] = qv[13] - km_slice[13];
                x[14] = qv[14] - km_slice[14];
                x[15] = qv[15] - km_slice[15];
            } else {
                x[0] = 0.0f;
                x[1] = 0.0f;
                x[2] = 0.0f;
                x[3] = 0.0f;
                x[4] = 0.0f;
                x[5] = 0.0f;
                x[6] = 0.0f;
                x[7] = 0.0f;
                x[8] = 0.0f;
                x[9] = 0.0f;
                x[10] = 0.0f;
                x[11] = 0.0f;
                x[12] = 0.0f;
                x[13] = 0.0f;
                x[14] = 0.0f;
                x[15] = 0.0f;
            }
            long long kc_off0 = ((kv_row0 + (long long)vrow) * 128 + (long long)(group * 16)) * 4;
            for (int e = 0; e < 4; e++) {
                xt[e] = __uint_as_float(__as_u32(x[e]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + kc_off0) + 0) = _v4;
            }
            for (int e_1 = 0; e_1 < 4; e_1++) {
                xt[e_1] = __uint_as_float(__as_u32(x[4 + e_1]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + (kc_off0 + 16)) + 0) = _v4;
            }
            for (int e_2 = 0; e_2 < 4; e_2++) {
                xt[e_2] = __uint_as_float(__as_u32(x[8 + e_2]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + (kc_off0 + 32)) + 0) = _v4;
            }
            for (int e_3 = 0; e_3 < 4; e_3++) {
                xt[e_3] = __uint_as_float(__as_u32(x[12 + e_3]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + (kc_off0 + 48)) + 0) = _v4;
            }
            float _max_8 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
            uint16_t _e4m3x2_58;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_58) : "f"(0.0f), "f"(_max_8));
            uint8_t _cvt_satfinite_e4m3_f32_57 = (uint8_t)_e4m3x2_58;
            uint32_t _f16x2_decode_59;
            float _fp8_e4m3_decode_56;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_59) : "h"(_e4m3x2_58));
            uint16_t _f16_decode_59 = (uint16_t)_f16x2_decode_59;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_56) : "h"(_f16_decode_59));
            float _fdiv_rn_8 = __fdiv_rn(1.0f, _fp8_e4m3_decode_56);
            x[0] = x[0] * _fdiv_rn_8;
            x[1] = x[1] * _fdiv_rn_8;
            x[2] = x[2] * _fdiv_rn_8;
            x[3] = x[3] * _fdiv_rn_8;
            x[4] = x[4] * _fdiv_rn_8;
            x[5] = x[5] * _fdiv_rn_8;
            x[6] = x[6] * _fdiv_rn_8;
            x[7] = x[7] * _fdiv_rn_8;
            x[8] = x[8] * _fdiv_rn_8;
            x[9] = x[9] * _fdiv_rn_8;
            x[10] = x[10] * _fdiv_rn_8;
            x[11] = x[11] * _fdiv_rn_8;
            x[12] = x[12] * _fdiv_rn_8;
            x[13] = x[13] * _fdiv_rn_8;
            x[14] = x[14] * _fdiv_rn_8;
            x[15] = x[15] * _fdiv_rn_8;
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
            sf_tmp[0] = _max_8;
            long long k_off0 = (kv_row0 + (long long)vrow) * 64 + (long long)(group * 8);
            *(reinterpret_cast<int*>(k4 + k_off0) + (0)) = codes[0];
            *(reinterpret_cast<int*>(k4 + (k_off0 + 4)) + (0)) = codes[1];
            uint16_t _cvt_satfinite_e4m3x2_f32_60;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_60) : "f"(0.0f), "f"(sf_tmp[0]));
            *reinterpret_cast<unsigned char*>(sfk + (kv_tile * 1024 + (long long)((((((vrow & 31) >> 3) * 2 + group_pair) * 8 + (vrow & 7)) * 4 + (vrow >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_60 & 127));
            if (n_real > vrow) {
                x[0] = qv[0] - km_slice[0];
                x[1] = qv[1] - km_slice[1];
                x[2] = qv[2] - km_slice[2];
                x[3] = qv[3] - km_slice[3];
                x[4] = qv[4] - km_slice[4];
                x[5] = qv[5] - km_slice[5];
                x[6] = qv[6] - km_slice[6];
                x[7] = qv[7] - km_slice[7];
                x[8] = qv[8] - km_slice[8];
                x[9] = qv[9] - km_slice[9];
                x[10] = qv[10] - km_slice[10];
                x[11] = qv[11] - km_slice[11];
                x[12] = qv[12] - km_slice[12];
                x[13] = qv[13] - km_slice[13];
                x[14] = qv[14] - km_slice[14];
                x[15] = qv[15] - km_slice[15];
            } else {
                x[0] = 0.0f;
                x[1] = 0.0f;
                x[2] = 0.0f;
                x[3] = 0.0f;
                x[4] = 0.0f;
                x[5] = 0.0f;
                x[6] = 0.0f;
                x[7] = 0.0f;
                x[8] = 0.0f;
                x[9] = 0.0f;
                x[10] = 0.0f;
                x[11] = 0.0f;
                x[12] = 0.0f;
                x[13] = 0.0f;
                x[14] = 0.0f;
                x[15] = 0.0f;
            }
            uint16_t _e4m3x2_62;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_62) : "f"(0.0f), "f"(sf_tmp[0]));
            uint8_t _cvt_satfinite_e4m3_f32_61 = (uint8_t)_e4m3x2_62;
            uint32_t _f16x2_decode_63;
            float _fp8_decode_4;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_63) : "h"(_e4m3x2_62));
            uint16_t _f16_decode_63 = (uint16_t)_f16x2_decode_63;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_4) : "h"(_f16_decode_63));
            float _unpack_fp4_4[16];
            {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                uint32_t _fp4_decode_pairs[4];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[0]));
                (&_unpack_fp4_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                (&_unpack_fp4_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                (&_unpack_fp4_4[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                (&_unpack_fp4_4[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                (&_unpack_fp4_4[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                (&_unpack_fp4_4[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                (&_unpack_fp4_4[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                (&_unpack_fp4_4[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                #else
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_4[0]), "=f"(_unpack_fp4_4[1]), "=f"(_unpack_fp4_4[2]), "=f"(_unpack_fp4_4[3]), "=f"(_unpack_fp4_4[4]), "=f"(_unpack_fp4_4[5]), "=f"(_unpack_fp4_4[6]), "=f"(_unpack_fp4_4[7]) : "r"(codes[0]));
                #endif
            }
            {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                uint32_t _fp4_decode_pairs[4];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[1]));
                (&_unpack_fp4_4[8])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                (&_unpack_fp4_4[8])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                (&_unpack_fp4_4[10])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                (&_unpack_fp4_4[10])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                (&_unpack_fp4_4[12])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                (&_unpack_fp4_4[12])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                (&_unpack_fp4_4[14])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                (&_unpack_fp4_4[14])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                #else
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_4[8]), "=f"(_unpack_fp4_4[9]), "=f"(_unpack_fp4_4[10]), "=f"(_unpack_fp4_4[11]), "=f"(_unpack_fp4_4[12]), "=f"(_unpack_fp4_4[13]), "=f"(_unpack_fp4_4[14]), "=f"(_unpack_fp4_4[15]) : "r"(codes[1]));
                #endif
            }
            x[0] = __fmaf_rn(-_unpack_fp4_4[0], _fp8_decode_4, x[0]);
            x[1] = __fmaf_rn(-_unpack_fp4_4[1], _fp8_decode_4, x[1]);
            x[2] = __fmaf_rn(-_unpack_fp4_4[2], _fp8_decode_4, x[2]);
            x[3] = __fmaf_rn(-_unpack_fp4_4[3], _fp8_decode_4, x[3]);
            x[4] = __fmaf_rn(-_unpack_fp4_4[4], _fp8_decode_4, x[4]);
            x[5] = __fmaf_rn(-_unpack_fp4_4[5], _fp8_decode_4, x[5]);
            x[6] = __fmaf_rn(-_unpack_fp4_4[6], _fp8_decode_4, x[6]);
            x[7] = __fmaf_rn(-_unpack_fp4_4[7], _fp8_decode_4, x[7]);
            x[8] = __fmaf_rn(-_unpack_fp4_4[8], _fp8_decode_4, x[8]);
            x[9] = __fmaf_rn(-_unpack_fp4_4[9], _fp8_decode_4, x[9]);
            x[10] = __fmaf_rn(-_unpack_fp4_4[10], _fp8_decode_4, x[10]);
            x[11] = __fmaf_rn(-_unpack_fp4_4[11], _fp8_decode_4, x[11]);
            x[12] = __fmaf_rn(-_unpack_fp4_4[12], _fp8_decode_4, x[12]);
            x[13] = __fmaf_rn(-_unpack_fp4_4[13], _fp8_decode_4, x[13]);
            x[14] = __fmaf_rn(-_unpack_fp4_4[14], _fp8_decode_4, x[14]);
            x[15] = __fmaf_rn(-_unpack_fp4_4[15], _fp8_decode_4, x[15]);
            float _max_9 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
            uint16_t _e4m3x2_66;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_66) : "f"(0.0f), "f"(_max_9));
            uint8_t _cvt_satfinite_e4m3_f32_65 = (uint8_t)_e4m3x2_66;
            uint32_t _f16x2_decode_67;
            float _fp8_e4m3_decode_64;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_67) : "h"(_e4m3x2_66));
            uint16_t _f16_decode_67 = (uint16_t)_f16x2_decode_67;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_64) : "h"(_f16_decode_67));
            float _fdiv_rn_9 = __fdiv_rn(1.0f, _fp8_e4m3_decode_64);
            x[0] = x[0] * _fdiv_rn_9;
            x[1] = x[1] * _fdiv_rn_9;
            x[2] = x[2] * _fdiv_rn_9;
            x[3] = x[3] * _fdiv_rn_9;
            x[4] = x[4] * _fdiv_rn_9;
            x[5] = x[5] * _fdiv_rn_9;
            x[6] = x[6] * _fdiv_rn_9;
            x[7] = x[7] * _fdiv_rn_9;
            x[8] = x[8] * _fdiv_rn_9;
            x[9] = x[9] * _fdiv_rn_9;
            x[10] = x[10] * _fdiv_rn_9;
            x[11] = x[11] * _fdiv_rn_9;
            x[12] = x[12] * _fdiv_rn_9;
            x[13] = x[13] * _fdiv_rn_9;
            x[14] = x[14] * _fdiv_rn_9;
            x[15] = x[15] * _fdiv_rn_9;
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
            sf_tmp[0] = _max_9;
            *(reinterpret_cast<int*>(k4l + k_off0) + (0)) = codes[0];
            *(reinterpret_cast<int*>(k4l + (k_off0 + 4)) + (0)) = codes[1];
            uint16_t _cvt_satfinite_e4m3x2_f32_68;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_68) : "f"(0.0f), "f"(sf_tmp[0]));
            *reinterpret_cast<unsigned char*>(sfk_l + (kv_tile * 1024 + (long long)((((((vrow & 31) >> 3) * 2 + group_pair) * 8 + (vrow & 7)) * 4 + (vrow >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_68 & 127));
            if (n_real > vrow + 32) {
                {
                    const uint4* _vptr_69 = reinterpret_cast<const uint4*>(K + k_base + (long long)(vrow + 32) * k_stride_s + (long long)(group * 16));
                    uint4 _vld_69[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
                        _vld_69[_blk] = _vptr_69[_blk];
                        uint32_t* _vpairs_69 = reinterpret_cast<uint32_t*>(&_vld_69[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_69[_pair]) << 16);
                            (&qv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_69[_pair]) & 0xffff0000u);
                        }
                    }
                }
                x[0] = qv[0] - km_slice[0];
                x[1] = qv[1] - km_slice[1];
                x[2] = qv[2] - km_slice[2];
                x[3] = qv[3] - km_slice[3];
                x[4] = qv[4] - km_slice[4];
                x[5] = qv[5] - km_slice[5];
                x[6] = qv[6] - km_slice[6];
                x[7] = qv[7] - km_slice[7];
                x[8] = qv[8] - km_slice[8];
                x[9] = qv[9] - km_slice[9];
                x[10] = qv[10] - km_slice[10];
                x[11] = qv[11] - km_slice[11];
                x[12] = qv[12] - km_slice[12];
                x[13] = qv[13] - km_slice[13];
                x[14] = qv[14] - km_slice[14];
                x[15] = qv[15] - km_slice[15];
            } else {
                x[0] = 0.0f;
                x[1] = 0.0f;
                x[2] = 0.0f;
                x[3] = 0.0f;
                x[4] = 0.0f;
                x[5] = 0.0f;
                x[6] = 0.0f;
                x[7] = 0.0f;
                x[8] = 0.0f;
                x[9] = 0.0f;
                x[10] = 0.0f;
                x[11] = 0.0f;
                x[12] = 0.0f;
                x[13] = 0.0f;
                x[14] = 0.0f;
                x[15] = 0.0f;
            }
            long long kc_off1 = ((kv_row0 + (long long)(vrow + 32)) * 128 + (long long)(group * 16)) * 4;
            for (int e_4 = 0; e_4 < 4; e_4++) {
                xt[e_4] = __uint_as_float(__as_u32(x[e_4]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + kc_off1) + 0) = _v4;
            }
            for (int e_5 = 0; e_5 < 4; e_5++) {
                xt[e_5] = __uint_as_float(__as_u32(x[4 + e_5]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + (kc_off1 + 16)) + 0) = _v4;
            }
            for (int e_6 = 0; e_6 < 4; e_6++) {
                xt[e_6] = __uint_as_float(__as_u32(x[8 + e_6]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + (kc_off1 + 32)) + 0) = _v4;
            }
            for (int e_7 = 0; e_7 < 4; e_7++) {
                xt[e_7] = __uint_as_float(__as_u32(x[12 + e_7]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + (kc_off1 + 48)) + 0) = _v4;
            }
            float _max_10 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
            uint16_t _e4m3x2_72;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_72) : "f"(0.0f), "f"(_max_10));
            uint8_t _cvt_satfinite_e4m3_f32_71 = (uint8_t)_e4m3x2_72;
            uint32_t _f16x2_decode_73;
            float _fp8_e4m3_decode_70;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_73) : "h"(_e4m3x2_72));
            uint16_t _f16_decode_73 = (uint16_t)_f16x2_decode_73;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_70) : "h"(_f16_decode_73));
            float _fdiv_rn_10 = __fdiv_rn(1.0f, _fp8_e4m3_decode_70);
            x[0] = x[0] * _fdiv_rn_10;
            x[1] = x[1] * _fdiv_rn_10;
            x[2] = x[2] * _fdiv_rn_10;
            x[3] = x[3] * _fdiv_rn_10;
            x[4] = x[4] * _fdiv_rn_10;
            x[5] = x[5] * _fdiv_rn_10;
            x[6] = x[6] * _fdiv_rn_10;
            x[7] = x[7] * _fdiv_rn_10;
            x[8] = x[8] * _fdiv_rn_10;
            x[9] = x[9] * _fdiv_rn_10;
            x[10] = x[10] * _fdiv_rn_10;
            x[11] = x[11] * _fdiv_rn_10;
            x[12] = x[12] * _fdiv_rn_10;
            x[13] = x[13] * _fdiv_rn_10;
            x[14] = x[14] * _fdiv_rn_10;
            x[15] = x[15] * _fdiv_rn_10;
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
            sf_tmp[0] = _max_10;
            long long k_off1 = (kv_row0 + (long long)(vrow + 32)) * 64 + (long long)(group * 8);
            *(reinterpret_cast<int*>(k4 + k_off1) + (0)) = codes[0];
            *(reinterpret_cast<int*>(k4 + (k_off1 + 4)) + (0)) = codes[1];
            uint16_t _cvt_satfinite_e4m3x2_f32_74;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_74) : "f"(0.0f), "f"(sf_tmp[0]));
            *reinterpret_cast<unsigned char*>(sfk + (kv_tile * 1024 + (long long)((((((vrow + 32 & 31) >> 3) * 2 + group_pair) * 8 + (vrow + 32 & 7)) * 4 + (vrow + 32 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_74 & 127));
            if (n_real > vrow + 32) {
                x[0] = qv[0] - km_slice[0];
                x[1] = qv[1] - km_slice[1];
                x[2] = qv[2] - km_slice[2];
                x[3] = qv[3] - km_slice[3];
                x[4] = qv[4] - km_slice[4];
                x[5] = qv[5] - km_slice[5];
                x[6] = qv[6] - km_slice[6];
                x[7] = qv[7] - km_slice[7];
                x[8] = qv[8] - km_slice[8];
                x[9] = qv[9] - km_slice[9];
                x[10] = qv[10] - km_slice[10];
                x[11] = qv[11] - km_slice[11];
                x[12] = qv[12] - km_slice[12];
                x[13] = qv[13] - km_slice[13];
                x[14] = qv[14] - km_slice[14];
                x[15] = qv[15] - km_slice[15];
            } else {
                x[0] = 0.0f;
                x[1] = 0.0f;
                x[2] = 0.0f;
                x[3] = 0.0f;
                x[4] = 0.0f;
                x[5] = 0.0f;
                x[6] = 0.0f;
                x[7] = 0.0f;
                x[8] = 0.0f;
                x[9] = 0.0f;
                x[10] = 0.0f;
                x[11] = 0.0f;
                x[12] = 0.0f;
                x[13] = 0.0f;
                x[14] = 0.0f;
                x[15] = 0.0f;
            }
            uint16_t _e4m3x2_76;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_76) : "f"(0.0f), "f"(sf_tmp[0]));
            uint8_t _cvt_satfinite_e4m3_f32_75 = (uint8_t)_e4m3x2_76;
            uint32_t _f16x2_decode_77;
            float _fp8_decode_5;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_77) : "h"(_e4m3x2_76));
            uint16_t _f16_decode_77 = (uint16_t)_f16x2_decode_77;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_5) : "h"(_f16_decode_77));
            float _unpack_fp4_5[16];
            {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                uint32_t _fp4_decode_pairs[4];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[0]));
                (&_unpack_fp4_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                (&_unpack_fp4_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                (&_unpack_fp4_5[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                (&_unpack_fp4_5[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                (&_unpack_fp4_5[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                (&_unpack_fp4_5[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                (&_unpack_fp4_5[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                (&_unpack_fp4_5[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                #else
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_5[0]), "=f"(_unpack_fp4_5[1]), "=f"(_unpack_fp4_5[2]), "=f"(_unpack_fp4_5[3]), "=f"(_unpack_fp4_5[4]), "=f"(_unpack_fp4_5[5]), "=f"(_unpack_fp4_5[6]), "=f"(_unpack_fp4_5[7]) : "r"(codes[0]));
                #endif
            }
            {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                uint32_t _fp4_decode_pairs[4];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[1]));
                (&_unpack_fp4_5[8])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                (&_unpack_fp4_5[8])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                (&_unpack_fp4_5[10])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                (&_unpack_fp4_5[10])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                (&_unpack_fp4_5[12])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                (&_unpack_fp4_5[12])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                (&_unpack_fp4_5[14])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                (&_unpack_fp4_5[14])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                #else
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_5[8]), "=f"(_unpack_fp4_5[9]), "=f"(_unpack_fp4_5[10]), "=f"(_unpack_fp4_5[11]), "=f"(_unpack_fp4_5[12]), "=f"(_unpack_fp4_5[13]), "=f"(_unpack_fp4_5[14]), "=f"(_unpack_fp4_5[15]) : "r"(codes[1]));
                #endif
            }
            x[0] = __fmaf_rn(-_unpack_fp4_5[0], _fp8_decode_5, x[0]);
            x[1] = __fmaf_rn(-_unpack_fp4_5[1], _fp8_decode_5, x[1]);
            x[2] = __fmaf_rn(-_unpack_fp4_5[2], _fp8_decode_5, x[2]);
            x[3] = __fmaf_rn(-_unpack_fp4_5[3], _fp8_decode_5, x[3]);
            x[4] = __fmaf_rn(-_unpack_fp4_5[4], _fp8_decode_5, x[4]);
            x[5] = __fmaf_rn(-_unpack_fp4_5[5], _fp8_decode_5, x[5]);
            x[6] = __fmaf_rn(-_unpack_fp4_5[6], _fp8_decode_5, x[6]);
            x[7] = __fmaf_rn(-_unpack_fp4_5[7], _fp8_decode_5, x[7]);
            x[8] = __fmaf_rn(-_unpack_fp4_5[8], _fp8_decode_5, x[8]);
            x[9] = __fmaf_rn(-_unpack_fp4_5[9], _fp8_decode_5, x[9]);
            x[10] = __fmaf_rn(-_unpack_fp4_5[10], _fp8_decode_5, x[10]);
            x[11] = __fmaf_rn(-_unpack_fp4_5[11], _fp8_decode_5, x[11]);
            x[12] = __fmaf_rn(-_unpack_fp4_5[12], _fp8_decode_5, x[12]);
            x[13] = __fmaf_rn(-_unpack_fp4_5[13], _fp8_decode_5, x[13]);
            x[14] = __fmaf_rn(-_unpack_fp4_5[14], _fp8_decode_5, x[14]);
            x[15] = __fmaf_rn(-_unpack_fp4_5[15], _fp8_decode_5, x[15]);
            float _max_11 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
            uint16_t _e4m3x2_80;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_80) : "f"(0.0f), "f"(_max_11));
            uint8_t _cvt_satfinite_e4m3_f32_79 = (uint8_t)_e4m3x2_80;
            uint32_t _f16x2_decode_81;
            float _fp8_e4m3_decode_78;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_81) : "h"(_e4m3x2_80));
            uint16_t _f16_decode_81 = (uint16_t)_f16x2_decode_81;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_78) : "h"(_f16_decode_81));
            float _fdiv_rn_11 = __fdiv_rn(1.0f, _fp8_e4m3_decode_78);
            x[0] = x[0] * _fdiv_rn_11;
            x[1] = x[1] * _fdiv_rn_11;
            x[2] = x[2] * _fdiv_rn_11;
            x[3] = x[3] * _fdiv_rn_11;
            x[4] = x[4] * _fdiv_rn_11;
            x[5] = x[5] * _fdiv_rn_11;
            x[6] = x[6] * _fdiv_rn_11;
            x[7] = x[7] * _fdiv_rn_11;
            x[8] = x[8] * _fdiv_rn_11;
            x[9] = x[9] * _fdiv_rn_11;
            x[10] = x[10] * _fdiv_rn_11;
            x[11] = x[11] * _fdiv_rn_11;
            x[12] = x[12] * _fdiv_rn_11;
            x[13] = x[13] * _fdiv_rn_11;
            x[14] = x[14] * _fdiv_rn_11;
            x[15] = x[15] * _fdiv_rn_11;
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
            sf_tmp[0] = _max_11;
            *(reinterpret_cast<int*>(k4l + k_off1) + (0)) = codes[0];
            *(reinterpret_cast<int*>(k4l + (k_off1 + 4)) + (0)) = codes[1];
            uint16_t _cvt_satfinite_e4m3x2_f32_82;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_82) : "f"(0.0f), "f"(sf_tmp[0]));
            *reinterpret_cast<unsigned char*>(sfk_l + (kv_tile * 1024 + (long long)((((((vrow + 32 & 31) >> 3) * 2 + group_pair) * 8 + (vrow + 32 & 7)) * 4 + (vrow + 32 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_82 & 127));
            if (n_real > vrow + 64) {
                {
                    const uint4* _vptr_83 = reinterpret_cast<const uint4*>(K + k_base + (long long)(vrow + 64) * k_stride_s + (long long)(group * 16));
                    uint4 _vld_83[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
                        _vld_83[_blk] = _vptr_83[_blk];
                        uint32_t* _vpairs_83 = reinterpret_cast<uint32_t*>(&_vld_83[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_83[_pair]) << 16);
                            (&qv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_83[_pair]) & 0xffff0000u);
                        }
                    }
                }
                x[0] = qv[0] - km_slice[0];
                x[1] = qv[1] - km_slice[1];
                x[2] = qv[2] - km_slice[2];
                x[3] = qv[3] - km_slice[3];
                x[4] = qv[4] - km_slice[4];
                x[5] = qv[5] - km_slice[5];
                x[6] = qv[6] - km_slice[6];
                x[7] = qv[7] - km_slice[7];
                x[8] = qv[8] - km_slice[8];
                x[9] = qv[9] - km_slice[9];
                x[10] = qv[10] - km_slice[10];
                x[11] = qv[11] - km_slice[11];
                x[12] = qv[12] - km_slice[12];
                x[13] = qv[13] - km_slice[13];
                x[14] = qv[14] - km_slice[14];
                x[15] = qv[15] - km_slice[15];
            } else {
                x[0] = 0.0f;
                x[1] = 0.0f;
                x[2] = 0.0f;
                x[3] = 0.0f;
                x[4] = 0.0f;
                x[5] = 0.0f;
                x[6] = 0.0f;
                x[7] = 0.0f;
                x[8] = 0.0f;
                x[9] = 0.0f;
                x[10] = 0.0f;
                x[11] = 0.0f;
                x[12] = 0.0f;
                x[13] = 0.0f;
                x[14] = 0.0f;
                x[15] = 0.0f;
            }
            long long kc_off2 = ((kv_row0 + (long long)(vrow + 64)) * 128 + (long long)(group * 16)) * 4;
            for (int e_8 = 0; e_8 < 4; e_8++) {
                xt[e_8] = __uint_as_float(__as_u32(x[e_8]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + kc_off2) + 0) = _v4;
            }
            for (int e_9 = 0; e_9 < 4; e_9++) {
                xt[e_9] = __uint_as_float(__as_u32(x[4 + e_9]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + (kc_off2 + 16)) + 0) = _v4;
            }
            for (int e_10 = 0; e_10 < 4; e_10++) {
                xt[e_10] = __uint_as_float(__as_u32(x[8 + e_10]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + (kc_off2 + 32)) + 0) = _v4;
            }
            for (int e_11 = 0; e_11 < 4; e_11++) {
                xt[e_11] = __uint_as_float(__as_u32(x[12 + e_11]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + (kc_off2 + 48)) + 0) = _v4;
            }
            float _max_12 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
            uint16_t _e4m3x2_86;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_86) : "f"(0.0f), "f"(_max_12));
            uint8_t _cvt_satfinite_e4m3_f32_85 = (uint8_t)_e4m3x2_86;
            uint32_t _f16x2_decode_87;
            float _fp8_e4m3_decode_84;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_87) : "h"(_e4m3x2_86));
            uint16_t _f16_decode_87 = (uint16_t)_f16x2_decode_87;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_84) : "h"(_f16_decode_87));
            float _fdiv_rn_12 = __fdiv_rn(1.0f, _fp8_e4m3_decode_84);
            x[0] = x[0] * _fdiv_rn_12;
            x[1] = x[1] * _fdiv_rn_12;
            x[2] = x[2] * _fdiv_rn_12;
            x[3] = x[3] * _fdiv_rn_12;
            x[4] = x[4] * _fdiv_rn_12;
            x[5] = x[5] * _fdiv_rn_12;
            x[6] = x[6] * _fdiv_rn_12;
            x[7] = x[7] * _fdiv_rn_12;
            x[8] = x[8] * _fdiv_rn_12;
            x[9] = x[9] * _fdiv_rn_12;
            x[10] = x[10] * _fdiv_rn_12;
            x[11] = x[11] * _fdiv_rn_12;
            x[12] = x[12] * _fdiv_rn_12;
            x[13] = x[13] * _fdiv_rn_12;
            x[14] = x[14] * _fdiv_rn_12;
            x[15] = x[15] * _fdiv_rn_12;
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
            sf_tmp[0] = _max_12;
            long long k_off2 = (kv_row0 + (long long)(vrow + 64)) * 64 + (long long)(group * 8);
            *(reinterpret_cast<int*>(k4 + k_off2) + (0)) = codes[0];
            *(reinterpret_cast<int*>(k4 + (k_off2 + 4)) + (0)) = codes[1];
            uint16_t _cvt_satfinite_e4m3x2_f32_88;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_88) : "f"(0.0f), "f"(sf_tmp[0]));
            *reinterpret_cast<unsigned char*>(sfk + (kv_tile * 1024 + (long long)((((((vrow + 64 & 31) >> 3) * 2 + group_pair) * 8 + (vrow + 64 & 7)) * 4 + (vrow + 64 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_88 & 127));
            if (n_real > vrow + 64) {
                x[0] = qv[0] - km_slice[0];
                x[1] = qv[1] - km_slice[1];
                x[2] = qv[2] - km_slice[2];
                x[3] = qv[3] - km_slice[3];
                x[4] = qv[4] - km_slice[4];
                x[5] = qv[5] - km_slice[5];
                x[6] = qv[6] - km_slice[6];
                x[7] = qv[7] - km_slice[7];
                x[8] = qv[8] - km_slice[8];
                x[9] = qv[9] - km_slice[9];
                x[10] = qv[10] - km_slice[10];
                x[11] = qv[11] - km_slice[11];
                x[12] = qv[12] - km_slice[12];
                x[13] = qv[13] - km_slice[13];
                x[14] = qv[14] - km_slice[14];
                x[15] = qv[15] - km_slice[15];
            } else {
                x[0] = 0.0f;
                x[1] = 0.0f;
                x[2] = 0.0f;
                x[3] = 0.0f;
                x[4] = 0.0f;
                x[5] = 0.0f;
                x[6] = 0.0f;
                x[7] = 0.0f;
                x[8] = 0.0f;
                x[9] = 0.0f;
                x[10] = 0.0f;
                x[11] = 0.0f;
                x[12] = 0.0f;
                x[13] = 0.0f;
                x[14] = 0.0f;
                x[15] = 0.0f;
            }
            uint16_t _e4m3x2_90;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_90) : "f"(0.0f), "f"(sf_tmp[0]));
            uint8_t _cvt_satfinite_e4m3_f32_89 = (uint8_t)_e4m3x2_90;
            uint32_t _f16x2_decode_91;
            float _fp8_decode_6;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_91) : "h"(_e4m3x2_90));
            uint16_t _f16_decode_91 = (uint16_t)_f16x2_decode_91;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_6) : "h"(_f16_decode_91));
            float _unpack_fp4_6[16];
            {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                uint32_t _fp4_decode_pairs[4];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[0]));
                (&_unpack_fp4_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                (&_unpack_fp4_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                (&_unpack_fp4_6[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                (&_unpack_fp4_6[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                (&_unpack_fp4_6[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                (&_unpack_fp4_6[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                (&_unpack_fp4_6[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                (&_unpack_fp4_6[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                #else
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_6[0]), "=f"(_unpack_fp4_6[1]), "=f"(_unpack_fp4_6[2]), "=f"(_unpack_fp4_6[3]), "=f"(_unpack_fp4_6[4]), "=f"(_unpack_fp4_6[5]), "=f"(_unpack_fp4_6[6]), "=f"(_unpack_fp4_6[7]) : "r"(codes[0]));
                #endif
            }
            {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                uint32_t _fp4_decode_pairs[4];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[1]));
                (&_unpack_fp4_6[8])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                (&_unpack_fp4_6[8])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                (&_unpack_fp4_6[10])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                (&_unpack_fp4_6[10])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                (&_unpack_fp4_6[12])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                (&_unpack_fp4_6[12])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                (&_unpack_fp4_6[14])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                (&_unpack_fp4_6[14])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                #else
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_6[8]), "=f"(_unpack_fp4_6[9]), "=f"(_unpack_fp4_6[10]), "=f"(_unpack_fp4_6[11]), "=f"(_unpack_fp4_6[12]), "=f"(_unpack_fp4_6[13]), "=f"(_unpack_fp4_6[14]), "=f"(_unpack_fp4_6[15]) : "r"(codes[1]));
                #endif
            }
            x[0] = __fmaf_rn(-_unpack_fp4_6[0], _fp8_decode_6, x[0]);
            x[1] = __fmaf_rn(-_unpack_fp4_6[1], _fp8_decode_6, x[1]);
            x[2] = __fmaf_rn(-_unpack_fp4_6[2], _fp8_decode_6, x[2]);
            x[3] = __fmaf_rn(-_unpack_fp4_6[3], _fp8_decode_6, x[3]);
            x[4] = __fmaf_rn(-_unpack_fp4_6[4], _fp8_decode_6, x[4]);
            x[5] = __fmaf_rn(-_unpack_fp4_6[5], _fp8_decode_6, x[5]);
            x[6] = __fmaf_rn(-_unpack_fp4_6[6], _fp8_decode_6, x[6]);
            x[7] = __fmaf_rn(-_unpack_fp4_6[7], _fp8_decode_6, x[7]);
            x[8] = __fmaf_rn(-_unpack_fp4_6[8], _fp8_decode_6, x[8]);
            x[9] = __fmaf_rn(-_unpack_fp4_6[9], _fp8_decode_6, x[9]);
            x[10] = __fmaf_rn(-_unpack_fp4_6[10], _fp8_decode_6, x[10]);
            x[11] = __fmaf_rn(-_unpack_fp4_6[11], _fp8_decode_6, x[11]);
            x[12] = __fmaf_rn(-_unpack_fp4_6[12], _fp8_decode_6, x[12]);
            x[13] = __fmaf_rn(-_unpack_fp4_6[13], _fp8_decode_6, x[13]);
            x[14] = __fmaf_rn(-_unpack_fp4_6[14], _fp8_decode_6, x[14]);
            x[15] = __fmaf_rn(-_unpack_fp4_6[15], _fp8_decode_6, x[15]);
            float _max_13 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
            uint16_t _e4m3x2_94;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_94) : "f"(0.0f), "f"(_max_13));
            uint8_t _cvt_satfinite_e4m3_f32_93 = (uint8_t)_e4m3x2_94;
            uint32_t _f16x2_decode_95;
            float _fp8_e4m3_decode_92;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_95) : "h"(_e4m3x2_94));
            uint16_t _f16_decode_95 = (uint16_t)_f16x2_decode_95;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_92) : "h"(_f16_decode_95));
            float _fdiv_rn_13 = __fdiv_rn(1.0f, _fp8_e4m3_decode_92);
            x[0] = x[0] * _fdiv_rn_13;
            x[1] = x[1] * _fdiv_rn_13;
            x[2] = x[2] * _fdiv_rn_13;
            x[3] = x[3] * _fdiv_rn_13;
            x[4] = x[4] * _fdiv_rn_13;
            x[5] = x[5] * _fdiv_rn_13;
            x[6] = x[6] * _fdiv_rn_13;
            x[7] = x[7] * _fdiv_rn_13;
            x[8] = x[8] * _fdiv_rn_13;
            x[9] = x[9] * _fdiv_rn_13;
            x[10] = x[10] * _fdiv_rn_13;
            x[11] = x[11] * _fdiv_rn_13;
            x[12] = x[12] * _fdiv_rn_13;
            x[13] = x[13] * _fdiv_rn_13;
            x[14] = x[14] * _fdiv_rn_13;
            x[15] = x[15] * _fdiv_rn_13;
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
            sf_tmp[0] = _max_13;
            *(reinterpret_cast<int*>(k4l + k_off2) + (0)) = codes[0];
            *(reinterpret_cast<int*>(k4l + (k_off2 + 4)) + (0)) = codes[1];
            uint16_t _cvt_satfinite_e4m3x2_f32_96;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_96) : "f"(0.0f), "f"(sf_tmp[0]));
            *reinterpret_cast<unsigned char*>(sfk_l + (kv_tile * 1024 + (long long)((((((vrow + 64 & 31) >> 3) * 2 + group_pair) * 8 + (vrow + 64 & 7)) * 4 + (vrow + 64 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_96 & 127));
            if (n_real > vrow + 96) {
                {
                    const uint4* _vptr_97 = reinterpret_cast<const uint4*>(K + k_base + (long long)(vrow + 96) * k_stride_s + (long long)(group * 16));
                    uint4 _vld_97[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
                        _vld_97[_blk] = _vptr_97[_blk];
                        uint32_t* _vpairs_97 = reinterpret_cast<uint32_t*>(&_vld_97[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_97[_pair]) << 16);
                            (&qv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_97[_pair]) & 0xffff0000u);
                        }
                    }
                }
                x[0] = qv[0] - km_slice[0];
                x[1] = qv[1] - km_slice[1];
                x[2] = qv[2] - km_slice[2];
                x[3] = qv[3] - km_slice[3];
                x[4] = qv[4] - km_slice[4];
                x[5] = qv[5] - km_slice[5];
                x[6] = qv[6] - km_slice[6];
                x[7] = qv[7] - km_slice[7];
                x[8] = qv[8] - km_slice[8];
                x[9] = qv[9] - km_slice[9];
                x[10] = qv[10] - km_slice[10];
                x[11] = qv[11] - km_slice[11];
                x[12] = qv[12] - km_slice[12];
                x[13] = qv[13] - km_slice[13];
                x[14] = qv[14] - km_slice[14];
                x[15] = qv[15] - km_slice[15];
            } else {
                x[0] = 0.0f;
                x[1] = 0.0f;
                x[2] = 0.0f;
                x[3] = 0.0f;
                x[4] = 0.0f;
                x[5] = 0.0f;
                x[6] = 0.0f;
                x[7] = 0.0f;
                x[8] = 0.0f;
                x[9] = 0.0f;
                x[10] = 0.0f;
                x[11] = 0.0f;
                x[12] = 0.0f;
                x[13] = 0.0f;
                x[14] = 0.0f;
                x[15] = 0.0f;
            }
            long long kc_off3 = ((kv_row0 + (long long)(vrow + 96)) * 128 + (long long)(group * 16)) * 4;
            for (int e_12 = 0; e_12 < 4; e_12++) {
                xt[e_12] = __uint_as_float(__as_u32(x[e_12]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + kc_off3) + 0) = _v4;
            }
            for (int e_13 = 0; e_13 < 4; e_13++) {
                xt[e_13] = __uint_as_float(__as_u32(x[4 + e_13]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + (kc_off3 + 16)) + 0) = _v4;
            }
            for (int e_14 = 0; e_14 < 4; e_14++) {
                xt[e_14] = __uint_as_float(__as_u32(x[8 + e_14]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + (kc_off3 + 32)) + 0) = _v4;
            }
            for (int e_15 = 0; e_15 < 4; e_15++) {
                xt[e_15] = __uint_as_float(__as_u32(x[12 + e_15]) + 4096 >> 13 << 13);
            }
            {
                float4 _v4 = make_float4((xt + 0)[0 + 0], (xt + 0)[0 + 1], (xt + 0)[0 + 2], (xt + 0)[0 + 3]);
                *reinterpret_cast<float4*>((sfqm + (kc_off3 + 48)) + 0) = _v4;
            }
            float _max_14 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
            uint16_t _e4m3x2_100;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_100) : "f"(0.0f), "f"(_max_14));
            uint8_t _cvt_satfinite_e4m3_f32_99 = (uint8_t)_e4m3x2_100;
            uint32_t _f16x2_decode_101;
            float _fp8_e4m3_decode_98;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_101) : "h"(_e4m3x2_100));
            uint16_t _f16_decode_101 = (uint16_t)_f16x2_decode_101;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_98) : "h"(_f16_decode_101));
            float _fdiv_rn_14 = __fdiv_rn(1.0f, _fp8_e4m3_decode_98);
            x[0] = x[0] * _fdiv_rn_14;
            x[1] = x[1] * _fdiv_rn_14;
            x[2] = x[2] * _fdiv_rn_14;
            x[3] = x[3] * _fdiv_rn_14;
            x[4] = x[4] * _fdiv_rn_14;
            x[5] = x[5] * _fdiv_rn_14;
            x[6] = x[6] * _fdiv_rn_14;
            x[7] = x[7] * _fdiv_rn_14;
            x[8] = x[8] * _fdiv_rn_14;
            x[9] = x[9] * _fdiv_rn_14;
            x[10] = x[10] * _fdiv_rn_14;
            x[11] = x[11] * _fdiv_rn_14;
            x[12] = x[12] * _fdiv_rn_14;
            x[13] = x[13] * _fdiv_rn_14;
            x[14] = x[14] * _fdiv_rn_14;
            x[15] = x[15] * _fdiv_rn_14;
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
            sf_tmp[0] = _max_14;
            long long k_off3 = (kv_row0 + (long long)(vrow + 96)) * 64 + (long long)(group * 8);
            *(reinterpret_cast<int*>(k4 + k_off3) + (0)) = codes[0];
            *(reinterpret_cast<int*>(k4 + (k_off3 + 4)) + (0)) = codes[1];
            uint16_t _cvt_satfinite_e4m3x2_f32_102;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_102) : "f"(0.0f), "f"(sf_tmp[0]));
            *reinterpret_cast<unsigned char*>(sfk + (kv_tile * 1024 + (long long)((((((vrow + 96 & 31) >> 3) * 2 + group_pair) * 8 + (vrow + 96 & 7)) * 4 + (vrow + 96 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_102 & 127));
            if (n_real > vrow + 96) {
                x[0] = qv[0] - km_slice[0];
                x[1] = qv[1] - km_slice[1];
                x[2] = qv[2] - km_slice[2];
                x[3] = qv[3] - km_slice[3];
                x[4] = qv[4] - km_slice[4];
                x[5] = qv[5] - km_slice[5];
                x[6] = qv[6] - km_slice[6];
                x[7] = qv[7] - km_slice[7];
                x[8] = qv[8] - km_slice[8];
                x[9] = qv[9] - km_slice[9];
                x[10] = qv[10] - km_slice[10];
                x[11] = qv[11] - km_slice[11];
                x[12] = qv[12] - km_slice[12];
                x[13] = qv[13] - km_slice[13];
                x[14] = qv[14] - km_slice[14];
                x[15] = qv[15] - km_slice[15];
            } else {
                x[0] = 0.0f;
                x[1] = 0.0f;
                x[2] = 0.0f;
                x[3] = 0.0f;
                x[4] = 0.0f;
                x[5] = 0.0f;
                x[6] = 0.0f;
                x[7] = 0.0f;
                x[8] = 0.0f;
                x[9] = 0.0f;
                x[10] = 0.0f;
                x[11] = 0.0f;
                x[12] = 0.0f;
                x[13] = 0.0f;
                x[14] = 0.0f;
                x[15] = 0.0f;
            }
            uint16_t _e4m3x2_104;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_104) : "f"(0.0f), "f"(sf_tmp[0]));
            uint8_t _cvt_satfinite_e4m3_f32_103 = (uint8_t)_e4m3x2_104;
            uint32_t _f16x2_decode_105;
            float _fp8_decode_7;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_105) : "h"(_e4m3x2_104));
            uint16_t _f16_decode_105 = (uint16_t)_f16x2_decode_105;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_7) : "h"(_f16_decode_105));
            float _unpack_fp4_7[16];
            {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                uint32_t _fp4_decode_pairs[4];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[0]));
                (&_unpack_fp4_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                (&_unpack_fp4_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                (&_unpack_fp4_7[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                (&_unpack_fp4_7[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                (&_unpack_fp4_7[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                (&_unpack_fp4_7[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                (&_unpack_fp4_7[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                (&_unpack_fp4_7[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                #else
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_7[0]), "=f"(_unpack_fp4_7[1]), "=f"(_unpack_fp4_7[2]), "=f"(_unpack_fp4_7[3]), "=f"(_unpack_fp4_7[4]), "=f"(_unpack_fp4_7[5]), "=f"(_unpack_fp4_7[6]), "=f"(_unpack_fp4_7[7]) : "r"(codes[0]));
                #endif
            }
            {
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                uint32_t _fp4_decode_pairs[4];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[1]));
                (&_unpack_fp4_7[8])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                (&_unpack_fp4_7[8])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                (&_unpack_fp4_7[10])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                (&_unpack_fp4_7[10])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                (&_unpack_fp4_7[12])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                (&_unpack_fp4_7[12])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                (&_unpack_fp4_7[14])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                (&_unpack_fp4_7[14])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                #else
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_7[8]), "=f"(_unpack_fp4_7[9]), "=f"(_unpack_fp4_7[10]), "=f"(_unpack_fp4_7[11]), "=f"(_unpack_fp4_7[12]), "=f"(_unpack_fp4_7[13]), "=f"(_unpack_fp4_7[14]), "=f"(_unpack_fp4_7[15]) : "r"(codes[1]));
                #endif
            }
            x[0] = __fmaf_rn(-_unpack_fp4_7[0], _fp8_decode_7, x[0]);
            x[1] = __fmaf_rn(-_unpack_fp4_7[1], _fp8_decode_7, x[1]);
            x[2] = __fmaf_rn(-_unpack_fp4_7[2], _fp8_decode_7, x[2]);
            x[3] = __fmaf_rn(-_unpack_fp4_7[3], _fp8_decode_7, x[3]);
            x[4] = __fmaf_rn(-_unpack_fp4_7[4], _fp8_decode_7, x[4]);
            x[5] = __fmaf_rn(-_unpack_fp4_7[5], _fp8_decode_7, x[5]);
            x[6] = __fmaf_rn(-_unpack_fp4_7[6], _fp8_decode_7, x[6]);
            x[7] = __fmaf_rn(-_unpack_fp4_7[7], _fp8_decode_7, x[7]);
            x[8] = __fmaf_rn(-_unpack_fp4_7[8], _fp8_decode_7, x[8]);
            x[9] = __fmaf_rn(-_unpack_fp4_7[9], _fp8_decode_7, x[9]);
            x[10] = __fmaf_rn(-_unpack_fp4_7[10], _fp8_decode_7, x[10]);
            x[11] = __fmaf_rn(-_unpack_fp4_7[11], _fp8_decode_7, x[11]);
            x[12] = __fmaf_rn(-_unpack_fp4_7[12], _fp8_decode_7, x[12]);
            x[13] = __fmaf_rn(-_unpack_fp4_7[13], _fp8_decode_7, x[13]);
            x[14] = __fmaf_rn(-_unpack_fp4_7[14], _fp8_decode_7, x[14]);
            x[15] = __fmaf_rn(-_unpack_fp4_7[15], _fp8_decode_7, x[15]);
            float _max_15 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
            uint16_t _e4m3x2_108;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_108) : "f"(0.0f), "f"(_max_15));
            uint8_t _cvt_satfinite_e4m3_f32_107 = (uint8_t)_e4m3x2_108;
            uint32_t _f16x2_decode_109;
            float _fp8_e4m3_decode_106;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_109) : "h"(_e4m3x2_108));
            uint16_t _f16_decode_109 = (uint16_t)_f16x2_decode_109;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_106) : "h"(_f16_decode_109));
            float _fdiv_rn_15 = __fdiv_rn(1.0f, _fp8_e4m3_decode_106);
            x[0] = x[0] * _fdiv_rn_15;
            x[1] = x[1] * _fdiv_rn_15;
            x[2] = x[2] * _fdiv_rn_15;
            x[3] = x[3] * _fdiv_rn_15;
            x[4] = x[4] * _fdiv_rn_15;
            x[5] = x[5] * _fdiv_rn_15;
            x[6] = x[6] * _fdiv_rn_15;
            x[7] = x[7] * _fdiv_rn_15;
            x[8] = x[8] * _fdiv_rn_15;
            x[9] = x[9] * _fdiv_rn_15;
            x[10] = x[10] * _fdiv_rn_15;
            x[11] = x[11] * _fdiv_rn_15;
            x[12] = x[12] * _fdiv_rn_15;
            x[13] = x[13] * _fdiv_rn_15;
            x[14] = x[14] * _fdiv_rn_15;
            x[15] = x[15] * _fdiv_rn_15;
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
            sf_tmp[0] = _max_15;
            *(reinterpret_cast<int*>(k4l + k_off3) + (0)) = codes[0];
            *(reinterpret_cast<int*>(k4l + (k_off3 + 4)) + (0)) = codes[1];
            uint16_t _cvt_satfinite_e4m3x2_f32_110;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_110) : "f"(0.0f), "f"(sf_tmp[0]));
            *reinterpret_cast<unsigned char*>(sfk_l + (kv_tile * 1024 + (long long)((((((vrow + 96 & 31) >> 3) * 2 + group_pair) * 8 + (vrow + 96 & 7)) * 4 + (vrow + 96 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_110 & 127));
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            __syncthreads();
            int dim_swz0 = vrow ^ group << 3;
            x[0] = v_stage[group * 16 * 128 + dim_swz0];
            x[1] = v_stage[(group * 16 + 1) * 128 + dim_swz0];
            x[2] = v_stage[(group * 16 + 2) * 128 + dim_swz0];
            x[3] = v_stage[(group * 16 + 3) * 128 + dim_swz0];
            x[4] = v_stage[(group * 16 + 4) * 128 + dim_swz0];
            x[5] = v_stage[(group * 16 + 5) * 128 + dim_swz0];
            x[6] = v_stage[(group * 16 + 6) * 128 + dim_swz0];
            x[7] = v_stage[(group * 16 + 7) * 128 + dim_swz0];
            x[8] = v_stage[(group * 16 + 8) * 128 + dim_swz0];
            x[9] = v_stage[(group * 16 + 9) * 128 + dim_swz0];
            x[10] = v_stage[(group * 16 + 10) * 128 + dim_swz0];
            x[11] = v_stage[(group * 16 + 11) * 128 + dim_swz0];
            x[12] = v_stage[(group * 16 + 12) * 128 + dim_swz0];
            x[13] = v_stage[(group * 16 + 13) * 128 + dim_swz0];
            x[14] = v_stage[(group * 16 + 14) * 128 + dim_swz0];
            x[15] = v_stage[(group * 16 + 15) * 128 + dim_swz0];
            float _max_16 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
            uint16_t _e4m3x2_113;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_113) : "f"(0.0f), "f"(_max_16));
            uint8_t _cvt_satfinite_e4m3_f32_112 = (uint8_t)_e4m3x2_113;
            uint32_t _f16x2_decode_114;
            float _fp8_e4m3_decode_111;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_114) : "h"(_e4m3x2_113));
            uint16_t _f16_decode_114 = (uint16_t)_f16x2_decode_114;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_111) : "h"(_f16_decode_114));
            float _fdiv_rn_16 = __fdiv_rn(1.0f, _fp8_e4m3_decode_111);
            x[0] = x[0] * _fdiv_rn_16;
            x[1] = x[1] * _fdiv_rn_16;
            x[2] = x[2] * _fdiv_rn_16;
            x[3] = x[3] * _fdiv_rn_16;
            x[4] = x[4] * _fdiv_rn_16;
            x[5] = x[5] * _fdiv_rn_16;
            x[6] = x[6] * _fdiv_rn_16;
            x[7] = x[7] * _fdiv_rn_16;
            x[8] = x[8] * _fdiv_rn_16;
            x[9] = x[9] * _fdiv_rn_16;
            x[10] = x[10] * _fdiv_rn_16;
            x[11] = x[11] * _fdiv_rn_16;
            x[12] = x[12] * _fdiv_rn_16;
            x[13] = x[13] * _fdiv_rn_16;
            x[14] = x[14] * _fdiv_rn_16;
            x[15] = x[15] * _fdiv_rn_16;
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
            sf_tmp[0] = _max_16;
            long long v_off0 = ((long long)bh * 128 + (long long)vrow) * (long long)(kv_stride >> 1) + (long long)(block * 64 + group * 8);
            *(reinterpret_cast<int*>(vt4 + v_off0) + (0)) = codes[0];
            *(reinterpret_cast<int*>(vt4 + (v_off0 + 4)) + (0)) = codes[1];
            if (group_pair == 0) {
                uint16_t _cvt_satfinite_e4m3x2_f32_115;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_115) : "f"(0.0f), "f"(sf_tmp[0]));
                *reinterpret_cast<unsigned char*>(sfv_lo + (kv_tile * 512 + (long long)(((((vrow & 31) >> 3) * 8 + (vrow & 7)) * 4 + (vrow >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_115 & 127));
            } else {
                uint16_t _cvt_satfinite_e4m3x2_f32_116;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_116) : "f"(0.0f), "f"(sf_tmp[0]));
                *reinterpret_cast<unsigned char*>(sfv_hi + (kv_tile * 512 + (long long)(((((vrow & 31) >> 3) * 8 + (vrow & 7)) * 4 + (vrow >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_116 & 127));
            }
            int dim_swz1 = vrow + 32 ^ group << 3;
            x[0] = v_stage[group * 16 * 128 + dim_swz1];
            x[1] = v_stage[(group * 16 + 1) * 128 + dim_swz1];
            x[2] = v_stage[(group * 16 + 2) * 128 + dim_swz1];
            x[3] = v_stage[(group * 16 + 3) * 128 + dim_swz1];
            x[4] = v_stage[(group * 16 + 4) * 128 + dim_swz1];
            x[5] = v_stage[(group * 16 + 5) * 128 + dim_swz1];
            x[6] = v_stage[(group * 16 + 6) * 128 + dim_swz1];
            x[7] = v_stage[(group * 16 + 7) * 128 + dim_swz1];
            x[8] = v_stage[(group * 16 + 8) * 128 + dim_swz1];
            x[9] = v_stage[(group * 16 + 9) * 128 + dim_swz1];
            x[10] = v_stage[(group * 16 + 10) * 128 + dim_swz1];
            x[11] = v_stage[(group * 16 + 11) * 128 + dim_swz1];
            x[12] = v_stage[(group * 16 + 12) * 128 + dim_swz1];
            x[13] = v_stage[(group * 16 + 13) * 128 + dim_swz1];
            x[14] = v_stage[(group * 16 + 14) * 128 + dim_swz1];
            x[15] = v_stage[(group * 16 + 15) * 128 + dim_swz1];
            float _max_17 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
            uint16_t _e4m3x2_119;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_119) : "f"(0.0f), "f"(_max_17));
            uint8_t _cvt_satfinite_e4m3_f32_118 = (uint8_t)_e4m3x2_119;
            uint32_t _f16x2_decode_120;
            float _fp8_e4m3_decode_117;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_120) : "h"(_e4m3x2_119));
            uint16_t _f16_decode_120 = (uint16_t)_f16x2_decode_120;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_117) : "h"(_f16_decode_120));
            float _fdiv_rn_17 = __fdiv_rn(1.0f, _fp8_e4m3_decode_117);
            x[0] = x[0] * _fdiv_rn_17;
            x[1] = x[1] * _fdiv_rn_17;
            x[2] = x[2] * _fdiv_rn_17;
            x[3] = x[3] * _fdiv_rn_17;
            x[4] = x[4] * _fdiv_rn_17;
            x[5] = x[5] * _fdiv_rn_17;
            x[6] = x[6] * _fdiv_rn_17;
            x[7] = x[7] * _fdiv_rn_17;
            x[8] = x[8] * _fdiv_rn_17;
            x[9] = x[9] * _fdiv_rn_17;
            x[10] = x[10] * _fdiv_rn_17;
            x[11] = x[11] * _fdiv_rn_17;
            x[12] = x[12] * _fdiv_rn_17;
            x[13] = x[13] * _fdiv_rn_17;
            x[14] = x[14] * _fdiv_rn_17;
            x[15] = x[15] * _fdiv_rn_17;
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
            sf_tmp[0] = _max_17;
            long long v_off1 = ((long long)bh * 128 + (long long)(vrow + 32)) * (long long)(kv_stride >> 1) + (long long)(block * 64 + group * 8);
            *(reinterpret_cast<int*>(vt4 + v_off1) + (0)) = codes[0];
            *(reinterpret_cast<int*>(vt4 + (v_off1 + 4)) + (0)) = codes[1];
            if (group_pair == 0) {
                uint16_t _cvt_satfinite_e4m3x2_f32_121;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_121) : "f"(0.0f), "f"(sf_tmp[0]));
                *reinterpret_cast<unsigned char*>(sfv_lo + (kv_tile * 512 + (long long)(((((vrow + 32 & 31) >> 3) * 8 + (vrow + 32 & 7)) * 4 + (vrow + 32 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_121 & 127));
            } else {
                uint16_t _cvt_satfinite_e4m3x2_f32_122;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_122) : "f"(0.0f), "f"(sf_tmp[0]));
                *reinterpret_cast<unsigned char*>(sfv_hi + (kv_tile * 512 + (long long)(((((vrow + 32 & 31) >> 3) * 8 + (vrow + 32 & 7)) * 4 + (vrow + 32 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_122 & 127));
            }
            int dim_swz2 = vrow + 64 ^ group << 3;
            x[0] = v_stage[group * 16 * 128 + dim_swz2];
            x[1] = v_stage[(group * 16 + 1) * 128 + dim_swz2];
            x[2] = v_stage[(group * 16 + 2) * 128 + dim_swz2];
            x[3] = v_stage[(group * 16 + 3) * 128 + dim_swz2];
            x[4] = v_stage[(group * 16 + 4) * 128 + dim_swz2];
            x[5] = v_stage[(group * 16 + 5) * 128 + dim_swz2];
            x[6] = v_stage[(group * 16 + 6) * 128 + dim_swz2];
            x[7] = v_stage[(group * 16 + 7) * 128 + dim_swz2];
            x[8] = v_stage[(group * 16 + 8) * 128 + dim_swz2];
            x[9] = v_stage[(group * 16 + 9) * 128 + dim_swz2];
            x[10] = v_stage[(group * 16 + 10) * 128 + dim_swz2];
            x[11] = v_stage[(group * 16 + 11) * 128 + dim_swz2];
            x[12] = v_stage[(group * 16 + 12) * 128 + dim_swz2];
            x[13] = v_stage[(group * 16 + 13) * 128 + dim_swz2];
            x[14] = v_stage[(group * 16 + 14) * 128 + dim_swz2];
            x[15] = v_stage[(group * 16 + 15) * 128 + dim_swz2];
            float _max_18 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
            uint16_t _e4m3x2_125;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_125) : "f"(0.0f), "f"(_max_18));
            uint8_t _cvt_satfinite_e4m3_f32_124 = (uint8_t)_e4m3x2_125;
            uint32_t _f16x2_decode_126;
            float _fp8_e4m3_decode_123;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_126) : "h"(_e4m3x2_125));
            uint16_t _f16_decode_126 = (uint16_t)_f16x2_decode_126;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_123) : "h"(_f16_decode_126));
            float _fdiv_rn_18 = __fdiv_rn(1.0f, _fp8_e4m3_decode_123);
            x[0] = x[0] * _fdiv_rn_18;
            x[1] = x[1] * _fdiv_rn_18;
            x[2] = x[2] * _fdiv_rn_18;
            x[3] = x[3] * _fdiv_rn_18;
            x[4] = x[4] * _fdiv_rn_18;
            x[5] = x[5] * _fdiv_rn_18;
            x[6] = x[6] * _fdiv_rn_18;
            x[7] = x[7] * _fdiv_rn_18;
            x[8] = x[8] * _fdiv_rn_18;
            x[9] = x[9] * _fdiv_rn_18;
            x[10] = x[10] * _fdiv_rn_18;
            x[11] = x[11] * _fdiv_rn_18;
            x[12] = x[12] * _fdiv_rn_18;
            x[13] = x[13] * _fdiv_rn_18;
            x[14] = x[14] * _fdiv_rn_18;
            x[15] = x[15] * _fdiv_rn_18;
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
            sf_tmp[0] = _max_18;
            long long v_off2 = ((long long)bh * 128 + (long long)(vrow + 64)) * (long long)(kv_stride >> 1) + (long long)(block * 64 + group * 8);
            *(reinterpret_cast<int*>(vt4 + v_off2) + (0)) = codes[0];
            *(reinterpret_cast<int*>(vt4 + (v_off2 + 4)) + (0)) = codes[1];
            if (group_pair == 0) {
                uint16_t _cvt_satfinite_e4m3x2_f32_127;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_127) : "f"(0.0f), "f"(sf_tmp[0]));
                *reinterpret_cast<unsigned char*>(sfv_lo + (kv_tile * 512 + (long long)(((((vrow + 64 & 31) >> 3) * 8 + (vrow + 64 & 7)) * 4 + (vrow + 64 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_127 & 127));
            } else {
                uint16_t _cvt_satfinite_e4m3x2_f32_128;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_128) : "f"(0.0f), "f"(sf_tmp[0]));
                *reinterpret_cast<unsigned char*>(sfv_hi + (kv_tile * 512 + (long long)(((((vrow + 64 & 31) >> 3) * 8 + (vrow + 64 & 7)) * 4 + (vrow + 64 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_128 & 127));
            }
            int dim_swz3 = vrow + 96 ^ group << 3;
            x[0] = v_stage[group * 16 * 128 + dim_swz3];
            x[1] = v_stage[(group * 16 + 1) * 128 + dim_swz3];
            x[2] = v_stage[(group * 16 + 2) * 128 + dim_swz3];
            x[3] = v_stage[(group * 16 + 3) * 128 + dim_swz3];
            x[4] = v_stage[(group * 16 + 4) * 128 + dim_swz3];
            x[5] = v_stage[(group * 16 + 5) * 128 + dim_swz3];
            x[6] = v_stage[(group * 16 + 6) * 128 + dim_swz3];
            x[7] = v_stage[(group * 16 + 7) * 128 + dim_swz3];
            x[8] = v_stage[(group * 16 + 8) * 128 + dim_swz3];
            x[9] = v_stage[(group * 16 + 9) * 128 + dim_swz3];
            x[10] = v_stage[(group * 16 + 10) * 128 + dim_swz3];
            x[11] = v_stage[(group * 16 + 11) * 128 + dim_swz3];
            x[12] = v_stage[(group * 16 + 12) * 128 + dim_swz3];
            x[13] = v_stage[(group * 16 + 13) * 128 + dim_swz3];
            x[14] = v_stage[(group * 16 + 14) * 128 + dim_swz3];
            x[15] = v_stage[(group * 16 + 15) * 128 + dim_swz3];
            float _max_19 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fabsf(x[2])), fabsf(x[3])), fabsf(x[4])), fabsf(x[5])), fabsf(x[6])), fabsf(x[7])), fabsf(x[8])), fabsf(x[9])), fabsf(x[10])), fabsf(x[11])), fabsf(x[12])), fabsf(x[13])), fabsf(x[14])), fabsf(x[15])) * 0.16666667f, 0.001953125f);
            uint16_t _e4m3x2_131;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_131) : "f"(0.0f), "f"(_max_19));
            uint8_t _cvt_satfinite_e4m3_f32_130 = (uint8_t)_e4m3x2_131;
            uint32_t _f16x2_decode_132;
            float _fp8_e4m3_decode_129;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_132) : "h"(_e4m3x2_131));
            uint16_t _f16_decode_132 = (uint16_t)_f16x2_decode_132;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_129) : "h"(_f16_decode_132));
            float _fdiv_rn_19 = __fdiv_rn(1.0f, _fp8_e4m3_decode_129);
            x[0] = x[0] * _fdiv_rn_19;
            x[1] = x[1] * _fdiv_rn_19;
            x[2] = x[2] * _fdiv_rn_19;
            x[3] = x[3] * _fdiv_rn_19;
            x[4] = x[4] * _fdiv_rn_19;
            x[5] = x[5] * _fdiv_rn_19;
            x[6] = x[6] * _fdiv_rn_19;
            x[7] = x[7] * _fdiv_rn_19;
            x[8] = x[8] * _fdiv_rn_19;
            x[9] = x[9] * _fdiv_rn_19;
            x[10] = x[10] * _fdiv_rn_19;
            x[11] = x[11] * _fdiv_rn_19;
            x[12] = x[12] * _fdiv_rn_19;
            x[13] = x[13] * _fdiv_rn_19;
            x[14] = x[14] * _fdiv_rn_19;
            x[15] = x[15] * _fdiv_rn_19;
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
            sf_tmp[0] = _max_19;
            long long v_off3 = ((long long)bh * 128 + (long long)(vrow + 96)) * (long long)(kv_stride >> 1) + (long long)(block * 64 + group * 8);
            *(reinterpret_cast<int*>(vt4 + v_off3) + (0)) = codes[0];
            *(reinterpret_cast<int*>(vt4 + (v_off3 + 4)) + (0)) = codes[1];
            if (group_pair == 0) {
                uint16_t _cvt_satfinite_e4m3x2_f32_133;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_133) : "f"(0.0f), "f"(sf_tmp[0]));
                *reinterpret_cast<unsigned char*>(sfv_lo + (kv_tile * 512 + (long long)(((((vrow + 96 & 31) >> 3) * 8 + (vrow + 96 & 7)) * 4 + (vrow + 96 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_133 & 127));
            } else {
                uint16_t _cvt_satfinite_e4m3x2_f32_134;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_134) : "f"(0.0f), "f"(sf_tmp[0]));
                *reinterpret_cast<unsigned char*>(sfv_hi + (kv_tile * 512 + (long long)(((((vrow + 96 & 31) >> 3) * 8 + (vrow + 96 & 7)) * 4 + (vrow + 96 >> 5)) * 4 + group_lane))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_134 & 127));
            }
        }
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
    }
}

} // extern "C"
