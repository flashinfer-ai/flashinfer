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
#define SMEM_S_STATE_OFF 1024
#define SMEM_S_STATE_STAGE_BYTES 8192
#define SMEM_S_STATE_STRIDE 8192
#define SMEM_S_X_OFF 99328
#define SMEM_S_X_STAGE_BYTES 4096
#define SMEM_S_X_STRIDE 4096
#define SMEM_S_SCALARS_OFF 103424
#define SMEM_S_SCALARS_STAGE_BYTES 160
#define SMEM_S_SCALARS_STRIDE 160
#define SMEM_TOTAL 103680
#define THREADS 288
#define LAUNCH_MIN_BLOCKS 2

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


__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

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


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(288, LAUNCH_MIN_BLOCKS) void
kernel_cake_selective_state_update_690e2f567f0c6c4be78d(const __grid_constant__ CUtensorMap state_tma, __nv_bfloat16* __restrict__ x, float* __restrict__ dt, float* __restrict__ A, __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C, float* __restrict__ D, __nv_bfloat16* __restrict__ z, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ output, long long* __restrict__ state_batch_indices, int batch_size, int nheads, int dim, int dstate, int ngroups, int token_steps, int head_tiles, int dt_softplus, long long pad_slot_id)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define state_full_addr (mbar_base + 0)
    #define state_updated_addr (mbar_base + 96)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* s_state = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int s_state_addr = smem + 1024;
    __nv_bfloat16* s_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int s_x_addr = smem + 99328;
    float* s_scalars = reinterpret_cast<float*>(smem_raw + 103424);
    const int s_scalars_addr = smem + 103424;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // state_full: 12 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // state_updated: 12 barriers, init_count=8
            mbarrier_init(smem + 96, 8);
            mbarrier_init(smem + 104, 8);
            mbarrier_init(smem + 112, 8);
            mbarrier_init(smem + 120, 8);
            mbarrier_init(smem + 128, 8);
            mbarrier_init(smem + 136, 8);
            mbarrier_init(smem + 144, 8);
            mbarrier_init(smem + 152, 8);
            mbarrier_init(smem + 160, 8);
            mbarrier_init(smem + 168, 8);
            mbarrier_init(smem + 176, 8);
            mbarrier_init(smem + 184, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: consumers ----
    if (warp <= 7) {
        { // consumers_main
            int head_tile = 0;
            int batch_group = bid;
            int batch = batch_group / 8;
            int group = batch_group % 8;
            int heads_per_group = 8;
            int first_local_head = head_tile * 8;
            int work_heads = heads_per_group - first_local_head;
            if (work_heads > 8) {
                work_heads = 8;
            }
            int first_head = group * heads_per_group + first_local_head;
            int warp_id_in_role = warp - 0;
            int local_warp = warp_id_in_role;
            int row_in_stage = local_warp * 4 + (lane >> 3);
            int member = lane & 7;
            int token_base = batch;
            long long source_slot = state_batch_indices[batch];
            int pad_tile = 0;
            if (source_slot == pad_slot_id) {
                pad_tile = 1;
            }
            long long bc_index = (long long)((token_base * 8 + group) * 128 + member * 8);
            unsigned int b_carriers[8];
            unsigned int c_carriers[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(B + bc_index);
                uint4* _vdst_0 = reinterpret_cast<uint4*>(&b_carriers[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_vdst_0[_blk].x), "=r"(_vdst_0[_blk].y), "=r"(_vdst_0[_blk].z), "=r"(_vdst_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)) : "memory");
                }
            }
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(B + bc_index + 64);
                uint4* _vdst_1 = reinterpret_cast<uint4*>(&b_carriers[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_vdst_1[_blk].x), "=r"(_vdst_1[_blk].y), "=r"(_vdst_1[_blk].z), "=r"(_vdst_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)) : "memory");
                }
            }
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(C + bc_index);
                uint4* _vdst_2 = reinterpret_cast<uint4*>(&c_carriers[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_vdst_2[_blk].x), "=r"(_vdst_2[_blk].y), "=r"(_vdst_2[_blk].z), "=r"(_vdst_2[_blk].w) : "l"((const void*)(_vptr_2 + _blk)) : "memory");
                }
            }
            {
                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(C + bc_index + 64);
                uint4* _vdst_3 = reinterpret_cast<uint4*>(&c_carriers[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_vdst_3[_blk].x), "=r"(_vdst_3[_blk].y), "=r"(_vdst_3[_blk].z), "=r"(_vdst_3[_blk].w) : "l"((const void*)(_vptr_3 + _blk)) : "memory");
                }
            }
            int x_head = tid / 32;
            int x_step = tid / 16 % 2;
            int x_col = tid % 16 * 8;
            int x_live = 1;
            if (x_head >= work_heads) {
                x_head = 0;
                x_live = 0;
            }
            int x_index = (token_base * 64 + first_head + x_head) * 128 + x_col;
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %0, 0;\n\t"
                "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                "}"
                :: "r"((x_live != 0 && x_step == 0) ? 1 : 0), "r"(s_x_addr + (unsigned int)(((x_head * 2 + x_step) * 128 + x_col) * 2)), "l"(x + x_index));
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %0, 0;\n\t"
                "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                "}"
                :: "r"((x_live != 0 && x_step == 1) ? 1 : 0), "r"(s_x_addr + (unsigned int)(((x_head * 2 + x_step) * 128 + x_col) * 2)), "l"(z + x_index));
            asm volatile("cp.async.commit_group;");
            if (local_warp == 2) {
                int scalar_head = lane / 2;
                int scalar_step = lane % 2;
                if (scalar_head < work_heads) {
                    int head_index = first_head + scalar_head;
                    if (scalar_step < 1) {
                        float dt_value = dt[(token_base + scalar_step) * 64 + head_index];
                        dt_value += dt_bias[head_index];
                        if (dt_softplus != 0) {
                            if (dt_value <= 20.0f) {
                                float _exp_5 = __expf(dt_value);
                                float _log2_4;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_4) : "f"(1.0f + _exp_5));
                                dt_value = _log2_4 * 0.6931472f;
                            }
                        }
                        float a_value = A[head_index];
                        int scalar_slot = (scalar_head * 2 + scalar_step) * 2;
                        s_scalars[scalar_slot] = dt_value;
                        float _exp_6 = expf(a_value * dt_value);
                        s_scalars[scalar_slot + 1] = _exp_6;
                    }
                    if (scalar_step == 0) {
                        s_scalars[32 + scalar_head] = D[head_index];
                    }
                }
            }
            float b_values[2];
            float c_values[2];
            asm volatile("cp.async.wait_group 0;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            #pragma unroll 1
            for (int head_offset = 0; head_offset < work_heads; head_offset++) {
                int head = first_head + head_offset;
                int scalar_base = head_offset * 4;
                float dt_value_0 = s_scalars[scalar_base];
                float decay_0 = s_scalars[scalar_base + 1];
                float d_value = s_scalars[32 + head_offset];
                int x_row = head_offset * 256 + row_in_stage;
                int out_row = (token_base * 64 + head) * 128 + row_in_stage;
                #pragma unroll
                for (int stage = 0; stage < 4; stage++) {
                    int ring_stage = stage + 4 * (int)((unsigned int)head_offset % 3);
                    float x_value_0 = s_x[x_row + stage * 32];
                    unsigned int state_carriers[8];
                    float state_values[16];
                    int _parity_state_full = (int)((unsigned int)head_offset / 3 % 2);
                    mbarrier_wait(state_full_addr + (ring_stage) * 8, _parity_state_full);
                    if (pad_tile == 0) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(0) + 3]))
                            : "r"(s_state_addr + (unsigned int)(ring_stage * 8192) + (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[4])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(4) + 3]))
                            : "r"(s_state_addr + (unsigned int)(ring_stage * 8192) + (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16) + 128));
                    } else {
                        state_carriers[0] = 0;
                        state_carriers[1] = 0;
                        state_carriers[2] = 0;
                        state_carriers[3] = 0;
                        state_carriers[4] = 0;
                        state_carriers[5] = 0;
                        state_carriers[6] = 0;
                        state_carriers[7] = 0;
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&state_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(state_carriers[_pair]) << 16);
                        (&state_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(state_carriers[_pair]) & 0xffff0000u);
                    }
                    float2 decay_pair = make_float2(decay_0, decay_0);
                    float2 dtx_pair = make_float2(dt_value_0 * x_value_0, dt_value_0 * x_value_0);
                    float2 projection_pair = make_float2(0.0f, 0.0f);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&b_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(b_carriers[_pair]) << 16);
                        (&b_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(b_carriers[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&c_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(c_carriers[_pair]) << 16);
                        (&c_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(c_carriers[_pair]) & 0xffff0000u);
                    }
                    float2 _f2_0 = make_float2(state_values[0], state_values[1]);
                    float2 _f2_1 = make_float2(b_values[0], b_values[1]);
                    float2 _f2_2 = make_float2(c_values[0], c_values[1]);
                    float2 next_state_pair = fma_f32x2_rn_ftz(_f2_0, decay_pair, mul_f32x2(_f2_1, dtx_pair));
                    projection_pair = fma_f32x2_rn_ftz(next_state_pair, _f2_2, projection_pair);
                    state_values[0] = next_state_pair.x;
                    state_values[1] = next_state_pair.y;
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&b_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(b_carriers[1 + _pair]) << 16);
                        (&b_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(b_carriers[1 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&c_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(c_carriers[1 + _pair]) << 16);
                        (&c_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(c_carriers[1 + _pair]) & 0xffff0000u);
                    }
                    float2 _f2_3 = make_float2(state_values[2], state_values[3]);
                    float2 _f2_4 = make_float2(b_values[0], b_values[1]);
                    float2 _f2_5 = make_float2(c_values[0], c_values[1]);
                    float2 next_state_pair_0 = fma_f32x2_rn_ftz(_f2_3, decay_pair, mul_f32x2(_f2_4, dtx_pair));
                    projection_pair = fma_f32x2_rn_ftz(next_state_pair_0, _f2_5, projection_pair);
                    state_values[2] = next_state_pair_0.x;
                    state_values[3] = next_state_pair_0.y;
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&b_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(b_carriers[2 + _pair]) << 16);
                        (&b_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(b_carriers[2 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&c_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(c_carriers[2 + _pair]) << 16);
                        (&c_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(c_carriers[2 + _pair]) & 0xffff0000u);
                    }
                    float2 _f2_6 = make_float2(state_values[4], state_values[5]);
                    float2 _f2_7 = make_float2(b_values[0], b_values[1]);
                    float2 _f2_8 = make_float2(c_values[0], c_values[1]);
                    float2 next_state_pair_1 = fma_f32x2_rn_ftz(_f2_6, decay_pair, mul_f32x2(_f2_7, dtx_pair));
                    projection_pair = fma_f32x2_rn_ftz(next_state_pair_1, _f2_8, projection_pair);
                    state_values[4] = next_state_pair_1.x;
                    state_values[5] = next_state_pair_1.y;
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&b_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(b_carriers[3 + _pair]) << 16);
                        (&b_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(b_carriers[3 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&c_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(c_carriers[3 + _pair]) << 16);
                        (&c_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(c_carriers[3 + _pair]) & 0xffff0000u);
                    }
                    float2 _f2_9 = make_float2(state_values[6], state_values[7]);
                    float2 _f2_10 = make_float2(b_values[0], b_values[1]);
                    float2 _f2_11 = make_float2(c_values[0], c_values[1]);
                    float2 next_state_pair_2 = fma_f32x2_rn_ftz(_f2_9, decay_pair, mul_f32x2(_f2_10, dtx_pair));
                    projection_pair = fma_f32x2_rn_ftz(next_state_pair_2, _f2_11, projection_pair);
                    state_values[6] = next_state_pair_2.x;
                    state_values[7] = next_state_pair_2.y;
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&b_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(b_carriers[4 + _pair]) << 16);
                        (&b_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(b_carriers[4 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&c_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(c_carriers[4 + _pair]) << 16);
                        (&c_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(c_carriers[4 + _pair]) & 0xffff0000u);
                    }
                    float2 _f2_12 = make_float2(state_values[8], state_values[9]);
                    float2 _f2_13 = make_float2(b_values[0], b_values[1]);
                    float2 _f2_14 = make_float2(c_values[0], c_values[1]);
                    float2 next_state_pair_3 = fma_f32x2_rn_ftz(_f2_12, decay_pair, mul_f32x2(_f2_13, dtx_pair));
                    projection_pair = fma_f32x2_rn_ftz(next_state_pair_3, _f2_14, projection_pair);
                    state_values[8] = next_state_pair_3.x;
                    state_values[9] = next_state_pair_3.y;
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&b_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(b_carriers[5 + _pair]) << 16);
                        (&b_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(b_carriers[5 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&c_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(c_carriers[5 + _pair]) << 16);
                        (&c_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(c_carriers[5 + _pair]) & 0xffff0000u);
                    }
                    float2 _f2_15 = make_float2(state_values[10], state_values[11]);
                    float2 _f2_16 = make_float2(b_values[0], b_values[1]);
                    float2 _f2_17 = make_float2(c_values[0], c_values[1]);
                    float2 next_state_pair_4 = fma_f32x2_rn_ftz(_f2_15, decay_pair, mul_f32x2(_f2_16, dtx_pair));
                    projection_pair = fma_f32x2_rn_ftz(next_state_pair_4, _f2_17, projection_pair);
                    state_values[10] = next_state_pair_4.x;
                    state_values[11] = next_state_pair_4.y;
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&b_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(b_carriers[6 + _pair]) << 16);
                        (&b_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(b_carriers[6 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&c_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(c_carriers[6 + _pair]) << 16);
                        (&c_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(c_carriers[6 + _pair]) & 0xffff0000u);
                    }
                    float2 _f2_18 = make_float2(state_values[12], state_values[13]);
                    float2 _f2_19 = make_float2(b_values[0], b_values[1]);
                    float2 _f2_20 = make_float2(c_values[0], c_values[1]);
                    float2 next_state_pair_5 = fma_f32x2_rn_ftz(_f2_18, decay_pair, mul_f32x2(_f2_19, dtx_pair));
                    projection_pair = fma_f32x2_rn_ftz(next_state_pair_5, _f2_20, projection_pair);
                    state_values[12] = next_state_pair_5.x;
                    state_values[13] = next_state_pair_5.y;
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&b_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(b_carriers[7 + _pair]) << 16);
                        (&b_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(b_carriers[7 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&c_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(c_carriers[7 + _pair]) << 16);
                        (&c_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(c_carriers[7 + _pair]) & 0xffff0000u);
                    }
                    float2 _f2_21 = make_float2(state_values[14], state_values[15]);
                    float2 _f2_22 = make_float2(b_values[0], b_values[1]);
                    float2 _f2_23 = make_float2(c_values[0], c_values[1]);
                    float2 next_state_pair_6 = fma_f32x2_rn_ftz(_f2_21, decay_pair, mul_f32x2(_f2_22, dtx_pair));
                    projection_pair = fma_f32x2_rn_ftz(next_state_pair_6, _f2_23, projection_pair);
                    state_values[14] = next_state_pair_6.x;
                    state_values[15] = next_state_pair_6.y;
                    unsigned int state_values_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_values[_lp*2 + 0], state_values[_lp*2+1 + 0]));
                        state_values_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + (unsigned int)(ring_stage * 8192) + (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16)), "r"(state_values_bf16[0]), "r"(state_values_bf16[1]), "r"(state_values_bf16[2]), "r"(state_values_bf16[3]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + (unsigned int)(ring_stage * 8192) + (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16) + 128), "r"(state_values_bf16[4]), "r"(state_values_bf16[5]), "r"(state_values_bf16[6]), "r"(state_values_bf16[7]));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(state_updated_addr + (ring_stage) * 8);
                    }
                    float partial = projection_pair.x + projection_pair.y;
                    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, partial, 4);
                    partial += _shfl_xor_sync_7;
                    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, partial, 2);
                    partial += _shfl_xor_sync_8;
                    float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, partial, 1);
                    partial += _shfl_xor_sync_9;
                    if (member == 0) {
                        float result = __fmaf_rn(d_value, x_value_0, partial);
                        float z_value = s_x[x_row + 128 + stage * 32];
                        float _exp_10 = expf(-z_value);
                        float sigmoid_z = 1.0f / (1.0f + _exp_10);
                        result *= z_value * sigmoid_z;
                        output[out_row + stage * 32] = result;
                    }
                }
            }
        }
    }
    // ---- Role: producer ----
    if (warp == 8) {
        { // producer_main
            int head_tile_1 = 0;
            int batch_group_1 = bid;
            int batch_1 = batch_group_1 / 8;
            int group_1 = batch_group_1 % 8;
            int heads_per_group_1 = 8;
            int first_local_head_1 = head_tile_1 * 8;
            int work_heads_1 = heads_per_group_1 - first_local_head_1;
            if (work_heads_1 > 8) {
                work_heads_1 = 8;
            }
            int first_head_1 = group_1 * heads_per_group_1 + first_local_head_1;
            long long source_slot_1 = state_batch_indices[batch_1];
            int slot = (int)source_slot_1;
            int live = 1;
            if (source_slot_1 == pad_slot_id) {
                live = 0;
            }
            if (elect_sync()) {
                if (work_heads_1 > 0) {
                    #pragma unroll
                    for (int stage_1 = 0; stage_1 < 4; stage_1++) {
                        int fill_stage = stage_1;
                        if (live != 0) {
                            tma_4d_gmem2smem(s_state_addr + (unsigned int)(fill_stage * 8192), (&state_tma), 0, stage_1 * 32, first_head_1, slot, state_full_addr + (fill_stage) * 8);
                            mbarrier_arrive_expect_tx(state_full_addr + (fill_stage) * 8, 8192);
                        } else {
                            mbarrier_arrive(state_full_addr + (fill_stage) * 8);
                        }
                    }
                }
                if (work_heads_1 > 1) {
                    #pragma unroll
                    for (int stage_2 = 0; stage_2 < 4; stage_2++) {
                        int fill_stage_1 = 4 + stage_2;
                        if (live != 0) {
                            tma_4d_gmem2smem(s_state_addr + (unsigned int)(fill_stage_1 * 8192), (&state_tma), 0, stage_2 * 32, first_head_1 + 1, slot, state_full_addr + (fill_stage_1) * 8);
                            mbarrier_arrive_expect_tx(state_full_addr + (fill_stage_1) * 8, 8192);
                        } else {
                            mbarrier_arrive(state_full_addr + (fill_stage_1) * 8);
                        }
                    }
                }
                if (work_heads_1 > 2) {
                    #pragma unroll
                    for (int stage_3 = 0; stage_3 < 4; stage_3++) {
                        int fill_stage_2 = 8 + stage_3;
                        if (live != 0) {
                            tma_4d_gmem2smem(s_state_addr + (unsigned int)(fill_stage_2 * 8192), (&state_tma), 0, stage_3 * 32, first_head_1 + 2, slot, state_full_addr + (fill_stage_2) * 8);
                            mbarrier_arrive_expect_tx(state_full_addr + (fill_stage_2) * 8, 8192);
                        } else {
                            mbarrier_arrive(state_full_addr + (fill_stage_2) * 8);
                        }
                    }
                }
                #pragma unroll 1
                for (int head_offset_1 = 0; head_offset_1 < work_heads_1; head_offset_1++) {
                    int head_1 = first_head_1 + head_offset_1;
                    int next_head_valid = ((work_heads_1 > head_offset_1 + 3) ? 1 : 0);
                    #pragma unroll
                    for (int stage_4 = 0; stage_4 < 4; stage_4++) {
                        int ring_stage_1 = stage_4 + 4 * (int)((unsigned int)head_offset_1 % 3);
                        int _parity_state_updated = (int)((unsigned int)head_offset_1 / 3 % 2);
                        mbarrier_wait(state_updated_addr + (ring_stage_1) * 8, _parity_state_updated);
                        if (live != 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_4d((&state_tma), 0, stage_4 * 32, head_1, slot, s_state_addr + (unsigned int)(ring_stage_1 * 8192));
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        if (next_head_valid != 0) {
                            if (live != 0) {
                                tma_4d_gmem2smem(s_state_addr + (unsigned int)(ring_stage_1 * 8192), (&state_tma), 0, stage_4 * 32, head_1 + 3, slot, state_full_addr + (ring_stage_1) * 8);
                                mbarrier_arrive_expect_tx(state_full_addr + (ring_stage_1) * 8, 8192);
                            } else {
                                mbarrier_arrive(state_full_addr + (ring_stage_1) * 8);
                            }
                        }
                    }
                }
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }

    // Cleanup
}

} // extern "C"
