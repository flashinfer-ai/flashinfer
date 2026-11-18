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
#define NUM_STATE_PIPE_STAGES 8
#define NUM_AUX_PIPE_STAGES 2
#define SMEM_S_STATE_OFF 1024
#define SMEM_S_STATE_STAGE_BYTES 8192
#define SMEM_S_STATE_STRIDE 8192
#define SMEM_S_AUX_OFF 66560
#define SMEM_S_AUX_STAGE_BYTES 4160
#define SMEM_S_AUX_STRIDE 4160
#define SMEM_TOTAL 70784
#define THREADS 288
#define LAUNCH_MIN_BLOCKS 3

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
kernel_cake_selective_state_update_a746af67567a508fa3c2(const __grid_constant__ CUtensorMap state_tma, __nv_bfloat16* __restrict__ x, unsigned long long dt_addr, unsigned long long a_addr, __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C, unsigned long long d_addr, __nv_bfloat16* __restrict__ z, unsigned long long dt_bias_addr, __nv_bfloat16* __restrict__ output, unsigned long long state_batch_indices_addr, unsigned long long dst_state_batch_indices_addr, int nheads, int ngroups, int units, long long x_batch_stride, long long b_batch_stride, long long c_batch_stride, long long out_batch_stride, long long dt_batch_stride, int dt_softplus, int has_z, int disable_state_update, long long pad_slot_id)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define state_full_addr (mbar_base + 0)
    #define state_updated_addr (mbar_base + 64)
    #define aux_full_addr (mbar_base + 128)
    #define aux_empty_addr (mbar_base + 144)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* s_state = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int s_state_addr = smem + 1024;
    float* s_aux = reinterpret_cast<float*>(smem_raw + 66560);
    const int s_aux_addr = smem + 66560;

    // Mbarrier init (4 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'state_pipe' ---
            // state_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // state_updated: 8 barriers, init_count=8
            mbarrier_init(smem + 64, 8);
            mbarrier_init(smem + 72, 8);
            mbarrier_init(smem + 80, 8);
            mbarrier_init(smem + 88, 8);
            mbarrier_init(smem + 96, 8);
            mbarrier_init(smem + 104, 8);
            mbarrier_init(smem + 112, 8);
            mbarrier_init(smem + 120, 8);
            // --- pipeline 'aux_pipe' ---
            // aux_full: 2 barriers, init_count=32
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
            // aux_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 144, 256);
            mbarrier_init(smem + 152, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: consumers ----
    if (warp <= 7) {
        { // consumers_main
            int count = (units - bid + num_bids - 1) / num_bids;
            int warp_id_in_role = warp - 0;
            int local_warp = warp_id_in_role;
            int row_in_stage = local_warp * 4 + (lane >> 3);
            int member = lane & 7;
            float b_values[16];
            float c_values[16];
            int ordinal = 0;
            #pragma unroll 1
            for (int pair = 0; pair < (count + 1) / 2; pair++) {
                ordinal = 2 * pair;
                if (ordinal < count) {
                    int batch = (bid + ordinal * num_bids) / nheads;
                    int head = bid + ordinal * num_bids - batch * nheads;
                    long long out_base = (long long)batch * out_batch_stride + (long long)(head * 128 + row_in_stage);
                    int base = (ordinal & 1) * 520;
                    int _parity_aux_full = (int)((unsigned int)pair % 2);
                    mbarrier_wait(aux_full_addr, _parity_aux_full);
                    float pad_tile = s_aux[base + 516];
                    #pragma unroll
                    for (int chunk = 0; chunk < 2; chunk++) {
                        #pragma unroll
                        for (int quad = 0; quad < 2; quad++) {
                            int column = chunk * 64 + member * 8 + quad * 4;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&b_values[chunk * 8 + quad * 4])), "=r"(*reinterpret_cast<uint32_t*>(&b_values[(chunk * 8 + quad * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_values[(chunk * 8 + quad * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_values[(chunk * 8 + quad * 4) + 3]))
                                : "r"(s_aux_addr + (unsigned int)((base + 256 + column) * 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&c_values[chunk * 8 + quad * 4])), "=r"(*reinterpret_cast<uint32_t*>(&c_values[(chunk * 8 + quad * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c_values[(chunk * 8 + quad * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c_values[(chunk * 8 + quad * 4) + 3]))
                                : "r"(s_aux_addr + (unsigned int)((base + 384 + column) * 4)));
                        }
                    }
                    int scalar_base = base + 512;
                    float dt_value = s_aux[scalar_base];
                    float decay = s_aux[scalar_base + 1];
                    float d_value = s_aux[scalar_base + 2];
                    float x_value = s_aux[base + row_in_stage];
                    float z_value = s_aux[base + 128 + row_in_stage];
                    float dtx = dt_value * x_value;
                    float2 decay_pair = make_float2(decay, decay);
                    float2 dtx_pair = make_float2(dtx, dtx);
                    float2 partial_pair = make_float2(0.0f, 0.0f);
                    unsigned int lane_offset = (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16);
                    unsigned int state_carriers[8];
                    float state_values[16];
                    int _parity_state_full = (int)((unsigned int)pair % 2);
                    mbarrier_wait(state_full_addr, _parity_state_full);
                    if (pad_tile == 0.0f) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(0) + 3]))
                            : "r"(s_state_addr + lane_offset));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[4])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(4) + 3]))
                            : "r"(s_state_addr + lane_offset + 128));
                    } else {
                        #pragma unroll
                        for (int carrier = 0; carrier < 8; carrier++) {
                            state_carriers[carrier] = 0;
                        }
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&state_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(state_carriers[_pair]) << 16);
                        (&state_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(state_carriers[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int pair_1 = 0; pair_1 < 8; pair_1++) {
                        float2 state_pair = make_float2(state_values[2 * pair_1], state_values[2 * pair_1 + 1]);
                        float2 b_pair = make_float2(b_values[2 * pair_1], b_values[2 * pair_1 + 1]);
                        float2 c_pair = make_float2(c_values[2 * pair_1], c_values[2 * pair_1 + 1]);
                        float2 dbx_pair;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dbx_pair) : "l"(*(const unsigned long long*)&b_pair), "l"(*(const unsigned long long*)&dtx_pair));
                        state_pair = fma_f32x2_rn_ftz(state_pair, decay_pair, dbx_pair);
                        partial_pair = fma_f32x2_rn_ftz(state_pair, c_pair, partial_pair);
                        state_values[2 * pair_1] = state_pair.x;
                        state_values[2 * pair_1 + 1] = state_pair.y;
                    }
                    unsigned int state_values_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_values[_lp*2 + 0], state_values[_lp*2+1 + 0]));
                        state_values_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + lane_offset), "r"(state_values_bf16[0]), "r"(state_values_bf16[1]), "r"(state_values_bf16[2]), "r"(state_values_bf16[3]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + lane_offset + 128), "r"(state_values_bf16[4]), "r"(state_values_bf16[5]), "r"(state_values_bf16[6]), "r"(state_values_bf16[7]));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(state_updated_addr);
                    }
                    float partial = partial_pair.x + partial_pair.y;
                    float _shfl_xor_sync_0 = __shfl_xor_sync(0xFFFFFFFF, partial, 4);
                    partial += _shfl_xor_sync_0;
                    float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, partial, 2);
                    partial += _shfl_xor_sync_1;
                    float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, partial, 1);
                    partial += _shfl_xor_sync_2;
                    if (member == 0) {
                        float result = __fmaf_rn(d_value, x_value, partial);
                        if (has_z != 0) {
                            float _exp_3 = expf(-z_value);
                            float sigmoid_z = 1.0f / (1.0f + _exp_3);
                            result *= z_value * sigmoid_z;
                        }
                        long long out_index = out_base;
                        output[out_index] = result;
                    }
                    int scalar_base_0 = base + 512;
                    float dt_value_1 = s_aux[scalar_base_0];
                    float decay_2 = s_aux[scalar_base_0 + 1];
                    float d_value_3 = s_aux[scalar_base_0 + 2];
                    float x_value_4 = s_aux[base + 32 + row_in_stage];
                    float z_value_5 = s_aux[base + 128 + 32 + row_in_stage];
                    float dtx_6 = dt_value_1 * x_value_4;
                    float2 decay_pair_7 = make_float2(decay_2, decay_2);
                    float2 dtx_pair_8 = make_float2(dtx_6, dtx_6);
                    float2 partial_pair_9 = make_float2(0.0f, 0.0f);
                    unsigned int lane_offset_10 = (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16);
                    unsigned int state_carriers_11[8];
                    float state_values_12[16];
                    int _parity_state_full_1 = (int)((unsigned int)pair % 2);
                    mbarrier_wait(state_full_addr + 8, _parity_state_full_1);
                    if (pad_tile == 0.0f) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11[(0) + 3]))
                            : "r"(s_state_addr + 8192 + lane_offset_10));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11[(4) + 3]))
                            : "r"(s_state_addr + 8192 + lane_offset_10 + 128));
                    } else {
                        #pragma unroll
                        for (int carrier_1 = 0; carrier_1 < 8; carrier_1++) {
                            state_carriers_11[carrier_1] = 0;
                        }
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&state_values_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(state_carriers_11[_pair]) << 16);
                        (&state_values_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(state_carriers_11[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int pair_1_1 = 0; pair_1_1 < 8; pair_1_1++) {
                        float2 state_pair_1 = make_float2(state_values_12[2 * pair_1_1], state_values_12[2 * pair_1_1 + 1]);
                        float2 b_pair_1 = make_float2(b_values[2 * pair_1_1], b_values[2 * pair_1_1 + 1]);
                        float2 c_pair_1 = make_float2(c_values[2 * pair_1_1], c_values[2 * pair_1_1 + 1]);
                        float2 dbx_pair_1;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dbx_pair_1) : "l"(*(const unsigned long long*)&b_pair_1), "l"(*(const unsigned long long*)&dtx_pair_8));
                        state_pair_1 = fma_f32x2_rn_ftz(state_pair_1, decay_pair_7, dbx_pair_1);
                        partial_pair_9 = fma_f32x2_rn_ftz(state_pair_1, c_pair_1, partial_pair_9);
                        state_values_12[2 * pair_1_1] = state_pair_1.x;
                        state_values_12[2 * pair_1_1 + 1] = state_pair_1.y;
                    }
                    unsigned int state_values_12_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_values_12[_lp*2 + 0], state_values_12[_lp*2+1 + 0]));
                        state_values_12_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + 8192 + lane_offset_10), "r"(state_values_12_bf16[0]), "r"(state_values_12_bf16[1]), "r"(state_values_12_bf16[2]), "r"(state_values_12_bf16[3]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + 8192 + lane_offset_10 + 128), "r"(state_values_12_bf16[4]), "r"(state_values_12_bf16[5]), "r"(state_values_12_bf16[6]), "r"(state_values_12_bf16[7]));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(state_updated_addr + 8);
                    }
                    float partial_13 = partial_pair_9.x + partial_pair_9.y;
                    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, partial_13, 4);
                    partial_13 += _shfl_xor_sync_4;
                    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, partial_13, 2);
                    partial_13 += _shfl_xor_sync_5;
                    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, partial_13, 1);
                    partial_13 += _shfl_xor_sync_6;
                    if (member == 0) {
                        float result_1 = __fmaf_rn(d_value_3, x_value_4, partial_13);
                        if (has_z != 0) {
                            float _exp_7 = expf(-z_value_5);
                            float sigmoid_z_1 = 1.0f / (1.0f + _exp_7);
                            result_1 *= z_value_5 * sigmoid_z_1;
                        }
                        long long out_index_1 = out_base + 32;
                        output[out_index_1] = result_1;
                    }
                    int scalar_base_14 = base + 512;
                    float dt_value_15 = s_aux[scalar_base_14];
                    float decay_16 = s_aux[scalar_base_14 + 1];
                    float d_value_17 = s_aux[scalar_base_14 + 2];
                    float x_value_18 = s_aux[base + 64 + row_in_stage];
                    float z_value_19 = s_aux[base + 128 + 64 + row_in_stage];
                    float dtx_20 = dt_value_15 * x_value_18;
                    float2 decay_pair_21 = make_float2(decay_16, decay_16);
                    float2 dtx_pair_22 = make_float2(dtx_20, dtx_20);
                    float2 partial_pair_23 = make_float2(0.0f, 0.0f);
                    unsigned int lane_offset_24 = (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16);
                    unsigned int state_carriers_25[8];
                    float state_values_26[16];
                    int _parity_state_full_2 = (int)((unsigned int)pair % 2);
                    mbarrier_wait(state_full_addr + 16, _parity_state_full_2);
                    if (pad_tile == 0.0f) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25[(0) + 3]))
                            : "r"(s_state_addr + 16384 + lane_offset_24));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25[4])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25[(4) + 3]))
                            : "r"(s_state_addr + 16384 + lane_offset_24 + 128));
                    } else {
                        #pragma unroll
                        for (int carrier_2 = 0; carrier_2 < 8; carrier_2++) {
                            state_carriers_25[carrier_2] = 0;
                        }
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&state_values_26[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(state_carriers_25[_pair]) << 16);
                        (&state_values_26[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(state_carriers_25[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int pair_1_2 = 0; pair_1_2 < 8; pair_1_2++) {
                        float2 state_pair_2 = make_float2(state_values_26[2 * pair_1_2], state_values_26[2 * pair_1_2 + 1]);
                        float2 b_pair_2 = make_float2(b_values[2 * pair_1_2], b_values[2 * pair_1_2 + 1]);
                        float2 c_pair_2 = make_float2(c_values[2 * pair_1_2], c_values[2 * pair_1_2 + 1]);
                        float2 dbx_pair_2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dbx_pair_2) : "l"(*(const unsigned long long*)&b_pair_2), "l"(*(const unsigned long long*)&dtx_pair_22));
                        state_pair_2 = fma_f32x2_rn_ftz(state_pair_2, decay_pair_21, dbx_pair_2);
                        partial_pair_23 = fma_f32x2_rn_ftz(state_pair_2, c_pair_2, partial_pair_23);
                        state_values_26[2 * pair_1_2] = state_pair_2.x;
                        state_values_26[2 * pair_1_2 + 1] = state_pair_2.y;
                    }
                    unsigned int state_values_26_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_values_26[_lp*2 + 0], state_values_26[_lp*2+1 + 0]));
                        state_values_26_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + 16384 + lane_offset_24), "r"(state_values_26_bf16[0]), "r"(state_values_26_bf16[1]), "r"(state_values_26_bf16[2]), "r"(state_values_26_bf16[3]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + 16384 + lane_offset_24 + 128), "r"(state_values_26_bf16[4]), "r"(state_values_26_bf16[5]), "r"(state_values_26_bf16[6]), "r"(state_values_26_bf16[7]));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(state_updated_addr + 16);
                    }
                    float partial_27 = partial_pair_23.x + partial_pair_23.y;
                    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, partial_27, 4);
                    partial_27 += _shfl_xor_sync_8;
                    float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, partial_27, 2);
                    partial_27 += _shfl_xor_sync_9;
                    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, partial_27, 1);
                    partial_27 += _shfl_xor_sync_10;
                    if (member == 0) {
                        float result_2 = __fmaf_rn(d_value_17, x_value_18, partial_27);
                        if (has_z != 0) {
                            float _exp_11 = expf(-z_value_19);
                            float sigmoid_z_2 = 1.0f / (1.0f + _exp_11);
                            result_2 *= z_value_19 * sigmoid_z_2;
                        }
                        long long out_index_2 = out_base + 64;
                        output[out_index_2] = result_2;
                    }
                    int scalar_base_28 = base + 512;
                    float dt_value_29 = s_aux[scalar_base_28];
                    float decay_30 = s_aux[scalar_base_28 + 1];
                    float d_value_31 = s_aux[scalar_base_28 + 2];
                    float x_value_32 = s_aux[base + 96 + row_in_stage];
                    float z_value_33 = s_aux[base + 128 + 96 + row_in_stage];
                    float dtx_34 = dt_value_29 * x_value_32;
                    float2 decay_pair_35 = make_float2(decay_30, decay_30);
                    float2 dtx_pair_36 = make_float2(dtx_34, dtx_34);
                    float2 partial_pair_37 = make_float2(0.0f, 0.0f);
                    unsigned int lane_offset_38 = (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16);
                    unsigned int state_carriers_39[8];
                    float state_values_40[16];
                    int _parity_state_full_3 = (int)((unsigned int)pair % 2);
                    mbarrier_wait(state_full_addr + 24, _parity_state_full_3);
                    if (pad_tile == 0.0f) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39[(0) + 3]))
                            : "r"(s_state_addr + 24576 + lane_offset_38));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39[4])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39[(4) + 3]))
                            : "r"(s_state_addr + 24576 + lane_offset_38 + 128));
                    } else {
                        #pragma unroll
                        for (int carrier_3 = 0; carrier_3 < 8; carrier_3++) {
                            state_carriers_39[carrier_3] = 0;
                        }
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&state_values_40[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(state_carriers_39[_pair]) << 16);
                        (&state_values_40[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(state_carriers_39[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int pair_1_3 = 0; pair_1_3 < 8; pair_1_3++) {
                        float2 state_pair_3 = make_float2(state_values_40[2 * pair_1_3], state_values_40[2 * pair_1_3 + 1]);
                        float2 b_pair_3 = make_float2(b_values[2 * pair_1_3], b_values[2 * pair_1_3 + 1]);
                        float2 c_pair_3 = make_float2(c_values[2 * pair_1_3], c_values[2 * pair_1_3 + 1]);
                        float2 dbx_pair_3;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dbx_pair_3) : "l"(*(const unsigned long long*)&b_pair_3), "l"(*(const unsigned long long*)&dtx_pair_36));
                        state_pair_3 = fma_f32x2_rn_ftz(state_pair_3, decay_pair_35, dbx_pair_3);
                        partial_pair_37 = fma_f32x2_rn_ftz(state_pair_3, c_pair_3, partial_pair_37);
                        state_values_40[2 * pair_1_3] = state_pair_3.x;
                        state_values_40[2 * pair_1_3 + 1] = state_pair_3.y;
                    }
                    unsigned int state_values_40_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_values_40[_lp*2 + 0], state_values_40[_lp*2+1 + 0]));
                        state_values_40_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + 24576 + lane_offset_38), "r"(state_values_40_bf16[0]), "r"(state_values_40_bf16[1]), "r"(state_values_40_bf16[2]), "r"(state_values_40_bf16[3]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + 24576 + lane_offset_38 + 128), "r"(state_values_40_bf16[4]), "r"(state_values_40_bf16[5]), "r"(state_values_40_bf16[6]), "r"(state_values_40_bf16[7]));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(state_updated_addr + 24);
                    }
                    float partial_41 = partial_pair_37.x + partial_pair_37.y;
                    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, partial_41, 4);
                    partial_41 += _shfl_xor_sync_12;
                    float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, partial_41, 2);
                    partial_41 += _shfl_xor_sync_13;
                    float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, partial_41, 1);
                    partial_41 += _shfl_xor_sync_14;
                    if (member == 0) {
                        float result_3 = __fmaf_rn(d_value_31, x_value_32, partial_41);
                        if (has_z != 0) {
                            float _exp_15 = expf(-z_value_33);
                            float sigmoid_z_3 = 1.0f / (1.0f + _exp_15);
                            result_3 *= z_value_33 * sigmoid_z_3;
                        }
                        long long out_index_3 = out_base + 96;
                        output[out_index_3] = result_3;
                    }
                    mbarrier_arrive(aux_empty_addr);
                }
                ordinal = 2 * pair + 1;
                if (ordinal < count) {
                    int batch_1 = (bid + ordinal * num_bids) / nheads;
                    int head_1 = bid + ordinal * num_bids - batch_1 * nheads;
                    long long out_base_1 = (long long)batch_1 * out_batch_stride + (long long)(head_1 * 128 + row_in_stage);
                    int base_1 = (ordinal & 1) * 520;
                    int _parity_aux_full_1 = (int)((unsigned int)pair % 2);
                    mbarrier_wait(aux_full_addr + 8, _parity_aux_full_1);
                    float pad_tile_1 = s_aux[base_1 + 516];
                    #pragma unroll
                    for (int chunk_1 = 0; chunk_1 < 2; chunk_1++) {
                        #pragma unroll
                        for (int quad_1 = 0; quad_1 < 2; quad_1++) {
                            int column_1 = chunk_1 * 64 + member * 8 + quad_1 * 4;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&b_values[chunk_1 * 8 + quad_1 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&b_values[(chunk_1 * 8 + quad_1 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_values[(chunk_1 * 8 + quad_1 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_values[(chunk_1 * 8 + quad_1 * 4) + 3]))
                                : "r"(s_aux_addr + (unsigned int)((base_1 + 256 + column_1) * 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&c_values[chunk_1 * 8 + quad_1 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&c_values[(chunk_1 * 8 + quad_1 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c_values[(chunk_1 * 8 + quad_1 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c_values[(chunk_1 * 8 + quad_1 * 4) + 3]))
                                : "r"(s_aux_addr + (unsigned int)((base_1 + 384 + column_1) * 4)));
                        }
                    }
                    int scalar_base_1 = base_1 + 512;
                    float dt_value_2 = s_aux[scalar_base_1];
                    float decay_1 = s_aux[scalar_base_1 + 1];
                    float d_value_1 = s_aux[scalar_base_1 + 2];
                    float x_value_1 = s_aux[base_1 + row_in_stage];
                    float z_value_1 = s_aux[base_1 + 128 + row_in_stage];
                    float dtx_1 = dt_value_2 * x_value_1;
                    float2 decay_pair_1 = make_float2(decay_1, decay_1);
                    float2 dtx_pair_1 = make_float2(dtx_1, dtx_1);
                    float2 partial_pair_1 = make_float2(0.0f, 0.0f);
                    unsigned int lane_offset_1 = (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16);
                    unsigned int state_carriers_1[8];
                    float state_values_1[16];
                    int _parity_state_full_4 = (int)((unsigned int)pair % 2);
                    mbarrier_wait(state_full_addr + 32, _parity_state_full_4);
                    if (pad_tile_1 == 0.0f) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_1[(0) + 3]))
                            : "r"(s_state_addr + 32768 + lane_offset_1));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_1[(4) + 3]))
                            : "r"(s_state_addr + 32768 + lane_offset_1 + 128));
                    } else {
                        #pragma unroll
                        for (int carrier_4 = 0; carrier_4 < 8; carrier_4++) {
                            state_carriers_1[carrier_4] = 0;
                        }
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&state_values_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(state_carriers_1[_pair]) << 16);
                        (&state_values_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(state_carriers_1[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int pair_1_4 = 0; pair_1_4 < 8; pair_1_4++) {
                        float2 state_pair_4 = make_float2(state_values_1[2 * pair_1_4], state_values_1[2 * pair_1_4 + 1]);
                        float2 b_pair_4 = make_float2(b_values[2 * pair_1_4], b_values[2 * pair_1_4 + 1]);
                        float2 c_pair_4 = make_float2(c_values[2 * pair_1_4], c_values[2 * pair_1_4 + 1]);
                        float2 dbx_pair_4;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dbx_pair_4) : "l"(*(const unsigned long long*)&b_pair_4), "l"(*(const unsigned long long*)&dtx_pair_1));
                        state_pair_4 = fma_f32x2_rn_ftz(state_pair_4, decay_pair_1, dbx_pair_4);
                        partial_pair_1 = fma_f32x2_rn_ftz(state_pair_4, c_pair_4, partial_pair_1);
                        state_values_1[2 * pair_1_4] = state_pair_4.x;
                        state_values_1[2 * pair_1_4 + 1] = state_pair_4.y;
                    }
                    unsigned int state_values_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_values_1[_lp*2 + 0], state_values_1[_lp*2+1 + 0]));
                        state_values_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + 32768 + lane_offset_1), "r"(state_values_bf16_1[0]), "r"(state_values_bf16_1[1]), "r"(state_values_bf16_1[2]), "r"(state_values_bf16_1[3]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + 32768 + lane_offset_1 + 128), "r"(state_values_bf16_1[4]), "r"(state_values_bf16_1[5]), "r"(state_values_bf16_1[6]), "r"(state_values_bf16_1[7]));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(state_updated_addr + 32);
                    }
                    float partial_1 = partial_pair_1.x + partial_pair_1.y;
                    float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, partial_1, 4);
                    partial_1 += _shfl_xor_sync_16;
                    float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, partial_1, 2);
                    partial_1 += _shfl_xor_sync_17;
                    float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, partial_1, 1);
                    partial_1 += _shfl_xor_sync_18;
                    if (member == 0) {
                        float result_4 = __fmaf_rn(d_value_1, x_value_1, partial_1);
                        if (has_z != 0) {
                            float _exp_19 = expf(-z_value_1);
                            float sigmoid_z_4 = 1.0f / (1.0f + _exp_19);
                            result_4 *= z_value_1 * sigmoid_z_4;
                        }
                        long long out_index_4 = out_base_1;
                        output[out_index_4] = result_4;
                    }
                    int scalar_base_0_1 = base_1 + 512;
                    float dt_value_1_1 = s_aux[scalar_base_0_1];
                    float decay_2_1 = s_aux[scalar_base_0_1 + 1];
                    float d_value_3_1 = s_aux[scalar_base_0_1 + 2];
                    float x_value_4_1 = s_aux[base_1 + 32 + row_in_stage];
                    float z_value_5_1 = s_aux[base_1 + 128 + 32 + row_in_stage];
                    float dtx_6_1 = dt_value_1_1 * x_value_4_1;
                    float2 decay_pair_7_1 = make_float2(decay_2_1, decay_2_1);
                    float2 dtx_pair_8_1 = make_float2(dtx_6_1, dtx_6_1);
                    float2 partial_pair_9_1 = make_float2(0.0f, 0.0f);
                    unsigned int lane_offset_10_1 = (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16);
                    unsigned int state_carriers_11_1[8];
                    float state_values_12_1[16];
                    int _parity_state_full_5 = (int)((unsigned int)pair % 2);
                    mbarrier_wait(state_full_addr + 40, _parity_state_full_5);
                    if (pad_tile_1 == 0.0f) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11_1[(0) + 3]))
                            : "r"(s_state_addr + 40960 + lane_offset_10_1));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_11_1[(4) + 3]))
                            : "r"(s_state_addr + 40960 + lane_offset_10_1 + 128));
                    } else {
                        #pragma unroll
                        for (int carrier_5 = 0; carrier_5 < 8; carrier_5++) {
                            state_carriers_11_1[carrier_5] = 0;
                        }
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&state_values_12_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(state_carriers_11_1[_pair]) << 16);
                        (&state_values_12_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(state_carriers_11_1[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int pair_1_5 = 0; pair_1_5 < 8; pair_1_5++) {
                        float2 state_pair_5 = make_float2(state_values_12_1[2 * pair_1_5], state_values_12_1[2 * pair_1_5 + 1]);
                        float2 b_pair_5 = make_float2(b_values[2 * pair_1_5], b_values[2 * pair_1_5 + 1]);
                        float2 c_pair_5 = make_float2(c_values[2 * pair_1_5], c_values[2 * pair_1_5 + 1]);
                        float2 dbx_pair_5;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dbx_pair_5) : "l"(*(const unsigned long long*)&b_pair_5), "l"(*(const unsigned long long*)&dtx_pair_8_1));
                        state_pair_5 = fma_f32x2_rn_ftz(state_pair_5, decay_pair_7_1, dbx_pair_5);
                        partial_pair_9_1 = fma_f32x2_rn_ftz(state_pair_5, c_pair_5, partial_pair_9_1);
                        state_values_12_1[2 * pair_1_5] = state_pair_5.x;
                        state_values_12_1[2 * pair_1_5 + 1] = state_pair_5.y;
                    }
                    unsigned int state_values_12_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_values_12_1[_lp*2 + 0], state_values_12_1[_lp*2+1 + 0]));
                        state_values_12_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + 40960 + lane_offset_10_1), "r"(state_values_12_bf16_1[0]), "r"(state_values_12_bf16_1[1]), "r"(state_values_12_bf16_1[2]), "r"(state_values_12_bf16_1[3]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + 40960 + lane_offset_10_1 + 128), "r"(state_values_12_bf16_1[4]), "r"(state_values_12_bf16_1[5]), "r"(state_values_12_bf16_1[6]), "r"(state_values_12_bf16_1[7]));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(state_updated_addr + 40);
                    }
                    float partial_13_1 = partial_pair_9_1.x + partial_pair_9_1.y;
                    float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, partial_13_1, 4);
                    partial_13_1 += _shfl_xor_sync_20;
                    float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, partial_13_1, 2);
                    partial_13_1 += _shfl_xor_sync_21;
                    float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, partial_13_1, 1);
                    partial_13_1 += _shfl_xor_sync_22;
                    if (member == 0) {
                        float result_5 = __fmaf_rn(d_value_3_1, x_value_4_1, partial_13_1);
                        if (has_z != 0) {
                            float _exp_23 = expf(-z_value_5_1);
                            float sigmoid_z_5 = 1.0f / (1.0f + _exp_23);
                            result_5 *= z_value_5_1 * sigmoid_z_5;
                        }
                        long long out_index_5 = out_base_1 + 32;
                        output[out_index_5] = result_5;
                    }
                    int scalar_base_14_1 = base_1 + 512;
                    float dt_value_15_1 = s_aux[scalar_base_14_1];
                    float decay_16_1 = s_aux[scalar_base_14_1 + 1];
                    float d_value_17_1 = s_aux[scalar_base_14_1 + 2];
                    float x_value_18_1 = s_aux[base_1 + 64 + row_in_stage];
                    float z_value_19_1 = s_aux[base_1 + 128 + 64 + row_in_stage];
                    float dtx_20_1 = dt_value_15_1 * x_value_18_1;
                    float2 decay_pair_21_1 = make_float2(decay_16_1, decay_16_1);
                    float2 dtx_pair_22_1 = make_float2(dtx_20_1, dtx_20_1);
                    float2 partial_pair_23_1 = make_float2(0.0f, 0.0f);
                    unsigned int lane_offset_24_1 = (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16);
                    unsigned int state_carriers_25_1[8];
                    float state_values_26_1[16];
                    int _parity_state_full_6 = (int)((unsigned int)pair % 2);
                    mbarrier_wait(state_full_addr + 48, _parity_state_full_6);
                    if (pad_tile_1 == 0.0f) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25_1[(0) + 3]))
                            : "r"(s_state_addr + 49152 + lane_offset_24_1));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_25_1[(4) + 3]))
                            : "r"(s_state_addr + 49152 + lane_offset_24_1 + 128));
                    } else {
                        #pragma unroll
                        for (int carrier_6 = 0; carrier_6 < 8; carrier_6++) {
                            state_carriers_25_1[carrier_6] = 0;
                        }
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&state_values_26_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(state_carriers_25_1[_pair]) << 16);
                        (&state_values_26_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(state_carriers_25_1[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int pair_1_6 = 0; pair_1_6 < 8; pair_1_6++) {
                        float2 state_pair_6 = make_float2(state_values_26_1[2 * pair_1_6], state_values_26_1[2 * pair_1_6 + 1]);
                        float2 b_pair_6 = make_float2(b_values[2 * pair_1_6], b_values[2 * pair_1_6 + 1]);
                        float2 c_pair_6 = make_float2(c_values[2 * pair_1_6], c_values[2 * pair_1_6 + 1]);
                        float2 dbx_pair_6;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dbx_pair_6) : "l"(*(const unsigned long long*)&b_pair_6), "l"(*(const unsigned long long*)&dtx_pair_22_1));
                        state_pair_6 = fma_f32x2_rn_ftz(state_pair_6, decay_pair_21_1, dbx_pair_6);
                        partial_pair_23_1 = fma_f32x2_rn_ftz(state_pair_6, c_pair_6, partial_pair_23_1);
                        state_values_26_1[2 * pair_1_6] = state_pair_6.x;
                        state_values_26_1[2 * pair_1_6 + 1] = state_pair_6.y;
                    }
                    unsigned int state_values_26_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_values_26_1[_lp*2 + 0], state_values_26_1[_lp*2+1 + 0]));
                        state_values_26_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + 49152 + lane_offset_24_1), "r"(state_values_26_bf16_1[0]), "r"(state_values_26_bf16_1[1]), "r"(state_values_26_bf16_1[2]), "r"(state_values_26_bf16_1[3]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + 49152 + lane_offset_24_1 + 128), "r"(state_values_26_bf16_1[4]), "r"(state_values_26_bf16_1[5]), "r"(state_values_26_bf16_1[6]), "r"(state_values_26_bf16_1[7]));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(state_updated_addr + 48);
                    }
                    float partial_27_1 = partial_pair_23_1.x + partial_pair_23_1.y;
                    float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, partial_27_1, 4);
                    partial_27_1 += _shfl_xor_sync_24;
                    float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, partial_27_1, 2);
                    partial_27_1 += _shfl_xor_sync_25;
                    float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, partial_27_1, 1);
                    partial_27_1 += _shfl_xor_sync_26;
                    if (member == 0) {
                        float result_6 = __fmaf_rn(d_value_17_1, x_value_18_1, partial_27_1);
                        if (has_z != 0) {
                            float _exp_27 = expf(-z_value_19_1);
                            float sigmoid_z_6 = 1.0f / (1.0f + _exp_27);
                            result_6 *= z_value_19_1 * sigmoid_z_6;
                        }
                        long long out_index_6 = out_base_1 + 64;
                        output[out_index_6] = result_6;
                    }
                    int scalar_base_28_1 = base_1 + 512;
                    float dt_value_29_1 = s_aux[scalar_base_28_1];
                    float decay_30_1 = s_aux[scalar_base_28_1 + 1];
                    float d_value_31_1 = s_aux[scalar_base_28_1 + 2];
                    float x_value_32_1 = s_aux[base_1 + 96 + row_in_stage];
                    float z_value_33_1 = s_aux[base_1 + 128 + 96 + row_in_stage];
                    float dtx_34_1 = dt_value_29_1 * x_value_32_1;
                    float2 decay_pair_35_1 = make_float2(decay_30_1, decay_30_1);
                    float2 dtx_pair_36_1 = make_float2(dtx_34_1, dtx_34_1);
                    float2 partial_pair_37_1 = make_float2(0.0f, 0.0f);
                    unsigned int lane_offset_38_1 = (unsigned int)(row_in_stage * 256) + (unsigned int)(member * 16);
                    unsigned int state_carriers_39_1[8];
                    float state_values_40_1[16];
                    int _parity_state_full_7 = (int)((unsigned int)pair % 2);
                    mbarrier_wait(state_full_addr + 56, _parity_state_full_7);
                    if (pad_tile_1 == 0.0f) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39_1[(0) + 3]))
                            : "r"(s_state_addr + 57344 + lane_offset_38_1));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers_39_1[(4) + 3]))
                            : "r"(s_state_addr + 57344 + lane_offset_38_1 + 128));
                    } else {
                        #pragma unroll
                        for (int carrier_7 = 0; carrier_7 < 8; carrier_7++) {
                            state_carriers_39_1[carrier_7] = 0;
                        }
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&state_values_40_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(state_carriers_39_1[_pair]) << 16);
                        (&state_values_40_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(state_carriers_39_1[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int pair_1_7 = 0; pair_1_7 < 8; pair_1_7++) {
                        float2 state_pair_7 = make_float2(state_values_40_1[2 * pair_1_7], state_values_40_1[2 * pair_1_7 + 1]);
                        float2 b_pair_7 = make_float2(b_values[2 * pair_1_7], b_values[2 * pair_1_7 + 1]);
                        float2 c_pair_7 = make_float2(c_values[2 * pair_1_7], c_values[2 * pair_1_7 + 1]);
                        float2 dbx_pair_7;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dbx_pair_7) : "l"(*(const unsigned long long*)&b_pair_7), "l"(*(const unsigned long long*)&dtx_pair_36_1));
                        state_pair_7 = fma_f32x2_rn_ftz(state_pair_7, decay_pair_35_1, dbx_pair_7);
                        partial_pair_37_1 = fma_f32x2_rn_ftz(state_pair_7, c_pair_7, partial_pair_37_1);
                        state_values_40_1[2 * pair_1_7] = state_pair_7.x;
                        state_values_40_1[2 * pair_1_7 + 1] = state_pair_7.y;
                    }
                    unsigned int state_values_40_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_values_40_1[_lp*2 + 0], state_values_40_1[_lp*2+1 + 0]));
                        state_values_40_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + 57344 + lane_offset_38_1), "r"(state_values_40_bf16_1[0]), "r"(state_values_40_bf16_1[1]), "r"(state_values_40_bf16_1[2]), "r"(state_values_40_bf16_1[3]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_state_addr + 57344 + lane_offset_38_1 + 128), "r"(state_values_40_bf16_1[4]), "r"(state_values_40_bf16_1[5]), "r"(state_values_40_bf16_1[6]), "r"(state_values_40_bf16_1[7]));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(state_updated_addr + 56);
                    }
                    float partial_41_1 = partial_pair_37_1.x + partial_pair_37_1.y;
                    float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, partial_41_1, 4);
                    partial_41_1 += _shfl_xor_sync_28;
                    float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, partial_41_1, 2);
                    partial_41_1 += _shfl_xor_sync_29;
                    float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, partial_41_1, 1);
                    partial_41_1 += _shfl_xor_sync_30;
                    if (member == 0) {
                        float result_7 = __fmaf_rn(d_value_31_1, x_value_32_1, partial_41_1);
                        if (has_z != 0) {
                            float _exp_31 = expf(-z_value_33_1);
                            float sigmoid_z_7 = 1.0f / (1.0f + _exp_31);
                            result_7 *= z_value_33_1 * sigmoid_z_7;
                        }
                        long long out_index_7 = out_base_1 + 96;
                        output[out_index_7] = result_7;
                    }
                    mbarrier_arrive(aux_empty_addr + 8);
                }
            }
        }
    }
    // ---- Role: producer ----
    if (warp == 8) {
        { // producer_main
            int count_1 = (units - bid + num_bids - 1) / num_bids;
            int heads_per_group = nheads / ngroups;
            int lane_0 = lane;
            int fetched_head[2];
            long long slots_0[2];
            long long slots_1[2];
            unsigned int x_words_0[2];
            unsigned int x_words_1[2];
            unsigned int z_words_0[2];
            unsigned int z_words_1[2];
            unsigned int b_words_0[2];
            unsigned int b_words_1[2];
            unsigned int c_words_0[2];
            unsigned int c_words_1[2];
            float scalars_0[4];
            float scalars_1[4];
            int flight_head[2];
            long long flight_destination[2];
            int flight_store[2];
            float flight_pad[2];
            int ordinal_1 = 0;
            int next_live = 0;
            int next_head = 0;
            if (count_1 > 0) {
                int fetch_unit = bid;
                fetched_head[0] = fetch_unit - fetch_unit / nheads * nheads;
                int batch_2 = fetch_unit / nheads;
                int head_2 = fetch_unit - batch_2 * nheads;
                int group = head_2 / heads_per_group;
                long long batch_i64 = (long long)batch_2;
                {
                    asm("ld.global.nc.s64 %0, [%1];" : "=l"(slots_0[0]) : "l"((const void*)(reinterpret_cast<const long long*>(state_batch_indices_addr) + (batch_i64))));
                }
                {
                    asm("ld.global.nc.s64 %0, [%1];" : "=l"(slots_0[1]) : "l"((const void*)(reinterpret_cast<const long long*>(dst_state_batch_indices_addr) + (batch_i64))));
                }
                long long quad_i64 = (long long)(4 * lane_0);
                long long x_row = batch_i64 * x_batch_stride + (long long)(head_2 * 128) + quad_i64;
                long long b_row = batch_i64 * b_batch_stride + (long long)(group * 128) + quad_i64;
                long long c_row = batch_i64 * c_batch_stride + (long long)(group * 128) + quad_i64;
                {
                    uint2 _vld_0;
                    asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                        : "=r"(_vld_0.x), "=r"(_vld_0.y) : "l"((const void*)(x + x_row)) : "memory");
                    uint2* _vdst_0 = reinterpret_cast<uint2*>(&x_words_0[0]);
                    *_vdst_0 = _vld_0;
                }
                if (has_z != 0) {
                    {
                        uint2 _vld_1;
                        asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                            : "=r"(_vld_1.x), "=r"(_vld_1.y) : "l"((const void*)(z + x_row)) : "memory");
                        uint2* _vdst_1 = reinterpret_cast<uint2*>(&z_words_0[0]);
                        *_vdst_1 = _vld_1;
                    }
                }
                {
                    uint2 _vld_2;
                    asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                        : "=r"(_vld_2.x), "=r"(_vld_2.y) : "l"((const void*)(B + b_row)) : "memory");
                    uint2* _vdst_2 = reinterpret_cast<uint2*>(&b_words_0[0]);
                    *_vdst_2 = _vld_2;
                }
                {
                    uint2 _vld_3;
                    asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                        : "=r"(_vld_3.x), "=r"(_vld_3.y) : "l"((const void*)(C + c_row)) : "memory");
                    uint2* _vdst_3 = reinterpret_cast<uint2*>(&c_words_0[0]);
                    *_vdst_3 = _vld_3;
                }
                if (lane_0 == 0) {
                    long long head_i64 = (long long)head_2;
                    long long dt_index = batch_i64 * dt_batch_stride + head_i64;
                    {
                        uint32_t _scalar_bits_4;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_4) : "l"((const void*)(reinterpret_cast<const float*>(dt_addr) + (dt_index))) : "memory");
                        scalars_0[0] = __uint_as_float(_scalar_bits_4);
                    }
                    {
                        uint32_t _scalar_bits_5;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_5) : "l"((const void*)(reinterpret_cast<const float*>(dt_bias_addr) + (head_i64))) : "memory");
                        scalars_0[1] = __uint_as_float(_scalar_bits_5);
                    }
                    {
                        uint32_t _scalar_bits_6;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_6) : "l"((const void*)(reinterpret_cast<const float*>(d_addr) + (head_i64))) : "memory");
                        scalars_0[2] = __uint_as_float(_scalar_bits_6);
                    }
                    {
                        uint32_t _scalar_bits_7;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_7) : "l"((const void*)(reinterpret_cast<const float*>(a_addr) + (head_i64))) : "memory");
                        scalars_0[3] = __uint_as_float(_scalar_bits_7);
                    }
                }
            }
            if (count_1 > 1) {
                int fetch_unit_1 = bid + num_bids;
                fetched_head[1] = fetch_unit_1 - fetch_unit_1 / nheads * nheads;
                int batch_3 = fetch_unit_1 / nheads;
                int head_3 = fetch_unit_1 - batch_3 * nheads;
                int group_1 = head_3 / heads_per_group;
                long long batch_i64_1 = (long long)batch_3;
                {
                    asm("ld.global.nc.s64 %0, [%1];" : "=l"(slots_1[0]) : "l"((const void*)(reinterpret_cast<const long long*>(state_batch_indices_addr) + (batch_i64_1))));
                }
                {
                    asm("ld.global.nc.s64 %0, [%1];" : "=l"(slots_1[1]) : "l"((const void*)(reinterpret_cast<const long long*>(dst_state_batch_indices_addr) + (batch_i64_1))));
                }
                long long quad_i64_1 = (long long)(4 * lane_0);
                long long x_row_1 = batch_i64_1 * x_batch_stride + (long long)(head_3 * 128) + quad_i64_1;
                long long b_row_1 = batch_i64_1 * b_batch_stride + (long long)(group_1 * 128) + quad_i64_1;
                long long c_row_1 = batch_i64_1 * c_batch_stride + (long long)(group_1 * 128) + quad_i64_1;
                {
                    uint2 _vld_8;
                    asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                        : "=r"(_vld_8.x), "=r"(_vld_8.y) : "l"((const void*)(x + x_row_1)) : "memory");
                    uint2* _vdst_8 = reinterpret_cast<uint2*>(&x_words_1[0]);
                    *_vdst_8 = _vld_8;
                }
                if (has_z != 0) {
                    {
                        uint2 _vld_9;
                        asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                            : "=r"(_vld_9.x), "=r"(_vld_9.y) : "l"((const void*)(z + x_row_1)) : "memory");
                        uint2* _vdst_9 = reinterpret_cast<uint2*>(&z_words_1[0]);
                        *_vdst_9 = _vld_9;
                    }
                }
                {
                    uint2 _vld_10;
                    asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                        : "=r"(_vld_10.x), "=r"(_vld_10.y) : "l"((const void*)(B + b_row_1)) : "memory");
                    uint2* _vdst_10 = reinterpret_cast<uint2*>(&b_words_1[0]);
                    *_vdst_10 = _vld_10;
                }
                {
                    uint2 _vld_11;
                    asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                        : "=r"(_vld_11.x), "=r"(_vld_11.y) : "l"((const void*)(C + c_row_1)) : "memory");
                    uint2* _vdst_11 = reinterpret_cast<uint2*>(&c_words_1[0]);
                    *_vdst_11 = _vld_11;
                }
                if (lane_0 == 0) {
                    long long head_i64_1 = (long long)head_3;
                    long long dt_index_1 = batch_i64_1 * dt_batch_stride + head_i64_1;
                    {
                        uint32_t _scalar_bits_12;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_12) : "l"((const void*)(reinterpret_cast<const float*>(dt_addr) + (dt_index_1))) : "memory");
                        scalars_1[0] = __uint_as_float(_scalar_bits_12);
                    }
                    {
                        uint32_t _scalar_bits_13;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_13) : "l"((const void*)(reinterpret_cast<const float*>(dt_bias_addr) + (head_i64_1))) : "memory");
                        scalars_1[1] = __uint_as_float(_scalar_bits_13);
                    }
                    {
                        uint32_t _scalar_bits_14;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_14) : "l"((const void*)(reinterpret_cast<const float*>(d_addr) + (head_i64_1))) : "memory");
                        scalars_1[2] = __uint_as_float(_scalar_bits_14);
                    }
                    {
                        uint32_t _scalar_bits_15;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_15) : "l"((const void*)(reinterpret_cast<const float*>(a_addr) + (head_i64_1))) : "memory");
                        scalars_1[3] = __uint_as_float(_scalar_bits_15);
                    }
                }
            }
            if (count_1 > 0) {
                long long source_slot = 0;
                long long destination_slot = 0;
                source_slot = slots_0[0];
                destination_slot = slots_0[1];
                int load_live = 1;
                int store_live = 1 - disable_state_update;
                float pad = 0.0f;
                if (source_slot == pad_slot_id) {
                    load_live = 0;
                    store_live = 0;
                    pad = 1.0f;
                }
                if (destination_slot == pad_slot_id) {
                    store_live = 0;
                }
                flight_head[0] = fetched_head[0];
                flight_destination[0] = destination_slot;
                flight_store[0] = store_live;
                flight_pad[0] = pad;
                if (elect_sync()) {
                    if (load_live != 0) {
                        tma_4d_gmem2smem(s_state_addr, (&state_tma), 0, 0, fetched_head[0], (int)source_slot, state_full_addr);
                        mbarrier_arrive_expect_tx(state_full_addr, 8192);
                    } else {
                        mbarrier_arrive(state_full_addr);
                    }
                    if (load_live != 0) {
                        tma_4d_gmem2smem(s_state_addr + 8192, (&state_tma), 0, 32, fetched_head[0], (int)source_slot, state_full_addr + 8);
                        mbarrier_arrive_expect_tx(state_full_addr + 8, 8192);
                    } else {
                        mbarrier_arrive(state_full_addr + 8);
                    }
                    if (load_live != 0) {
                        tma_4d_gmem2smem(s_state_addr + 16384, (&state_tma), 0, 64, fetched_head[0], (int)source_slot, state_full_addr + 16);
                        mbarrier_arrive_expect_tx(state_full_addr + 16, 8192);
                    } else {
                        mbarrier_arrive(state_full_addr + 16);
                    }
                    if (load_live != 0) {
                        tma_4d_gmem2smem(s_state_addr + 24576, (&state_tma), 0, 96, fetched_head[0], (int)source_slot, state_full_addr + 24);
                        mbarrier_arrive_expect_tx(state_full_addr + 24, 8192);
                    } else {
                        mbarrier_arrive(state_full_addr + 24);
                    }
                }
                int slot_byte = 4 * lane_0 * 4;
                float values[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_words_0[_pair]) << 16);
                    (&values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_words_0[_pair]) & 0xffff0000u);
                }
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_aux_addr + (unsigned int)slot_byte), "f"(values[0]), "f"(values[1]), "f"(values[2]), "f"(values[3]));
                float values_0[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&values_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(z_words_0[_pair]) << 16);
                    (&values_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(z_words_0[_pair]) & 0xffff0000u);
                }
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_aux_addr + (unsigned int)(slot_byte + 512)), "f"(values_0[0]), "f"(values_0[1]), "f"(values_0[2]), "f"(values_0[3]));
                float values_1[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&values_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(b_words_0[_pair]) << 16);
                    (&values_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(b_words_0[_pair]) & 0xffff0000u);
                }
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_aux_addr + (unsigned int)(slot_byte + 1024)), "f"(values_1[0]), "f"(values_1[1]), "f"(values_1[2]), "f"(values_1[3]));
                float values_2[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&values_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(c_words_0[_pair]) << 16);
                    (&values_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(c_words_0[_pair]) & 0xffff0000u);
                }
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_aux_addr + (unsigned int)(slot_byte + 1536)), "f"(values_2[0]), "f"(values_2[1]), "f"(values_2[2]), "f"(values_2[3]));
                if (lane_0 == 0) {
                    float scalar_dt = scalars_0[0] + scalars_0[1];
                    if (dt_softplus != 0) {
                        if (scalar_dt <= 20.0f) {
                            float _exp_17 = __expf(scalar_dt);
                            float _log2_16;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_16) : "f"(1.0f + _exp_17));
                            scalar_dt = _log2_16 * 0.6931472f;
                        }
                    }
                    int scalar_base_2 = 512;
                    s_aux[scalar_base_2] = scalar_dt;
                    float _exp_18 = expf(scalars_0[3] * scalar_dt);
                    s_aux[scalar_base_2 + 1] = _exp_18;
                    s_aux[scalar_base_2 + 2] = scalars_0[2];
                    s_aux[516] = pad;
                }
                mbarrier_arrive(aux_full_addr);
                if (count_1 > 2) {
                    int fetch_unit_2 = bid + 2 * num_bids;
                    fetched_head[0] = fetch_unit_2 - fetch_unit_2 / nheads * nheads;
                    int batch_4 = fetch_unit_2 / nheads;
                    int head_4 = fetch_unit_2 - batch_4 * nheads;
                    int group_2 = head_4 / heads_per_group;
                    long long batch_i64_2 = (long long)batch_4;
                    {
                        asm("ld.global.nc.s64 %0, [%1];" : "=l"(slots_0[0]) : "l"((const void*)(reinterpret_cast<const long long*>(state_batch_indices_addr) + (batch_i64_2))));
                    }
                    {
                        asm("ld.global.nc.s64 %0, [%1];" : "=l"(slots_0[1]) : "l"((const void*)(reinterpret_cast<const long long*>(dst_state_batch_indices_addr) + (batch_i64_2))));
                    }
                    long long quad_i64_2 = (long long)(4 * lane_0);
                    long long x_row_2 = batch_i64_2 * x_batch_stride + (long long)(head_4 * 128) + quad_i64_2;
                    long long b_row_2 = batch_i64_2 * b_batch_stride + (long long)(group_2 * 128) + quad_i64_2;
                    long long c_row_2 = batch_i64_2 * c_batch_stride + (long long)(group_2 * 128) + quad_i64_2;
                    {
                        uint2 _vld_19;
                        asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                            : "=r"(_vld_19.x), "=r"(_vld_19.y) : "l"((const void*)(x + x_row_2)) : "memory");
                        uint2* _vdst_19 = reinterpret_cast<uint2*>(&x_words_0[0]);
                        *_vdst_19 = _vld_19;
                    }
                    if (has_z != 0) {
                        {
                            uint2 _vld_20;
                            asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                                : "=r"(_vld_20.x), "=r"(_vld_20.y) : "l"((const void*)(z + x_row_2)) : "memory");
                            uint2* _vdst_20 = reinterpret_cast<uint2*>(&z_words_0[0]);
                            *_vdst_20 = _vld_20;
                        }
                    }
                    {
                        uint2 _vld_21;
                        asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                            : "=r"(_vld_21.x), "=r"(_vld_21.y) : "l"((const void*)(B + b_row_2)) : "memory");
                        uint2* _vdst_21 = reinterpret_cast<uint2*>(&b_words_0[0]);
                        *_vdst_21 = _vld_21;
                    }
                    {
                        uint2 _vld_22;
                        asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                            : "=r"(_vld_22.x), "=r"(_vld_22.y) : "l"((const void*)(C + c_row_2)) : "memory");
                        uint2* _vdst_22 = reinterpret_cast<uint2*>(&c_words_0[0]);
                        *_vdst_22 = _vld_22;
                    }
                    if (lane_0 == 0) {
                        long long head_i64_2 = (long long)head_4;
                        long long dt_index_2 = batch_i64_2 * dt_batch_stride + head_i64_2;
                        {
                            uint32_t _scalar_bits_23;
                            asm volatile("ld.global.nc.b32 %0, [%1];"
                                : "=r"(_scalar_bits_23) : "l"((const void*)(reinterpret_cast<const float*>(dt_addr) + (dt_index_2))) : "memory");
                            scalars_0[0] = __uint_as_float(_scalar_bits_23);
                        }
                        {
                            uint32_t _scalar_bits_24;
                            asm volatile("ld.global.nc.b32 %0, [%1];"
                                : "=r"(_scalar_bits_24) : "l"((const void*)(reinterpret_cast<const float*>(dt_bias_addr) + (head_i64_2))) : "memory");
                            scalars_0[1] = __uint_as_float(_scalar_bits_24);
                        }
                        {
                            uint32_t _scalar_bits_25;
                            asm volatile("ld.global.nc.b32 %0, [%1];"
                                : "=r"(_scalar_bits_25) : "l"((const void*)(reinterpret_cast<const float*>(d_addr) + (head_i64_2))) : "memory");
                            scalars_0[2] = __uint_as_float(_scalar_bits_25);
                        }
                        {
                            uint32_t _scalar_bits_26;
                            asm volatile("ld.global.nc.b32 %0, [%1];"
                                : "=r"(_scalar_bits_26) : "l"((const void*)(reinterpret_cast<const float*>(a_addr) + (head_i64_2))) : "memory");
                            scalars_0[3] = __uint_as_float(_scalar_bits_26);
                        }
                    }
                }
            }
            if (count_1 > 1) {
                long long source_slot_1 = 0;
                long long destination_slot_1 = 0;
                source_slot_1 = slots_1[0];
                destination_slot_1 = slots_1[1];
                int load_live_1 = 1;
                int store_live_1 = 1 - disable_state_update;
                float pad_1 = 0.0f;
                if (source_slot_1 == pad_slot_id) {
                    load_live_1 = 0;
                    store_live_1 = 0;
                    pad_1 = 1.0f;
                }
                if (destination_slot_1 == pad_slot_id) {
                    store_live_1 = 0;
                }
                flight_head[1] = fetched_head[1];
                flight_destination[1] = destination_slot_1;
                flight_store[1] = store_live_1;
                flight_pad[1] = pad_1;
                if (elect_sync()) {
                    if (load_live_1 != 0) {
                        tma_4d_gmem2smem(s_state_addr + 32768, (&state_tma), 0, 0, fetched_head[1], (int)source_slot_1, state_full_addr + 32);
                        mbarrier_arrive_expect_tx(state_full_addr + 32, 8192);
                    } else {
                        mbarrier_arrive(state_full_addr + 32);
                    }
                    if (load_live_1 != 0) {
                        tma_4d_gmem2smem(s_state_addr + 40960, (&state_tma), 0, 32, fetched_head[1], (int)source_slot_1, state_full_addr + 40);
                        mbarrier_arrive_expect_tx(state_full_addr + 40, 8192);
                    } else {
                        mbarrier_arrive(state_full_addr + 40);
                    }
                    if (load_live_1 != 0) {
                        tma_4d_gmem2smem(s_state_addr + 49152, (&state_tma), 0, 64, fetched_head[1], (int)source_slot_1, state_full_addr + 48);
                        mbarrier_arrive_expect_tx(state_full_addr + 48, 8192);
                    } else {
                        mbarrier_arrive(state_full_addr + 48);
                    }
                    if (load_live_1 != 0) {
                        tma_4d_gmem2smem(s_state_addr + 57344, (&state_tma), 0, 96, fetched_head[1], (int)source_slot_1, state_full_addr + 56);
                        mbarrier_arrive_expect_tx(state_full_addr + 56, 8192);
                    } else {
                        mbarrier_arrive(state_full_addr + 56);
                    }
                }
            }
            #pragma unroll 1
            for (int pair_2 = 0; pair_2 < (count_1 + 1) / 2; pair_2++) {
                ordinal_1 = 2 * pair_2;
                if (ordinal_1 < count_1) {
                    next_live = (int)(count_1 > ordinal_1 + 2);
                    next_head = fetched_head[0];
                    long long source_slot_2 = 0;
                    long long destination_slot_2 = 0;
                    if (next_live != 0) {
                        source_slot_2 = slots_0[0];
                        destination_slot_2 = slots_0[1];
                    }
                    int load_live_2 = next_live;
                    int store_live_2 = next_live * (1 - disable_state_update);
                    float pad_2 = 0.0f;
                    if (source_slot_2 == pad_slot_id) {
                        load_live_2 = 0;
                        store_live_2 = 0;
                        pad_2 = 1.0f;
                    }
                    if (destination_slot_2 == pad_slot_id) {
                        store_live_2 = 0;
                    }
                    if (elect_sync()) {
                        int _parity_state_updated = (int)((unsigned int)pair_2 % 2);
                        mbarrier_wait(state_updated_addr, _parity_state_updated);
                        if (flight_store[0] != 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_4d((&state_tma), 0, 0, flight_head[0], (int)flight_destination[0], s_state_addr);
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        if (next_live != 0) {
                            if (load_live_2 != 0) {
                                tma_4d_gmem2smem(s_state_addr, (&state_tma), 0, 0, next_head, (int)source_slot_2, state_full_addr);
                                mbarrier_arrive_expect_tx(state_full_addr, 8192);
                            } else {
                                mbarrier_arrive(state_full_addr);
                            }
                        }
                    }
                    if (count_1 > ordinal_1 + 1) {
                        if (ordinal_1 + 1 >= 2) {
                            int _parity_aux_empty = (int)((unsigned int)pair_2 % 2) ^ 1;
                            mbarrier_wait(aux_empty_addr + 8, _parity_aux_empty);
                        }
                        int slot_byte_1 = ((ordinal_1 + 1 & 1) * 520 + 4 * lane_0) * 4;
                        float values_3[4];
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            (&values_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_words_1[_pair]) << 16);
                            (&values_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_words_1[_pair]) & 0xffff0000u);
                        }
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_aux_addr + (unsigned int)slot_byte_1), "f"(values_3[0]), "f"(values_3[1]), "f"(values_3[2]), "f"(values_3[3]));
                        float values_0_1[4];
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            (&values_0_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(z_words_1[_pair]) << 16);
                            (&values_0_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(z_words_1[_pair]) & 0xffff0000u);
                        }
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_aux_addr + (unsigned int)(slot_byte_1 + 512)), "f"(values_0_1[0]), "f"(values_0_1[1]), "f"(values_0_1[2]), "f"(values_0_1[3]));
                        float values_1_1[4];
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            (&values_1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(b_words_1[_pair]) << 16);
                            (&values_1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(b_words_1[_pair]) & 0xffff0000u);
                        }
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_aux_addr + (unsigned int)(slot_byte_1 + 1024)), "f"(values_1_1[0]), "f"(values_1_1[1]), "f"(values_1_1[2]), "f"(values_1_1[3]));
                        float values_2_1[4];
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            (&values_2_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(c_words_1[_pair]) << 16);
                            (&values_2_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(c_words_1[_pair]) & 0xffff0000u);
                        }
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_aux_addr + (unsigned int)(slot_byte_1 + 1536)), "f"(values_2_1[0]), "f"(values_2_1[1]), "f"(values_2_1[2]), "f"(values_2_1[3]));
                        if (lane_0 == 0) {
                            float scalar_dt_1 = scalars_1[0] + scalars_1[1];
                            if (dt_softplus != 0) {
                                if (scalar_dt_1 <= 20.0f) {
                                    float _exp_28 = __expf(scalar_dt_1);
                                    float _log2_27;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_27) : "f"(1.0f + _exp_28));
                                    scalar_dt_1 = _log2_27 * 0.6931472f;
                                }
                            }
                            int scalar_base_3 = (ordinal_1 + 1 & 1) * 520 + 512;
                            s_aux[scalar_base_3] = scalar_dt_1;
                            float _exp_29 = expf(scalars_1[3] * scalar_dt_1);
                            s_aux[scalar_base_3 + 1] = _exp_29;
                            s_aux[scalar_base_3 + 2] = scalars_1[2];
                            s_aux[(ordinal_1 + 1 & 1) * 520 + 516] = flight_pad[1];
                        }
                        mbarrier_arrive(aux_full_addr + 8);
                    }
                    if (count_1 > ordinal_1 + 3) {
                        int fetch_unit_3 = bid + (ordinal_1 + 3) * num_bids;
                        fetched_head[1] = fetch_unit_3 - fetch_unit_3 / nheads * nheads;
                        int batch_5 = fetch_unit_3 / nheads;
                        int head_5 = fetch_unit_3 - batch_5 * nheads;
                        int group_3 = head_5 / heads_per_group;
                        long long batch_i64_3 = (long long)batch_5;
                        {
                            asm("ld.global.nc.s64 %0, [%1];" : "=l"(slots_1[0]) : "l"((const void*)(reinterpret_cast<const long long*>(state_batch_indices_addr) + (batch_i64_3))));
                        }
                        {
                            asm("ld.global.nc.s64 %0, [%1];" : "=l"(slots_1[1]) : "l"((const void*)(reinterpret_cast<const long long*>(dst_state_batch_indices_addr) + (batch_i64_3))));
                        }
                        long long quad_i64_3 = (long long)(4 * lane_0);
                        long long x_row_3 = batch_i64_3 * x_batch_stride + (long long)(head_5 * 128) + quad_i64_3;
                        long long b_row_3 = batch_i64_3 * b_batch_stride + (long long)(group_3 * 128) + quad_i64_3;
                        long long c_row_3 = batch_i64_3 * c_batch_stride + (long long)(group_3 * 128) + quad_i64_3;
                        {
                            uint2 _vld_30;
                            asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                                : "=r"(_vld_30.x), "=r"(_vld_30.y) : "l"((const void*)(x + x_row_3)) : "memory");
                            uint2* _vdst_30 = reinterpret_cast<uint2*>(&x_words_1[0]);
                            *_vdst_30 = _vld_30;
                        }
                        if (has_z != 0) {
                            {
                                uint2 _vld_31;
                                asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                                    : "=r"(_vld_31.x), "=r"(_vld_31.y) : "l"((const void*)(z + x_row_3)) : "memory");
                                uint2* _vdst_31 = reinterpret_cast<uint2*>(&z_words_1[0]);
                                *_vdst_31 = _vld_31;
                            }
                        }
                        {
                            uint2 _vld_32;
                            asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                                : "=r"(_vld_32.x), "=r"(_vld_32.y) : "l"((const void*)(B + b_row_3)) : "memory");
                            uint2* _vdst_32 = reinterpret_cast<uint2*>(&b_words_1[0]);
                            *_vdst_32 = _vld_32;
                        }
                        {
                            uint2 _vld_33;
                            asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                                : "=r"(_vld_33.x), "=r"(_vld_33.y) : "l"((const void*)(C + c_row_3)) : "memory");
                            uint2* _vdst_33 = reinterpret_cast<uint2*>(&c_words_1[0]);
                            *_vdst_33 = _vld_33;
                        }
                        if (lane_0 == 0) {
                            long long head_i64_3 = (long long)head_5;
                            long long dt_index_3 = batch_i64_3 * dt_batch_stride + head_i64_3;
                            {
                                uint32_t _scalar_bits_34;
                                asm volatile("ld.global.nc.b32 %0, [%1];"
                                    : "=r"(_scalar_bits_34) : "l"((const void*)(reinterpret_cast<const float*>(dt_addr) + (dt_index_3))) : "memory");
                                scalars_1[0] = __uint_as_float(_scalar_bits_34);
                            }
                            {
                                uint32_t _scalar_bits_35;
                                asm volatile("ld.global.nc.b32 %0, [%1];"
                                    : "=r"(_scalar_bits_35) : "l"((const void*)(reinterpret_cast<const float*>(dt_bias_addr) + (head_i64_3))) : "memory");
                                scalars_1[1] = __uint_as_float(_scalar_bits_35);
                            }
                            {
                                uint32_t _scalar_bits_36;
                                asm volatile("ld.global.nc.b32 %0, [%1];"
                                    : "=r"(_scalar_bits_36) : "l"((const void*)(reinterpret_cast<const float*>(d_addr) + (head_i64_3))) : "memory");
                                scalars_1[2] = __uint_as_float(_scalar_bits_36);
                            }
                            {
                                uint32_t _scalar_bits_37;
                                asm volatile("ld.global.nc.b32 %0, [%1];"
                                    : "=r"(_scalar_bits_37) : "l"((const void*)(reinterpret_cast<const float*>(a_addr) + (head_i64_3))) : "memory");
                                scalars_1[3] = __uint_as_float(_scalar_bits_37);
                            }
                        }
                    }
                    if (elect_sync()) {
                        int _parity_state_updated_1 = (int)((unsigned int)pair_2 % 2);
                        mbarrier_wait(state_updated_addr + 8, _parity_state_updated_1);
                        if (flight_store[0] != 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_4d((&state_tma), 0, 32, flight_head[0], (int)flight_destination[0], s_state_addr + 8192);
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        if (next_live != 0) {
                            if (load_live_2 != 0) {
                                tma_4d_gmem2smem(s_state_addr + 8192, (&state_tma), 0, 32, next_head, (int)source_slot_2, state_full_addr + 8);
                                mbarrier_arrive_expect_tx(state_full_addr + 8, 8192);
                            } else {
                                mbarrier_arrive(state_full_addr + 8);
                            }
                        }
                    }
                    if (elect_sync()) {
                        int _parity_state_updated_2 = (int)((unsigned int)pair_2 % 2);
                        mbarrier_wait(state_updated_addr + 16, _parity_state_updated_2);
                        if (flight_store[0] != 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_4d((&state_tma), 0, 64, flight_head[0], (int)flight_destination[0], s_state_addr + 16384);
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        if (next_live != 0) {
                            if (load_live_2 != 0) {
                                tma_4d_gmem2smem(s_state_addr + 16384, (&state_tma), 0, 64, next_head, (int)source_slot_2, state_full_addr + 16);
                                mbarrier_arrive_expect_tx(state_full_addr + 16, 8192);
                            } else {
                                mbarrier_arrive(state_full_addr + 16);
                            }
                        }
                    }
                    if (elect_sync()) {
                        int _parity_state_updated_3 = (int)((unsigned int)pair_2 % 2);
                        mbarrier_wait(state_updated_addr + 24, _parity_state_updated_3);
                        if (flight_store[0] != 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_4d((&state_tma), 0, 96, flight_head[0], (int)flight_destination[0], s_state_addr + 24576);
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        if (next_live != 0) {
                            if (load_live_2 != 0) {
                                tma_4d_gmem2smem(s_state_addr + 24576, (&state_tma), 0, 96, next_head, (int)source_slot_2, state_full_addr + 24);
                                mbarrier_arrive_expect_tx(state_full_addr + 24, 8192);
                            } else {
                                mbarrier_arrive(state_full_addr + 24);
                            }
                        }
                    }
                    flight_head[0] = next_head;
                    flight_destination[0] = destination_slot_2;
                    flight_store[0] = store_live_2;
                    flight_pad[0] = pad_2;
                }
                ordinal_1 = 2 * pair_2 + 1;
                if (ordinal_1 < count_1) {
                    next_live = (int)(count_1 > ordinal_1 + 2);
                    next_head = fetched_head[1];
                    long long source_slot_3 = 0;
                    long long destination_slot_3 = 0;
                    if (next_live != 0) {
                        source_slot_3 = slots_1[0];
                        destination_slot_3 = slots_1[1];
                    }
                    int load_live_3 = next_live;
                    int store_live_3 = next_live * (1 - disable_state_update);
                    float pad_3 = 0.0f;
                    if (source_slot_3 == pad_slot_id) {
                        load_live_3 = 0;
                        store_live_3 = 0;
                        pad_3 = 1.0f;
                    }
                    if (destination_slot_3 == pad_slot_id) {
                        store_live_3 = 0;
                    }
                    if (elect_sync()) {
                        int _parity_state_updated_4 = (int)((unsigned int)pair_2 % 2);
                        mbarrier_wait(state_updated_addr + 32, _parity_state_updated_4);
                        if (flight_store[1] != 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_4d((&state_tma), 0, 0, flight_head[1], (int)flight_destination[1], s_state_addr + 32768);
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        if (next_live != 0) {
                            if (load_live_3 != 0) {
                                tma_4d_gmem2smem(s_state_addr + 32768, (&state_tma), 0, 0, next_head, (int)source_slot_3, state_full_addr + 32);
                                mbarrier_arrive_expect_tx(state_full_addr + 32, 8192);
                            } else {
                                mbarrier_arrive(state_full_addr + 32);
                            }
                        }
                    }
                    if (count_1 > ordinal_1 + 1) {
                        if (ordinal_1 + 1 >= 2) {
                            int _parity_aux_empty_1 = (int)((unsigned int)pair_2 % 2);
                            mbarrier_wait(aux_empty_addr, _parity_aux_empty_1);
                        }
                        int slot_byte_2 = ((ordinal_1 + 1 & 1) * 520 + 4 * lane_0) * 4;
                        float values_4[4];
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            (&values_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_words_0[_pair]) << 16);
                            (&values_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_words_0[_pair]) & 0xffff0000u);
                        }
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_aux_addr + (unsigned int)slot_byte_2), "f"(values_4[0]), "f"(values_4[1]), "f"(values_4[2]), "f"(values_4[3]));
                        float values_0_2[4];
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            (&values_0_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(z_words_0[_pair]) << 16);
                            (&values_0_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(z_words_0[_pair]) & 0xffff0000u);
                        }
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_aux_addr + (unsigned int)(slot_byte_2 + 512)), "f"(values_0_2[0]), "f"(values_0_2[1]), "f"(values_0_2[2]), "f"(values_0_2[3]));
                        float values_1_2[4];
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            (&values_1_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(b_words_0[_pair]) << 16);
                            (&values_1_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(b_words_0[_pair]) & 0xffff0000u);
                        }
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_aux_addr + (unsigned int)(slot_byte_2 + 1024)), "f"(values_1_2[0]), "f"(values_1_2[1]), "f"(values_1_2[2]), "f"(values_1_2[3]));
                        float values_2_2[4];
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            (&values_2_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(c_words_0[_pair]) << 16);
                            (&values_2_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(c_words_0[_pair]) & 0xffff0000u);
                        }
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_aux_addr + (unsigned int)(slot_byte_2 + 1536)), "f"(values_2_2[0]), "f"(values_2_2[1]), "f"(values_2_2[2]), "f"(values_2_2[3]));
                        if (lane_0 == 0) {
                            float scalar_dt_2 = scalars_0[0] + scalars_0[1];
                            if (dt_softplus != 0) {
                                if (scalar_dt_2 <= 20.0f) {
                                    float _exp_39 = __expf(scalar_dt_2);
                                    float _log2_38;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_38) : "f"(1.0f + _exp_39));
                                    scalar_dt_2 = _log2_38 * 0.6931472f;
                                }
                            }
                            int scalar_base_4 = (ordinal_1 + 1 & 1) * 520 + 512;
                            s_aux[scalar_base_4] = scalar_dt_2;
                            float _exp_40 = expf(scalars_0[3] * scalar_dt_2);
                            s_aux[scalar_base_4 + 1] = _exp_40;
                            s_aux[scalar_base_4 + 2] = scalars_0[2];
                            s_aux[(ordinal_1 + 1 & 1) * 520 + 516] = flight_pad[0];
                        }
                        mbarrier_arrive(aux_full_addr);
                    }
                    if (count_1 > ordinal_1 + 3) {
                        int fetch_unit_4 = bid + (ordinal_1 + 3) * num_bids;
                        fetched_head[0] = fetch_unit_4 - fetch_unit_4 / nheads * nheads;
                        int batch_6 = fetch_unit_4 / nheads;
                        int head_6 = fetch_unit_4 - batch_6 * nheads;
                        int group_4 = head_6 / heads_per_group;
                        long long batch_i64_4 = (long long)batch_6;
                        {
                            asm("ld.global.nc.s64 %0, [%1];" : "=l"(slots_0[0]) : "l"((const void*)(reinterpret_cast<const long long*>(state_batch_indices_addr) + (batch_i64_4))));
                        }
                        {
                            asm("ld.global.nc.s64 %0, [%1];" : "=l"(slots_0[1]) : "l"((const void*)(reinterpret_cast<const long long*>(dst_state_batch_indices_addr) + (batch_i64_4))));
                        }
                        long long quad_i64_4 = (long long)(4 * lane_0);
                        long long x_row_4 = batch_i64_4 * x_batch_stride + (long long)(head_6 * 128) + quad_i64_4;
                        long long b_row_4 = batch_i64_4 * b_batch_stride + (long long)(group_4 * 128) + quad_i64_4;
                        long long c_row_4 = batch_i64_4 * c_batch_stride + (long long)(group_4 * 128) + quad_i64_4;
                        {
                            uint2 _vld_41;
                            asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                                : "=r"(_vld_41.x), "=r"(_vld_41.y) : "l"((const void*)(x + x_row_4)) : "memory");
                            uint2* _vdst_41 = reinterpret_cast<uint2*>(&x_words_0[0]);
                            *_vdst_41 = _vld_41;
                        }
                        if (has_z != 0) {
                            {
                                uint2 _vld_42;
                                asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                                    : "=r"(_vld_42.x), "=r"(_vld_42.y) : "l"((const void*)(z + x_row_4)) : "memory");
                                uint2* _vdst_42 = reinterpret_cast<uint2*>(&z_words_0[0]);
                                *_vdst_42 = _vld_42;
                            }
                        }
                        {
                            uint2 _vld_43;
                            asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                                : "=r"(_vld_43.x), "=r"(_vld_43.y) : "l"((const void*)(B + b_row_4)) : "memory");
                            uint2* _vdst_43 = reinterpret_cast<uint2*>(&b_words_0[0]);
                            *_vdst_43 = _vld_43;
                        }
                        {
                            uint2 _vld_44;
                            asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                                : "=r"(_vld_44.x), "=r"(_vld_44.y) : "l"((const void*)(C + c_row_4)) : "memory");
                            uint2* _vdst_44 = reinterpret_cast<uint2*>(&c_words_0[0]);
                            *_vdst_44 = _vld_44;
                        }
                        if (lane_0 == 0) {
                            long long head_i64_4 = (long long)head_6;
                            long long dt_index_4 = batch_i64_4 * dt_batch_stride + head_i64_4;
                            {
                                uint32_t _scalar_bits_45;
                                asm volatile("ld.global.nc.b32 %0, [%1];"
                                    : "=r"(_scalar_bits_45) : "l"((const void*)(reinterpret_cast<const float*>(dt_addr) + (dt_index_4))) : "memory");
                                scalars_0[0] = __uint_as_float(_scalar_bits_45);
                            }
                            {
                                uint32_t _scalar_bits_46;
                                asm volatile("ld.global.nc.b32 %0, [%1];"
                                    : "=r"(_scalar_bits_46) : "l"((const void*)(reinterpret_cast<const float*>(dt_bias_addr) + (head_i64_4))) : "memory");
                                scalars_0[1] = __uint_as_float(_scalar_bits_46);
                            }
                            {
                                uint32_t _scalar_bits_47;
                                asm volatile("ld.global.nc.b32 %0, [%1];"
                                    : "=r"(_scalar_bits_47) : "l"((const void*)(reinterpret_cast<const float*>(d_addr) + (head_i64_4))) : "memory");
                                scalars_0[2] = __uint_as_float(_scalar_bits_47);
                            }
                            {
                                uint32_t _scalar_bits_48;
                                asm volatile("ld.global.nc.b32 %0, [%1];"
                                    : "=r"(_scalar_bits_48) : "l"((const void*)(reinterpret_cast<const float*>(a_addr) + (head_i64_4))) : "memory");
                                scalars_0[3] = __uint_as_float(_scalar_bits_48);
                            }
                        }
                    }
                    if (elect_sync()) {
                        int _parity_state_updated_5 = (int)((unsigned int)pair_2 % 2);
                        mbarrier_wait(state_updated_addr + 40, _parity_state_updated_5);
                        if (flight_store[1] != 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_4d((&state_tma), 0, 32, flight_head[1], (int)flight_destination[1], s_state_addr + 40960);
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        if (next_live != 0) {
                            if (load_live_3 != 0) {
                                tma_4d_gmem2smem(s_state_addr + 40960, (&state_tma), 0, 32, next_head, (int)source_slot_3, state_full_addr + 40);
                                mbarrier_arrive_expect_tx(state_full_addr + 40, 8192);
                            } else {
                                mbarrier_arrive(state_full_addr + 40);
                            }
                        }
                    }
                    if (elect_sync()) {
                        int _parity_state_updated_6 = (int)((unsigned int)pair_2 % 2);
                        mbarrier_wait(state_updated_addr + 48, _parity_state_updated_6);
                        if (flight_store[1] != 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_4d((&state_tma), 0, 64, flight_head[1], (int)flight_destination[1], s_state_addr + 49152);
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        if (next_live != 0) {
                            if (load_live_3 != 0) {
                                tma_4d_gmem2smem(s_state_addr + 49152, (&state_tma), 0, 64, next_head, (int)source_slot_3, state_full_addr + 48);
                                mbarrier_arrive_expect_tx(state_full_addr + 48, 8192);
                            } else {
                                mbarrier_arrive(state_full_addr + 48);
                            }
                        }
                    }
                    if (elect_sync()) {
                        int _parity_state_updated_7 = (int)((unsigned int)pair_2 % 2);
                        mbarrier_wait(state_updated_addr + 56, _parity_state_updated_7);
                        if (flight_store[1] != 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_4d((&state_tma), 0, 96, flight_head[1], (int)flight_destination[1], s_state_addr + 57344);
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        if (next_live != 0) {
                            if (load_live_3 != 0) {
                                tma_4d_gmem2smem(s_state_addr + 57344, (&state_tma), 0, 96, next_head, (int)source_slot_3, state_full_addr + 56);
                                mbarrier_arrive_expect_tx(state_full_addr + 56, 8192);
                            } else {
                                mbarrier_arrive(state_full_addr + 56);
                            }
                        }
                    }
                    flight_head[1] = next_head;
                    flight_destination[1] = destination_slot_3;
                    flight_store[1] = store_live_3;
                    flight_pad[1] = pad_3;
                }
            }
            if (elect_sync()) {
                if (disable_state_update == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
