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
#define NUM_MAIN_STAGES 1
#define SMEM_WARP_TAB_OFF 0
#define SMEM_WARP_TAB_STAGE_BYTES 128
#define SMEM_WARP_TAB_STRIDE 128
#define SMEM_CTA_TAB_OFF 128
#define SMEM_CTA_TAB_STAGE_BYTES 128
#define SMEM_CTA_TAB_STRIDE 128
#define SMEM_TOTAL 256
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_tp_moe_finalize_rmsnorm_4b8865d07abb02c98d23(__nv_bfloat16* __restrict__ gemm2, int row_stride, int rows, int* __restrict__ idx, unsigned int* __restrict__ expert_weights, int weights_fp32, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ output, int n_tok, int slots, float eps, float gain_center, int mode, int rank, long long* __restrict__ peer_table, int* __restrict__ gen)
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
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* warp_tab = reinterpret_cast<float*>(smem_raw + 0);
    const int warp_tab_addr = smem + 0;
    float* cta_tab = reinterpret_cast<float*>(smem_raw + 128);
    const int cta_tab_addr = smem + 128;

    // === Task calls (declaration order) ===
    int slot = cluster_id;
    int cta_rank_0 = cta_rank;
    if (__syncthreads_and(slot >= n_tok)) return;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    int col = (cta_rank_0 * 128 + tid) * 8;
    long long col64 = (long long)col;
    int g = gen[slot];
    int g_old = ((g == 0) ? 2 : g - 1);
    long long gen_elems = (long long)slots * 65536;
    long long slot_base = (long long)slot * 65536 + col64;
    long long cur = gen_elems * (long long)g + slot_base;
    long long old = gen_elems * (long long)g_old + slot_base;
    long long mine = cur + (long long)rank * 4096;
    long long local_addr = peer_table[rank];
    __nv_bfloat16* local = reinterpret_cast<__nv_bfloat16*>(local_addr);
    long long peer_addr_0 = peer_table[0];
    long long peer_addr_1 = peer_table[1];
    __nv_bfloat16* peers_0 = reinterpret_cast<__nv_bfloat16*>(peer_addr_0);
    __nv_bfloat16* peers_1 = reinterpret_cast<__nv_bfloat16*>(peer_addr_1);
    int idx8[8];
    {
        const int4* _ivptr_0 = reinterpret_cast<const int4*>(idx + (slot * 8) + 0);
        int4 _ivld_0;
        _ivld_0 = *_ivptr_0;
        idx8[0 + 0] = _ivld_0.x;
        idx8[0 + 1] = _ivld_0.y;
        idx8[0 + 2] = _ivld_0.z;
        idx8[0 + 3] = _ivld_0.w;
    }
    {
        const int4* _ivptr_1 = reinterpret_cast<const int4*>(idx + (slot * 8 + 4) + 0);
        int4 _ivld_1;
        _ivld_1 = *_ivptr_1;
        idx8[4 + 0] = _ivld_1.x;
        idx8[4 + 1] = _ivld_1.y;
        idx8[4 + 2] = _ivld_1.z;
        idx8[4 + 3] = _ivld_1.w;
    }
    float w8[8];
    if (weights_fp32 != 0) {
        float _vec_load_0[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(expert_weights + (slot * 8)) + 0);
            _vec_load_0[0 + 0] = _v4.x;
            _vec_load_0[0 + 1] = _v4.y;
            _vec_load_0[0 + 2] = _v4.z;
            _vec_load_0[0 + 3] = _v4.w;
        }
        float _vec_load_1[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(expert_weights + (slot * 8 + 4)) + 0);
            _vec_load_1[0 + 0] = _v4.x;
            _vec_load_1[0 + 1] = _v4.y;
            _vec_load_1[0 + 2] = _v4.z;
            _vec_load_1[0 + 3] = _v4.w;
        }
        w8[0] = _vec_load_0[0];
        w8[4] = _vec_load_1[0];
        w8[1] = _vec_load_0[1];
        w8[5] = _vec_load_1[1];
        w8[2] = _vec_load_0[2];
        w8[6] = _vec_load_1[2];
        w8[3] = _vec_load_0[3];
        w8[7] = _vec_load_1[3];
    } else {
        unsigned int _vec_load_2[4];
        {
            uint4 _uv4_4 = *reinterpret_cast<const uint4*>(expert_weights + (slot * 4) + 0);
            _vec_load_2[0 + 0] = _uv4_4.x;
            _vec_load_2[0 + 1] = _uv4_4.y;
            _vec_load_2[0 + 2] = _uv4_4.z;
            _vec_load_2[0 + 3] = _uv4_4.w;
        }
        float _unpack_bf16_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_0[_pair * 2])[0]), "=f"((&_unpack_bf16_0[_pair * 2])[1])
                : "r"(_vec_load_2[_pair]));
        }
        w8[0] = _unpack_bf16_0[0];
        w8[1] = _unpack_bf16_0[1];
        w8[2] = _unpack_bf16_0[2];
        w8[3] = _unpack_bf16_0[3];
        w8[4] = _unpack_bf16_0[4];
        w8[5] = _unpack_bf16_0[5];
        w8[6] = _unpack_bf16_0[6];
        w8[7] = _unpack_bf16_0[7];
    }
    unsigned int _vec_load_3[4];
    {
        uint4 _uv4_5 = *reinterpret_cast<const uint4*>(norm_weight + col + 0);
        _vec_load_3[0 + 0] = _uv4_5.x;
        _vec_load_3[0 + 1] = _uv4_5.y;
        _vec_load_3[0 + 2] = _uv4_5.z;
        _vec_load_3[0 + 3] = _uv4_5.w;
    }
    float _unpack_bf16_1[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_1[_pair * 2])[0]), "=f"((&_unpack_bf16_1[_pair * 2])[1])
            : "r"(_vec_load_3[_pair]));
    }
    float gw8[8];
    gw8[0] = _unpack_bf16_1[0] + gain_center;
    gw8[1] = _unpack_bf16_1[1] + gain_center;
    gw8[2] = _unpack_bf16_1[2] + gain_center;
    gw8[3] = _unpack_bf16_1[3] + gain_center;
    gw8[4] = _unpack_bf16_1[4] + gain_center;
    gw8[5] = _unpack_bf16_1[5] + gain_center;
    gw8[6] = _unpack_bf16_1[6] + gain_center;
    gw8[7] = _unpack_bf16_1[7] + gain_center;
    unsigned int sent4[4];
    sent4[0] = 2147516416;
    sent4[1] = 2147516416;
    sent4[2] = 2147516416;
    sent4[3] = 2147516416;
    {
        auto _gst_6 = local + old;
        __builtin_assume(__isGlobal(_gst_6));
        reinterpret_cast<int4*>(_gst_6)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_7 = local + (old + 4096);
        __builtin_assume(__isGlobal(_gst_7));
        reinterpret_cast<int4*>(_gst_7)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_8 = local + (old + 8192);
        __builtin_assume(__isGlobal(_gst_8));
        reinterpret_cast<int4*>(_gst_8)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_9 = local + (old + 12288);
        __builtin_assume(__isGlobal(_gst_9));
        reinterpret_cast<int4*>(_gst_9)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_10 = local + (old + 16384);
        __builtin_assume(__isGlobal(_gst_10));
        reinterpret_cast<int4*>(_gst_10)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_11 = local + (old + 20480);
        __builtin_assume(__isGlobal(_gst_11));
        reinterpret_cast<int4*>(_gst_11)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_12 = local + (old + 24576);
        __builtin_assume(__isGlobal(_gst_12));
        reinterpret_cast<int4*>(_gst_12)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_13 = local + (old + 28672);
        __builtin_assume(__isGlobal(_gst_13));
        reinterpret_cast<int4*>(_gst_13)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_14 = local + (old + 32768);
        __builtin_assume(__isGlobal(_gst_14));
        reinterpret_cast<int4*>(_gst_14)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_15 = local + (old + 36864);
        __builtin_assume(__isGlobal(_gst_15));
        reinterpret_cast<int4*>(_gst_15)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_16 = local + (old + 40960);
        __builtin_assume(__isGlobal(_gst_16));
        reinterpret_cast<int4*>(_gst_16)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_17 = local + (old + 45056);
        __builtin_assume(__isGlobal(_gst_17));
        reinterpret_cast<int4*>(_gst_17)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_18 = local + (old + 49152);
        __builtin_assume(__isGlobal(_gst_18));
        reinterpret_cast<int4*>(_gst_18)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_19 = local + (old + 53248);
        __builtin_assume(__isGlobal(_gst_19));
        reinterpret_cast<int4*>(_gst_19)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_20 = local + (old + 57344);
        __builtin_assume(__isGlobal(_gst_20));
        reinterpret_cast<int4*>(_gst_20)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_21 = local + (old + 61440);
        __builtin_assume(__isGlobal(_gst_21));
        reinterpret_cast<int4*>(_gst_21)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    unsigned int packets[32];
    packets[0] = 0;
    packets[1] = 0;
    packets[2] = 0;
    packets[3] = 0;
    packets[4] = 0;
    packets[5] = 0;
    packets[6] = 0;
    packets[7] = 0;
    packets[8] = 0;
    packets[9] = 0;
    packets[10] = 0;
    packets[11] = 0;
    packets[12] = 0;
    packets[13] = 0;
    packets[14] = 0;
    packets[15] = 0;
    packets[16] = 0;
    packets[17] = 0;
    packets[18] = 0;
    packets[19] = 0;
    packets[20] = 0;
    packets[21] = 0;
    packets[22] = 0;
    packets[23] = 0;
    packets[24] = 0;
    packets[25] = 0;
    packets[26] = 0;
    packets[27] = 0;
    packets[28] = 0;
    packets[29] = 0;
    packets[30] = 0;
    packets[31] = 0;
    int ridx0 = idx8[0];
    if (ridx0 >= 0 && ridx0 < rows) {
        {
            uint4 _uv4_22 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx0 * (long long)row_stride + col64) + 0);
            packets[0 + 0] = _uv4_22.x;
            packets[0 + 1] = _uv4_22.y;
            packets[0 + 2] = _uv4_22.z;
            packets[0 + 3] = _uv4_22.w;
        }
    }
    int ridx1 = idx8[1];
    if (ridx1 >= 0 && ridx1 < rows) {
        {
            uint4 _uv4_23 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx1 * (long long)row_stride + col64) + 0);
            packets[4 + 0] = _uv4_23.x;
            packets[4 + 1] = _uv4_23.y;
            packets[4 + 2] = _uv4_23.z;
            packets[4 + 3] = _uv4_23.w;
        }
    }
    int ridx2 = idx8[2];
    if (ridx2 >= 0 && ridx2 < rows) {
        {
            uint4 _uv4_24 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx2 * (long long)row_stride + col64) + 0);
            packets[8 + 0] = _uv4_24.x;
            packets[8 + 1] = _uv4_24.y;
            packets[8 + 2] = _uv4_24.z;
            packets[8 + 3] = _uv4_24.w;
        }
    }
    int ridx3 = idx8[3];
    if (ridx3 >= 0 && ridx3 < rows) {
        {
            uint4 _uv4_25 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx3 * (long long)row_stride + col64) + 0);
            packets[12 + 0] = _uv4_25.x;
            packets[12 + 1] = _uv4_25.y;
            packets[12 + 2] = _uv4_25.z;
            packets[12 + 3] = _uv4_25.w;
        }
    }
    int ridx4 = idx8[4];
    if (ridx4 >= 0 && ridx4 < rows) {
        {
            uint4 _uv4_26 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx4 * (long long)row_stride + col64) + 0);
            packets[16 + 0] = _uv4_26.x;
            packets[16 + 1] = _uv4_26.y;
            packets[16 + 2] = _uv4_26.z;
            packets[16 + 3] = _uv4_26.w;
        }
    }
    int ridx5 = idx8[5];
    if (ridx5 >= 0 && ridx5 < rows) {
        {
            uint4 _uv4_27 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx5 * (long long)row_stride + col64) + 0);
            packets[20 + 0] = _uv4_27.x;
            packets[20 + 1] = _uv4_27.y;
            packets[20 + 2] = _uv4_27.z;
            packets[20 + 3] = _uv4_27.w;
        }
    }
    int ridx6 = idx8[6];
    if (ridx6 >= 0 && ridx6 < rows) {
        {
            uint4 _uv4_28 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx6 * (long long)row_stride + col64) + 0);
            packets[24 + 0] = _uv4_28.x;
            packets[24 + 1] = _uv4_28.y;
            packets[24 + 2] = _uv4_28.z;
            packets[24 + 3] = _uv4_28.w;
        }
    }
    int ridx7 = idx8[7];
    if (ridx7 >= 0 && ridx7 < rows) {
        {
            uint4 _uv4_29 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx7 * (long long)row_stride + col64) + 0);
            packets[28 + 0] = _uv4_29.x;
            packets[28 + 1] = _uv4_29.y;
            packets[28 + 2] = _uv4_29.z;
            packets[28 + 3] = _uv4_29.w;
        }
    }
    unsigned int raw0[4];
    raw0[0] = packets[0];
    raw0[1] = packets[1];
    raw0[2] = packets[2];
    raw0[3] = packets[3];
    uint32_t _packed16_sanitize_0[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_30 = static_cast<uint32_t>(raw0[_p16s_word]);
        uint32_t _p16s_lo_30 = _p16s_carrier_30 & 0x0000ffffu;
        uint32_t _p16s_hi_30 = (_p16s_carrier_30 >> 16) & 0x0000ffffu;
        if (_p16s_lo_30 == 0x00008000u) _p16s_lo_30 = 0u;
        if (_p16s_hi_30 == 0x00008000u) _p16s_hi_30 = 0u;
        _packed16_sanitize_0[_p16s_word] = _p16s_lo_30 | (_p16s_hi_30 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + mine), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + mine), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
    unsigned int raw1[4];
    raw1[0] = packets[4];
    raw1[1] = packets[5];
    raw1[2] = packets[6];
    raw1[3] = packets[7];
    uint32_t _packed16_sanitize_1[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_31 = static_cast<uint32_t>(raw1[_p16s_word]);
        uint32_t _p16s_lo_31 = _p16s_carrier_31 & 0x0000ffffu;
        uint32_t _p16s_hi_31 = (_p16s_carrier_31 >> 16) & 0x0000ffffu;
        if (_p16s_lo_31 == 0x00008000u) _p16s_lo_31 = 0u;
        if (_p16s_hi_31 == 0x00008000u) _p16s_hi_31 = 0u;
        _packed16_sanitize_1[_p16s_word] = _p16s_lo_31 | (_p16s_hi_31 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + (mine + 8192)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + (mine + 8192)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
    unsigned int raw2[4];
    raw2[0] = packets[8];
    raw2[1] = packets[9];
    raw2[2] = packets[10];
    raw2[3] = packets[11];
    uint32_t _packed16_sanitize_2[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_32 = static_cast<uint32_t>(raw2[_p16s_word]);
        uint32_t _p16s_lo_32 = _p16s_carrier_32 & 0x0000ffffu;
        uint32_t _p16s_hi_32 = (_p16s_carrier_32 >> 16) & 0x0000ffffu;
        if (_p16s_lo_32 == 0x00008000u) _p16s_lo_32 = 0u;
        if (_p16s_hi_32 == 0x00008000u) _p16s_hi_32 = 0u;
        _packed16_sanitize_2[_p16s_word] = _p16s_lo_32 | (_p16s_hi_32 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + (mine + 16384)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + (mine + 16384)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
    unsigned int raw3[4];
    raw3[0] = packets[12];
    raw3[1] = packets[13];
    raw3[2] = packets[14];
    raw3[3] = packets[15];
    uint32_t _packed16_sanitize_3[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_33 = static_cast<uint32_t>(raw3[_p16s_word]);
        uint32_t _p16s_lo_33 = _p16s_carrier_33 & 0x0000ffffu;
        uint32_t _p16s_hi_33 = (_p16s_carrier_33 >> 16) & 0x0000ffffu;
        if (_p16s_lo_33 == 0x00008000u) _p16s_lo_33 = 0u;
        if (_p16s_hi_33 == 0x00008000u) _p16s_hi_33 = 0u;
        _packed16_sanitize_3[_p16s_word] = _p16s_lo_33 | (_p16s_hi_33 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + (mine + 24576)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + (mine + 24576)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
    unsigned int raw4[4];
    raw4[0] = packets[16];
    raw4[1] = packets[17];
    raw4[2] = packets[18];
    raw4[3] = packets[19];
    uint32_t _packed16_sanitize_4[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_34 = static_cast<uint32_t>(raw4[_p16s_word]);
        uint32_t _p16s_lo_34 = _p16s_carrier_34 & 0x0000ffffu;
        uint32_t _p16s_hi_34 = (_p16s_carrier_34 >> 16) & 0x0000ffffu;
        if (_p16s_lo_34 == 0x00008000u) _p16s_lo_34 = 0u;
        if (_p16s_hi_34 == 0x00008000u) _p16s_hi_34 = 0u;
        _packed16_sanitize_4[_p16s_word] = _p16s_lo_34 | (_p16s_hi_34 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + (mine + 32768)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + (mine + 32768)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
    unsigned int raw5[4];
    raw5[0] = packets[20];
    raw5[1] = packets[21];
    raw5[2] = packets[22];
    raw5[3] = packets[23];
    uint32_t _packed16_sanitize_5[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_35 = static_cast<uint32_t>(raw5[_p16s_word]);
        uint32_t _p16s_lo_35 = _p16s_carrier_35 & 0x0000ffffu;
        uint32_t _p16s_hi_35 = (_p16s_carrier_35 >> 16) & 0x0000ffffu;
        if (_p16s_lo_35 == 0x00008000u) _p16s_lo_35 = 0u;
        if (_p16s_hi_35 == 0x00008000u) _p16s_hi_35 = 0u;
        _packed16_sanitize_5[_p16s_word] = _p16s_lo_35 | (_p16s_hi_35 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + (mine + 40960)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + (mine + 40960)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
    unsigned int raw6[4];
    raw6[0] = packets[24];
    raw6[1] = packets[25];
    raw6[2] = packets[26];
    raw6[3] = packets[27];
    uint32_t _packed16_sanitize_6[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_36 = static_cast<uint32_t>(raw6[_p16s_word]);
        uint32_t _p16s_lo_36 = _p16s_carrier_36 & 0x0000ffffu;
        uint32_t _p16s_hi_36 = (_p16s_carrier_36 >> 16) & 0x0000ffffu;
        if (_p16s_lo_36 == 0x00008000u) _p16s_lo_36 = 0u;
        if (_p16s_hi_36 == 0x00008000u) _p16s_hi_36 = 0u;
        _packed16_sanitize_6[_p16s_word] = _p16s_lo_36 | (_p16s_hi_36 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + (mine + 49152)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + (mine + 49152)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
    unsigned int raw7[4];
    raw7[0] = packets[28];
    raw7[1] = packets[29];
    raw7[2] = packets[30];
    raw7[3] = packets[31];
    uint32_t _packed16_sanitize_7[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_37 = static_cast<uint32_t>(raw7[_p16s_word]);
        uint32_t _p16s_lo_37 = _p16s_carrier_37 & 0x0000ffffu;
        uint32_t _p16s_hi_37 = (_p16s_carrier_37 >> 16) & 0x0000ffffu;
        if (_p16s_lo_37 == 0x00008000u) _p16s_lo_37 = 0u;
        if (_p16s_hi_37 == 0x00008000u) _p16s_hi_37 = 0u;
        _packed16_sanitize_7[_p16s_word] = _p16s_lo_37 | (_p16s_hi_37 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + (mine + 57344)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + (mine + 57344)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
    float ys[64];
    float ssq8[8];
    uint32_t _sysv_poll_group_0[32];
    do {
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + cur)) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4]), "=r"(_sysv_poll_group_0[5]), "=r"(_sysv_poll_group_0[6]), "=r"(_sysv_poll_group_0[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 4096))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[8]), "=r"(_sysv_poll_group_0[9]), "=r"(_sysv_poll_group_0[10]), "=r"(_sysv_poll_group_0[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 8192))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[12]), "=r"(_sysv_poll_group_0[13]), "=r"(_sysv_poll_group_0[14]), "=r"(_sysv_poll_group_0[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 12288))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[16]), "=r"(_sysv_poll_group_0[17]), "=r"(_sysv_poll_group_0[18]), "=r"(_sysv_poll_group_0[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 16384))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[20]), "=r"(_sysv_poll_group_0[21]), "=r"(_sysv_poll_group_0[22]), "=r"(_sysv_poll_group_0[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 20480))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[24]), "=r"(_sysv_poll_group_0[25]), "=r"(_sysv_poll_group_0[26]), "=r"(_sysv_poll_group_0[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 24576))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[28]), "=r"(_sysv_poll_group_0[29]), "=r"(_sysv_poll_group_0[30]), "=r"(_sysv_poll_group_0[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 28672))) : "memory");
    } while (
        (((_sysv_poll_group_0[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_0[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_0[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_0[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_0[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[9] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_0[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[11] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_0[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[13] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_0[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[15] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_0[16] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[16] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[17] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[17] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_0[18] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[18] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[19] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[19] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_0[20] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[20] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[21] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[21] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_0[22] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[22] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[23] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[23] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_0[24] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[24] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[25] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[25] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_0[26] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[26] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[27] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[27] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_0[28] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[28] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[29] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[29] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_0[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[31] >> 16) & 0xffffu) == 0x8000u)
    );
    float s8_0[8];
    float _unpack_bf16_2[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_2[_pair * 2])[0]), "=f"((&_unpack_bf16_2[_pair * 2])[1])
            : "r"(_sysv_poll_group_0[_pair]));
    }
    s8_0[0] = _unpack_bf16_2[0];
    s8_0[1] = _unpack_bf16_2[1];
    s8_0[2] = _unpack_bf16_2[2];
    s8_0[3] = _unpack_bf16_2[3];
    s8_0[4] = _unpack_bf16_2[4];
    s8_0[5] = _unpack_bf16_2[5];
    s8_0[6] = _unpack_bf16_2[6];
    s8_0[7] = _unpack_bf16_2[7];
    float _unpack_bf16_3[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_3[_pair * 2])[0]), "=f"((&_unpack_bf16_3[_pair * 2])[1])
            : "r"(_sysv_poll_group_0[4 + _pair]));
    }
    s8_0[0] = s8_0[0] + _unpack_bf16_3[0];
    s8_0[1] = s8_0[1] + _unpack_bf16_3[1];
    s8_0[2] = s8_0[2] + _unpack_bf16_3[2];
    s8_0[3] = s8_0[3] + _unpack_bf16_3[3];
    s8_0[4] = s8_0[4] + _unpack_bf16_3[4];
    s8_0[5] = s8_0[5] + _unpack_bf16_3[5];
    s8_0[6] = s8_0[6] + _unpack_bf16_3[6];
    s8_0[7] = s8_0[7] + _unpack_bf16_3[7];
    unsigned int s8_0_bf16[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_0[_lp*2 + 0], s8_0[_lp*2+1 + 0]));
        s8_0_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    float _unpack_bf16_4[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_4[_pair * 2])[0]), "=f"((&_unpack_bf16_4[_pair * 2])[1])
            : "r"(s8_0_bf16[_pair]));
    }
    float ssq0 = 0.0f;
    float y0_0 = ((mode != 0) ? _unpack_bf16_4[0] * gw8[0] : _unpack_bf16_4[0]);
    ys[0] = y0_0;
    ssq0 = __fmaf_rn(y0_0, y0_0, ssq0);
    float y0_1 = ((mode != 0) ? _unpack_bf16_4[1] * gw8[1] : _unpack_bf16_4[1]);
    ys[1] = y0_1;
    ssq0 = __fmaf_rn(y0_1, y0_1, ssq0);
    float y0_2 = ((mode != 0) ? _unpack_bf16_4[2] * gw8[2] : _unpack_bf16_4[2]);
    ys[2] = y0_2;
    ssq0 = __fmaf_rn(y0_2, y0_2, ssq0);
    float y0_3 = ((mode != 0) ? _unpack_bf16_4[3] * gw8[3] : _unpack_bf16_4[3]);
    ys[3] = y0_3;
    ssq0 = __fmaf_rn(y0_3, y0_3, ssq0);
    float y0_4 = ((mode != 0) ? _unpack_bf16_4[4] * gw8[4] : _unpack_bf16_4[4]);
    ys[4] = y0_4;
    ssq0 = __fmaf_rn(y0_4, y0_4, ssq0);
    float y0_5 = ((mode != 0) ? _unpack_bf16_4[5] * gw8[5] : _unpack_bf16_4[5]);
    ys[5] = y0_5;
    ssq0 = __fmaf_rn(y0_5, y0_5, ssq0);
    float y0_6 = ((mode != 0) ? _unpack_bf16_4[6] * gw8[6] : _unpack_bf16_4[6]);
    ys[6] = y0_6;
    ssq0 = __fmaf_rn(y0_6, y0_6, ssq0);
    float y0_7 = ((mode != 0) ? _unpack_bf16_4[7] * gw8[7] : _unpack_bf16_4[7]);
    ys[7] = y0_7;
    ssq0 = __fmaf_rn(y0_7, y0_7, ssq0);
    ssq8[0] = ssq0;
    float s8_1[8];
    float _unpack_bf16_5[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_5[_pair * 2])[0]), "=f"((&_unpack_bf16_5[_pair * 2])[1])
            : "r"(_sysv_poll_group_0[8 + _pair]));
    }
    s8_1[0] = _unpack_bf16_5[0];
    s8_1[1] = _unpack_bf16_5[1];
    s8_1[2] = _unpack_bf16_5[2];
    s8_1[3] = _unpack_bf16_5[3];
    s8_1[4] = _unpack_bf16_5[4];
    s8_1[5] = _unpack_bf16_5[5];
    s8_1[6] = _unpack_bf16_5[6];
    s8_1[7] = _unpack_bf16_5[7];
    float _unpack_bf16_6[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_6[_pair * 2])[0]), "=f"((&_unpack_bf16_6[_pair * 2])[1])
            : "r"(_sysv_poll_group_0[12 + _pair]));
    }
    s8_1[0] = s8_1[0] + _unpack_bf16_6[0];
    s8_1[1] = s8_1[1] + _unpack_bf16_6[1];
    s8_1[2] = s8_1[2] + _unpack_bf16_6[2];
    s8_1[3] = s8_1[3] + _unpack_bf16_6[3];
    s8_1[4] = s8_1[4] + _unpack_bf16_6[4];
    s8_1[5] = s8_1[5] + _unpack_bf16_6[5];
    s8_1[6] = s8_1[6] + _unpack_bf16_6[6];
    s8_1[7] = s8_1[7] + _unpack_bf16_6[7];
    unsigned int s8_1_bf16[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_1[_lp*2 + 0], s8_1[_lp*2+1 + 0]));
        s8_1_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    float _unpack_bf16_7[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_7[_pair * 2])[0]), "=f"((&_unpack_bf16_7[_pair * 2])[1])
            : "r"(s8_1_bf16[_pair]));
    }
    float ssq1 = 0.0f;
    float y1_0 = ((mode != 0) ? _unpack_bf16_7[0] * gw8[0] : _unpack_bf16_7[0]);
    ys[8] = y1_0;
    ssq1 = __fmaf_rn(y1_0, y1_0, ssq1);
    float y1_1 = ((mode != 0) ? _unpack_bf16_7[1] * gw8[1] : _unpack_bf16_7[1]);
    ys[9] = y1_1;
    ssq1 = __fmaf_rn(y1_1, y1_1, ssq1);
    float y1_2 = ((mode != 0) ? _unpack_bf16_7[2] * gw8[2] : _unpack_bf16_7[2]);
    ys[10] = y1_2;
    ssq1 = __fmaf_rn(y1_2, y1_2, ssq1);
    float y1_3 = ((mode != 0) ? _unpack_bf16_7[3] * gw8[3] : _unpack_bf16_7[3]);
    ys[11] = y1_3;
    ssq1 = __fmaf_rn(y1_3, y1_3, ssq1);
    float y1_4 = ((mode != 0) ? _unpack_bf16_7[4] * gw8[4] : _unpack_bf16_7[4]);
    ys[12] = y1_4;
    ssq1 = __fmaf_rn(y1_4, y1_4, ssq1);
    float y1_5 = ((mode != 0) ? _unpack_bf16_7[5] * gw8[5] : _unpack_bf16_7[5]);
    ys[13] = y1_5;
    ssq1 = __fmaf_rn(y1_5, y1_5, ssq1);
    float y1_6 = ((mode != 0) ? _unpack_bf16_7[6] * gw8[6] : _unpack_bf16_7[6]);
    ys[14] = y1_6;
    ssq1 = __fmaf_rn(y1_6, y1_6, ssq1);
    float y1_7 = ((mode != 0) ? _unpack_bf16_7[7] * gw8[7] : _unpack_bf16_7[7]);
    ys[15] = y1_7;
    ssq1 = __fmaf_rn(y1_7, y1_7, ssq1);
    ssq8[1] = ssq1;
    float s8_2[8];
    float _unpack_bf16_8[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_8[_pair * 2])[0]), "=f"((&_unpack_bf16_8[_pair * 2])[1])
            : "r"(_sysv_poll_group_0[16 + _pair]));
    }
    s8_2[0] = _unpack_bf16_8[0];
    s8_2[1] = _unpack_bf16_8[1];
    s8_2[2] = _unpack_bf16_8[2];
    s8_2[3] = _unpack_bf16_8[3];
    s8_2[4] = _unpack_bf16_8[4];
    s8_2[5] = _unpack_bf16_8[5];
    s8_2[6] = _unpack_bf16_8[6];
    s8_2[7] = _unpack_bf16_8[7];
    float _unpack_bf16_9[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_9[_pair * 2])[0]), "=f"((&_unpack_bf16_9[_pair * 2])[1])
            : "r"(_sysv_poll_group_0[20 + _pair]));
    }
    s8_2[0] = s8_2[0] + _unpack_bf16_9[0];
    s8_2[1] = s8_2[1] + _unpack_bf16_9[1];
    s8_2[2] = s8_2[2] + _unpack_bf16_9[2];
    s8_2[3] = s8_2[3] + _unpack_bf16_9[3];
    s8_2[4] = s8_2[4] + _unpack_bf16_9[4];
    s8_2[5] = s8_2[5] + _unpack_bf16_9[5];
    s8_2[6] = s8_2[6] + _unpack_bf16_9[6];
    s8_2[7] = s8_2[7] + _unpack_bf16_9[7];
    unsigned int s8_2_bf16[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_2[_lp*2 + 0], s8_2[_lp*2+1 + 0]));
        s8_2_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    float _unpack_bf16_10[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_10[_pair * 2])[0]), "=f"((&_unpack_bf16_10[_pair * 2])[1])
            : "r"(s8_2_bf16[_pair]));
    }
    float ssq2 = 0.0f;
    float y2_0 = ((mode != 0) ? _unpack_bf16_10[0] * gw8[0] : _unpack_bf16_10[0]);
    ys[16] = y2_0;
    ssq2 = __fmaf_rn(y2_0, y2_0, ssq2);
    float y2_1 = ((mode != 0) ? _unpack_bf16_10[1] * gw8[1] : _unpack_bf16_10[1]);
    ys[17] = y2_1;
    ssq2 = __fmaf_rn(y2_1, y2_1, ssq2);
    float y2_2 = ((mode != 0) ? _unpack_bf16_10[2] * gw8[2] : _unpack_bf16_10[2]);
    ys[18] = y2_2;
    ssq2 = __fmaf_rn(y2_2, y2_2, ssq2);
    float y2_3 = ((mode != 0) ? _unpack_bf16_10[3] * gw8[3] : _unpack_bf16_10[3]);
    ys[19] = y2_3;
    ssq2 = __fmaf_rn(y2_3, y2_3, ssq2);
    float y2_4 = ((mode != 0) ? _unpack_bf16_10[4] * gw8[4] : _unpack_bf16_10[4]);
    ys[20] = y2_4;
    ssq2 = __fmaf_rn(y2_4, y2_4, ssq2);
    float y2_5 = ((mode != 0) ? _unpack_bf16_10[5] * gw8[5] : _unpack_bf16_10[5]);
    ys[21] = y2_5;
    ssq2 = __fmaf_rn(y2_5, y2_5, ssq2);
    float y2_6 = ((mode != 0) ? _unpack_bf16_10[6] * gw8[6] : _unpack_bf16_10[6]);
    ys[22] = y2_6;
    ssq2 = __fmaf_rn(y2_6, y2_6, ssq2);
    float y2_7 = ((mode != 0) ? _unpack_bf16_10[7] * gw8[7] : _unpack_bf16_10[7]);
    ys[23] = y2_7;
    ssq2 = __fmaf_rn(y2_7, y2_7, ssq2);
    ssq8[2] = ssq2;
    float s8_3[8];
    float _unpack_bf16_11[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_11[_pair * 2])[0]), "=f"((&_unpack_bf16_11[_pair * 2])[1])
            : "r"(_sysv_poll_group_0[24 + _pair]));
    }
    s8_3[0] = _unpack_bf16_11[0];
    s8_3[1] = _unpack_bf16_11[1];
    s8_3[2] = _unpack_bf16_11[2];
    s8_3[3] = _unpack_bf16_11[3];
    s8_3[4] = _unpack_bf16_11[4];
    s8_3[5] = _unpack_bf16_11[5];
    s8_3[6] = _unpack_bf16_11[6];
    s8_3[7] = _unpack_bf16_11[7];
    float _unpack_bf16_12[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_12[_pair * 2])[0]), "=f"((&_unpack_bf16_12[_pair * 2])[1])
            : "r"(_sysv_poll_group_0[28 + _pair]));
    }
    s8_3[0] = s8_3[0] + _unpack_bf16_12[0];
    s8_3[1] = s8_3[1] + _unpack_bf16_12[1];
    s8_3[2] = s8_3[2] + _unpack_bf16_12[2];
    s8_3[3] = s8_3[3] + _unpack_bf16_12[3];
    s8_3[4] = s8_3[4] + _unpack_bf16_12[4];
    s8_3[5] = s8_3[5] + _unpack_bf16_12[5];
    s8_3[6] = s8_3[6] + _unpack_bf16_12[6];
    s8_3[7] = s8_3[7] + _unpack_bf16_12[7];
    unsigned int s8_3_bf16[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_3[_lp*2 + 0], s8_3[_lp*2+1 + 0]));
        s8_3_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    float _unpack_bf16_13[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_13[_pair * 2])[0]), "=f"((&_unpack_bf16_13[_pair * 2])[1])
            : "r"(s8_3_bf16[_pair]));
    }
    float ssq3 = 0.0f;
    float y3_0 = ((mode != 0) ? _unpack_bf16_13[0] * gw8[0] : _unpack_bf16_13[0]);
    ys[24] = y3_0;
    ssq3 = __fmaf_rn(y3_0, y3_0, ssq3);
    float y3_1 = ((mode != 0) ? _unpack_bf16_13[1] * gw8[1] : _unpack_bf16_13[1]);
    ys[25] = y3_1;
    ssq3 = __fmaf_rn(y3_1, y3_1, ssq3);
    float y3_2 = ((mode != 0) ? _unpack_bf16_13[2] * gw8[2] : _unpack_bf16_13[2]);
    ys[26] = y3_2;
    ssq3 = __fmaf_rn(y3_2, y3_2, ssq3);
    float y3_3 = ((mode != 0) ? _unpack_bf16_13[3] * gw8[3] : _unpack_bf16_13[3]);
    ys[27] = y3_3;
    ssq3 = __fmaf_rn(y3_3, y3_3, ssq3);
    float y3_4 = ((mode != 0) ? _unpack_bf16_13[4] * gw8[4] : _unpack_bf16_13[4]);
    ys[28] = y3_4;
    ssq3 = __fmaf_rn(y3_4, y3_4, ssq3);
    float y3_5 = ((mode != 0) ? _unpack_bf16_13[5] * gw8[5] : _unpack_bf16_13[5]);
    ys[29] = y3_5;
    ssq3 = __fmaf_rn(y3_5, y3_5, ssq3);
    float y3_6 = ((mode != 0) ? _unpack_bf16_13[6] * gw8[6] : _unpack_bf16_13[6]);
    ys[30] = y3_6;
    ssq3 = __fmaf_rn(y3_6, y3_6, ssq3);
    float y3_7 = ((mode != 0) ? _unpack_bf16_13[7] * gw8[7] : _unpack_bf16_13[7]);
    ys[31] = y3_7;
    ssq3 = __fmaf_rn(y3_7, y3_7, ssq3);
    ssq8[3] = ssq3;
    uint32_t _sysv_poll_group_1[32];
    do {
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[0]), "=r"(_sysv_poll_group_1[1]), "=r"(_sysv_poll_group_1[2]), "=r"(_sysv_poll_group_1[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 32768))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[4]), "=r"(_sysv_poll_group_1[5]), "=r"(_sysv_poll_group_1[6]), "=r"(_sysv_poll_group_1[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 36864))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[8]), "=r"(_sysv_poll_group_1[9]), "=r"(_sysv_poll_group_1[10]), "=r"(_sysv_poll_group_1[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 40960))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[12]), "=r"(_sysv_poll_group_1[13]), "=r"(_sysv_poll_group_1[14]), "=r"(_sysv_poll_group_1[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 45056))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[16]), "=r"(_sysv_poll_group_1[17]), "=r"(_sysv_poll_group_1[18]), "=r"(_sysv_poll_group_1[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 49152))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[20]), "=r"(_sysv_poll_group_1[21]), "=r"(_sysv_poll_group_1[22]), "=r"(_sysv_poll_group_1[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 53248))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[24]), "=r"(_sysv_poll_group_1[25]), "=r"(_sysv_poll_group_1[26]), "=r"(_sysv_poll_group_1[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 57344))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[28]), "=r"(_sysv_poll_group_1[29]), "=r"(_sysv_poll_group_1[30]), "=r"(_sysv_poll_group_1[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 61440))) : "memory");
    } while (
        (((_sysv_poll_group_1[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[1] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_1[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[3] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_1[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[5] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_1[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[7] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_1[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[9] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_1[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[11] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_1[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[13] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_1[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[15] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_1[16] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[16] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[17] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[17] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_1[18] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[18] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[19] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[19] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_1[20] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[20] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[21] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[21] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_1[22] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[22] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[23] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[23] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_1[24] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[24] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[25] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[25] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_1[26] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[26] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[27] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[27] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_1[28] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[28] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[29] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[29] >> 16) & 0xffffu) == 0x8000u) ||
        (((_sysv_poll_group_1[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_1[31] >> 16) & 0xffffu) == 0x8000u)
    );
    float s8_4[8];
    float _unpack_bf16_14[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_14[_pair * 2])[0]), "=f"((&_unpack_bf16_14[_pair * 2])[1])
            : "r"(_sysv_poll_group_1[_pair]));
    }
    s8_4[0] = _unpack_bf16_14[0];
    s8_4[1] = _unpack_bf16_14[1];
    s8_4[2] = _unpack_bf16_14[2];
    s8_4[3] = _unpack_bf16_14[3];
    s8_4[4] = _unpack_bf16_14[4];
    s8_4[5] = _unpack_bf16_14[5];
    s8_4[6] = _unpack_bf16_14[6];
    s8_4[7] = _unpack_bf16_14[7];
    float _unpack_bf16_15[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_15[_pair * 2])[0]), "=f"((&_unpack_bf16_15[_pair * 2])[1])
            : "r"(_sysv_poll_group_1[4 + _pair]));
    }
    s8_4[0] = s8_4[0] + _unpack_bf16_15[0];
    s8_4[1] = s8_4[1] + _unpack_bf16_15[1];
    s8_4[2] = s8_4[2] + _unpack_bf16_15[2];
    s8_4[3] = s8_4[3] + _unpack_bf16_15[3];
    s8_4[4] = s8_4[4] + _unpack_bf16_15[4];
    s8_4[5] = s8_4[5] + _unpack_bf16_15[5];
    s8_4[6] = s8_4[6] + _unpack_bf16_15[6];
    s8_4[7] = s8_4[7] + _unpack_bf16_15[7];
    unsigned int s8_4_bf16[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_4[_lp*2 + 0], s8_4[_lp*2+1 + 0]));
        s8_4_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    float _unpack_bf16_16[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_16[_pair * 2])[0]), "=f"((&_unpack_bf16_16[_pair * 2])[1])
            : "r"(s8_4_bf16[_pair]));
    }
    float ssq4 = 0.0f;
    float y4_0 = ((mode != 0) ? _unpack_bf16_16[0] * gw8[0] : _unpack_bf16_16[0]);
    ys[32] = y4_0;
    ssq4 = __fmaf_rn(y4_0, y4_0, ssq4);
    float y4_1 = ((mode != 0) ? _unpack_bf16_16[1] * gw8[1] : _unpack_bf16_16[1]);
    ys[33] = y4_1;
    ssq4 = __fmaf_rn(y4_1, y4_1, ssq4);
    float y4_2 = ((mode != 0) ? _unpack_bf16_16[2] * gw8[2] : _unpack_bf16_16[2]);
    ys[34] = y4_2;
    ssq4 = __fmaf_rn(y4_2, y4_2, ssq4);
    float y4_3 = ((mode != 0) ? _unpack_bf16_16[3] * gw8[3] : _unpack_bf16_16[3]);
    ys[35] = y4_3;
    ssq4 = __fmaf_rn(y4_3, y4_3, ssq4);
    float y4_4 = ((mode != 0) ? _unpack_bf16_16[4] * gw8[4] : _unpack_bf16_16[4]);
    ys[36] = y4_4;
    ssq4 = __fmaf_rn(y4_4, y4_4, ssq4);
    float y4_5 = ((mode != 0) ? _unpack_bf16_16[5] * gw8[5] : _unpack_bf16_16[5]);
    ys[37] = y4_5;
    ssq4 = __fmaf_rn(y4_5, y4_5, ssq4);
    float y4_6 = ((mode != 0) ? _unpack_bf16_16[6] * gw8[6] : _unpack_bf16_16[6]);
    ys[38] = y4_6;
    ssq4 = __fmaf_rn(y4_6, y4_6, ssq4);
    float y4_7 = ((mode != 0) ? _unpack_bf16_16[7] * gw8[7] : _unpack_bf16_16[7]);
    ys[39] = y4_7;
    ssq4 = __fmaf_rn(y4_7, y4_7, ssq4);
    ssq8[4] = ssq4;
    float s8_5[8];
    float _unpack_bf16_17[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_17[_pair * 2])[0]), "=f"((&_unpack_bf16_17[_pair * 2])[1])
            : "r"(_sysv_poll_group_1[8 + _pair]));
    }
    s8_5[0] = _unpack_bf16_17[0];
    s8_5[1] = _unpack_bf16_17[1];
    s8_5[2] = _unpack_bf16_17[2];
    s8_5[3] = _unpack_bf16_17[3];
    s8_5[4] = _unpack_bf16_17[4];
    s8_5[5] = _unpack_bf16_17[5];
    s8_5[6] = _unpack_bf16_17[6];
    s8_5[7] = _unpack_bf16_17[7];
    float _unpack_bf16_18[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_18[_pair * 2])[0]), "=f"((&_unpack_bf16_18[_pair * 2])[1])
            : "r"(_sysv_poll_group_1[12 + _pair]));
    }
    s8_5[0] = s8_5[0] + _unpack_bf16_18[0];
    s8_5[1] = s8_5[1] + _unpack_bf16_18[1];
    s8_5[2] = s8_5[2] + _unpack_bf16_18[2];
    s8_5[3] = s8_5[3] + _unpack_bf16_18[3];
    s8_5[4] = s8_5[4] + _unpack_bf16_18[4];
    s8_5[5] = s8_5[5] + _unpack_bf16_18[5];
    s8_5[6] = s8_5[6] + _unpack_bf16_18[6];
    s8_5[7] = s8_5[7] + _unpack_bf16_18[7];
    unsigned int s8_5_bf16[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_5[_lp*2 + 0], s8_5[_lp*2+1 + 0]));
        s8_5_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    float _unpack_bf16_19[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_19[_pair * 2])[0]), "=f"((&_unpack_bf16_19[_pair * 2])[1])
            : "r"(s8_5_bf16[_pair]));
    }
    float ssq5 = 0.0f;
    float y5_0 = ((mode != 0) ? _unpack_bf16_19[0] * gw8[0] : _unpack_bf16_19[0]);
    ys[40] = y5_0;
    ssq5 = __fmaf_rn(y5_0, y5_0, ssq5);
    float y5_1 = ((mode != 0) ? _unpack_bf16_19[1] * gw8[1] : _unpack_bf16_19[1]);
    ys[41] = y5_1;
    ssq5 = __fmaf_rn(y5_1, y5_1, ssq5);
    float y5_2 = ((mode != 0) ? _unpack_bf16_19[2] * gw8[2] : _unpack_bf16_19[2]);
    ys[42] = y5_2;
    ssq5 = __fmaf_rn(y5_2, y5_2, ssq5);
    float y5_3 = ((mode != 0) ? _unpack_bf16_19[3] * gw8[3] : _unpack_bf16_19[3]);
    ys[43] = y5_3;
    ssq5 = __fmaf_rn(y5_3, y5_3, ssq5);
    float y5_4 = ((mode != 0) ? _unpack_bf16_19[4] * gw8[4] : _unpack_bf16_19[4]);
    ys[44] = y5_4;
    ssq5 = __fmaf_rn(y5_4, y5_4, ssq5);
    float y5_5 = ((mode != 0) ? _unpack_bf16_19[5] * gw8[5] : _unpack_bf16_19[5]);
    ys[45] = y5_5;
    ssq5 = __fmaf_rn(y5_5, y5_5, ssq5);
    float y5_6 = ((mode != 0) ? _unpack_bf16_19[6] * gw8[6] : _unpack_bf16_19[6]);
    ys[46] = y5_6;
    ssq5 = __fmaf_rn(y5_6, y5_6, ssq5);
    float y5_7 = ((mode != 0) ? _unpack_bf16_19[7] * gw8[7] : _unpack_bf16_19[7]);
    ys[47] = y5_7;
    ssq5 = __fmaf_rn(y5_7, y5_7, ssq5);
    ssq8[5] = ssq5;
    float s8_6[8];
    float _unpack_bf16_20[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_20[_pair * 2])[0]), "=f"((&_unpack_bf16_20[_pair * 2])[1])
            : "r"(_sysv_poll_group_1[16 + _pair]));
    }
    s8_6[0] = _unpack_bf16_20[0];
    s8_6[1] = _unpack_bf16_20[1];
    s8_6[2] = _unpack_bf16_20[2];
    s8_6[3] = _unpack_bf16_20[3];
    s8_6[4] = _unpack_bf16_20[4];
    s8_6[5] = _unpack_bf16_20[5];
    s8_6[6] = _unpack_bf16_20[6];
    s8_6[7] = _unpack_bf16_20[7];
    float _unpack_bf16_21[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_21[_pair * 2])[0]), "=f"((&_unpack_bf16_21[_pair * 2])[1])
            : "r"(_sysv_poll_group_1[20 + _pair]));
    }
    s8_6[0] = s8_6[0] + _unpack_bf16_21[0];
    s8_6[1] = s8_6[1] + _unpack_bf16_21[1];
    s8_6[2] = s8_6[2] + _unpack_bf16_21[2];
    s8_6[3] = s8_6[3] + _unpack_bf16_21[3];
    s8_6[4] = s8_6[4] + _unpack_bf16_21[4];
    s8_6[5] = s8_6[5] + _unpack_bf16_21[5];
    s8_6[6] = s8_6[6] + _unpack_bf16_21[6];
    s8_6[7] = s8_6[7] + _unpack_bf16_21[7];
    unsigned int s8_6_bf16[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_6[_lp*2 + 0], s8_6[_lp*2+1 + 0]));
        s8_6_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    float _unpack_bf16_22[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_22[_pair * 2])[0]), "=f"((&_unpack_bf16_22[_pair * 2])[1])
            : "r"(s8_6_bf16[_pair]));
    }
    float ssq6 = 0.0f;
    float y6_0 = ((mode != 0) ? _unpack_bf16_22[0] * gw8[0] : _unpack_bf16_22[0]);
    ys[48] = y6_0;
    ssq6 = __fmaf_rn(y6_0, y6_0, ssq6);
    float y6_1 = ((mode != 0) ? _unpack_bf16_22[1] * gw8[1] : _unpack_bf16_22[1]);
    ys[49] = y6_1;
    ssq6 = __fmaf_rn(y6_1, y6_1, ssq6);
    float y6_2 = ((mode != 0) ? _unpack_bf16_22[2] * gw8[2] : _unpack_bf16_22[2]);
    ys[50] = y6_2;
    ssq6 = __fmaf_rn(y6_2, y6_2, ssq6);
    float y6_3 = ((mode != 0) ? _unpack_bf16_22[3] * gw8[3] : _unpack_bf16_22[3]);
    ys[51] = y6_3;
    ssq6 = __fmaf_rn(y6_3, y6_3, ssq6);
    float y6_4 = ((mode != 0) ? _unpack_bf16_22[4] * gw8[4] : _unpack_bf16_22[4]);
    ys[52] = y6_4;
    ssq6 = __fmaf_rn(y6_4, y6_4, ssq6);
    float y6_5 = ((mode != 0) ? _unpack_bf16_22[5] * gw8[5] : _unpack_bf16_22[5]);
    ys[53] = y6_5;
    ssq6 = __fmaf_rn(y6_5, y6_5, ssq6);
    float y6_6 = ((mode != 0) ? _unpack_bf16_22[6] * gw8[6] : _unpack_bf16_22[6]);
    ys[54] = y6_6;
    ssq6 = __fmaf_rn(y6_6, y6_6, ssq6);
    float y6_7 = ((mode != 0) ? _unpack_bf16_22[7] * gw8[7] : _unpack_bf16_22[7]);
    ys[55] = y6_7;
    ssq6 = __fmaf_rn(y6_7, y6_7, ssq6);
    ssq8[6] = ssq6;
    float s8_7[8];
    float _unpack_bf16_23[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_23[_pair * 2])[0]), "=f"((&_unpack_bf16_23[_pair * 2])[1])
            : "r"(_sysv_poll_group_1[24 + _pair]));
    }
    s8_7[0] = _unpack_bf16_23[0];
    s8_7[1] = _unpack_bf16_23[1];
    s8_7[2] = _unpack_bf16_23[2];
    s8_7[3] = _unpack_bf16_23[3];
    s8_7[4] = _unpack_bf16_23[4];
    s8_7[5] = _unpack_bf16_23[5];
    s8_7[6] = _unpack_bf16_23[6];
    s8_7[7] = _unpack_bf16_23[7];
    float _unpack_bf16_24[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_24[_pair * 2])[0]), "=f"((&_unpack_bf16_24[_pair * 2])[1])
            : "r"(_sysv_poll_group_1[28 + _pair]));
    }
    s8_7[0] = s8_7[0] + _unpack_bf16_24[0];
    s8_7[1] = s8_7[1] + _unpack_bf16_24[1];
    s8_7[2] = s8_7[2] + _unpack_bf16_24[2];
    s8_7[3] = s8_7[3] + _unpack_bf16_24[3];
    s8_7[4] = s8_7[4] + _unpack_bf16_24[4];
    s8_7[5] = s8_7[5] + _unpack_bf16_24[5];
    s8_7[6] = s8_7[6] + _unpack_bf16_24[6];
    s8_7[7] = s8_7[7] + _unpack_bf16_24[7];
    unsigned int s8_7_bf16[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_7[_lp*2 + 0], s8_7[_lp*2+1 + 0]));
        s8_7_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    float _unpack_bf16_25[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_25[_pair * 2])[0]), "=f"((&_unpack_bf16_25[_pair * 2])[1])
            : "r"(s8_7_bf16[_pair]));
    }
    float ssq7 = 0.0f;
    float y7_0 = ((mode != 0) ? _unpack_bf16_25[0] * gw8[0] : _unpack_bf16_25[0]);
    ys[56] = y7_0;
    ssq7 = __fmaf_rn(y7_0, y7_0, ssq7);
    float y7_1 = ((mode != 0) ? _unpack_bf16_25[1] * gw8[1] : _unpack_bf16_25[1]);
    ys[57] = y7_1;
    ssq7 = __fmaf_rn(y7_1, y7_1, ssq7);
    float y7_2 = ((mode != 0) ? _unpack_bf16_25[2] * gw8[2] : _unpack_bf16_25[2]);
    ys[58] = y7_2;
    ssq7 = __fmaf_rn(y7_2, y7_2, ssq7);
    float y7_3 = ((mode != 0) ? _unpack_bf16_25[3] * gw8[3] : _unpack_bf16_25[3]);
    ys[59] = y7_3;
    ssq7 = __fmaf_rn(y7_3, y7_3, ssq7);
    float y7_4 = ((mode != 0) ? _unpack_bf16_25[4] * gw8[4] : _unpack_bf16_25[4]);
    ys[60] = y7_4;
    ssq7 = __fmaf_rn(y7_4, y7_4, ssq7);
    float y7_5 = ((mode != 0) ? _unpack_bf16_25[5] * gw8[5] : _unpack_bf16_25[5]);
    ys[61] = y7_5;
    ssq7 = __fmaf_rn(y7_5, y7_5, ssq7);
    float y7_6 = ((mode != 0) ? _unpack_bf16_25[6] * gw8[6] : _unpack_bf16_25[6]);
    ys[62] = y7_6;
    ssq7 = __fmaf_rn(y7_6, y7_6, ssq7);
    float y7_7 = ((mode != 0) ? _unpack_bf16_25[7] * gw8[7] : _unpack_bf16_25[7]);
    ys[63] = y7_7;
    ssq7 = __fmaf_rn(y7_7, y7_7, ssq7);
    ssq8[7] = ssq7;
    float _warp_reduce_0 = ssq8[0];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    if (lane == 0) {
        warp_tab[warp * 8] = _warp_reduce_0;
    }
    float _warp_reduce_1 = ssq8[1];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    if (lane == 0) {
        warp_tab[warp * 8 + 1] = _warp_reduce_1;
    }
    float _warp_reduce_2 = ssq8[2];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
    if (lane == 0) {
        warp_tab[warp * 8 + 2] = _warp_reduce_2;
    }
    float _warp_reduce_3 = ssq8[3];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
    if (lane == 0) {
        warp_tab[warp * 8 + 3] = _warp_reduce_3;
    }
    float _warp_reduce_4 = ssq8[4];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_4 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_4, offset);
    if (lane == 0) {
        warp_tab[warp * 8 + 4] = _warp_reduce_4;
    }
    float _warp_reduce_5 = ssq8[5];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_5 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_5, offset);
    if (lane == 0) {
        warp_tab[warp * 8 + 5] = _warp_reduce_5;
    }
    float _warp_reduce_6 = ssq8[6];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_6 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_6, offset);
    if (lane == 0) {
        warp_tab[warp * 8 + 6] = _warp_reduce_6;
    }
    float _warp_reduce_7 = ssq8[7];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_7 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_7, offset);
    if (lane == 0) {
        warp_tab[warp * 8 + 7] = _warp_reduce_7;
    }
    __syncthreads();
    float tot8[8];
    int kk = tid & 7;
    float part = warp_tab[kk];
    part = part + warp_tab[8 + kk];
    part = part + warp_tab[16 + kk];
    part = part + warp_tab[24 + kk];
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (tid < 8) {
        unsigned int slot_addr = cta_tab_addr + (unsigned int)((cta_rank_0 * 8 + tid) * 4);
        uint32_t _cluster_mapa_38;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_38) : "r"(slot_addr), "r"(0));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_38), "f"(part) : "memory");
        uint32_t _cluster_mapa_39;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_39) : "r"(slot_addr), "r"(1));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_39), "f"(part) : "memory");
        uint32_t _cluster_mapa_40;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_40) : "r"(slot_addr), "r"(2));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_40), "f"(part) : "memory");
        uint32_t _cluster_mapa_41;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_41) : "r"(slot_addr), "r"(3));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_41), "f"(part) : "memory");
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    tot8[0] = cta_tab[0];
    tot8[0] = tot8[0] + cta_tab[8];
    tot8[0] = tot8[0] + cta_tab[16];
    tot8[0] = tot8[0] + cta_tab[24];
    tot8[1] = cta_tab[1];
    tot8[1] = tot8[1] + cta_tab[9];
    tot8[1] = tot8[1] + cta_tab[17];
    tot8[1] = tot8[1] + cta_tab[25];
    tot8[2] = cta_tab[2];
    tot8[2] = tot8[2] + cta_tab[10];
    tot8[2] = tot8[2] + cta_tab[18];
    tot8[2] = tot8[2] + cta_tab[26];
    tot8[3] = cta_tab[3];
    tot8[3] = tot8[3] + cta_tab[11];
    tot8[3] = tot8[3] + cta_tab[19];
    tot8[3] = tot8[3] + cta_tab[27];
    tot8[4] = cta_tab[4];
    tot8[4] = tot8[4] + cta_tab[12];
    tot8[4] = tot8[4] + cta_tab[20];
    tot8[4] = tot8[4] + cta_tab[28];
    tot8[5] = cta_tab[5];
    tot8[5] = tot8[5] + cta_tab[13];
    tot8[5] = tot8[5] + cta_tab[21];
    tot8[5] = tot8[5] + cta_tab[29];
    tot8[6] = cta_tab[6];
    tot8[6] = tot8[6] + cta_tab[14];
    tot8[6] = tot8[6] + cta_tab[22];
    tot8[6] = tot8[6] + cta_tab[30];
    tot8[7] = cta_tab[7];
    tot8[7] = tot8[7] + cta_tab[15];
    tot8[7] = tot8[7] + cta_tab[23];
    tot8[7] = tot8[7] + cta_tab[31];
    float acc8[8];
    acc8[0] = 0.0f;
    acc8[1] = 0.0f;
    acc8[2] = 0.0f;
    acc8[3] = 0.0f;
    acc8[4] = 0.0f;
    acc8[5] = 0.0f;
    acc8[6] = 0.0f;
    acc8[7] = 0.0f;
    float _sqrt_42;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_42) : "f"(__fmaf_rn(tot8[0], 0.00024414062f, eps)));
    float rs = __frcp_rn(_sqrt_42);
    float o0_0 = ys[0] * rs;
    float o2_0_0 = ((mode != 0) ? o0_0 : o0_0 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_0_0, w8[0], acc8[0]);
    float o0_1 = ys[1] * rs;
    float o2_0_1 = ((mode != 0) ? o0_1 : o0_1 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_0_1, w8[0], acc8[1]);
    float o0_2 = ys[2] * rs;
    float o2_0_2 = ((mode != 0) ? o0_2 : o0_2 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_0_2, w8[0], acc8[2]);
    float o0_3 = ys[3] * rs;
    float o2_0_3 = ((mode != 0) ? o0_3 : o0_3 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_0_3, w8[0], acc8[3]);
    float o0_4 = ys[4] * rs;
    float o2_0_4 = ((mode != 0) ? o0_4 : o0_4 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_0_4, w8[0], acc8[4]);
    float o0_5 = ys[5] * rs;
    float o2_0_5 = ((mode != 0) ? o0_5 : o0_5 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_0_5, w8[0], acc8[5]);
    float o0_6 = ys[6] * rs;
    float o2_0_6 = ((mode != 0) ? o0_6 : o0_6 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_0_6, w8[0], acc8[6]);
    float o0_7 = ys[7] * rs;
    float o2_0_7 = ((mode != 0) ? o0_7 : o0_7 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_0_7, w8[0], acc8[7]);
    float _sqrt_43;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_43) : "f"(__fmaf_rn(tot8[1], 0.00024414062f, eps)));
    float rs_1 = __frcp_rn(_sqrt_43);
    float o1_0 = ys[8] * rs_1;
    float o2_1_0 = ((mode != 0) ? o1_0 : o1_0 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_1_0, w8[1], acc8[0]);
    float o1_1 = ys[9] * rs_1;
    float o2_1_1 = ((mode != 0) ? o1_1 : o1_1 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_1_1, w8[1], acc8[1]);
    float o1_2 = ys[10] * rs_1;
    float o2_1_2 = ((mode != 0) ? o1_2 : o1_2 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_1_2, w8[1], acc8[2]);
    float o1_3 = ys[11] * rs_1;
    float o2_1_3 = ((mode != 0) ? o1_3 : o1_3 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_1_3, w8[1], acc8[3]);
    float o1_4 = ys[12] * rs_1;
    float o2_1_4 = ((mode != 0) ? o1_4 : o1_4 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_1_4, w8[1], acc8[4]);
    float o1_5 = ys[13] * rs_1;
    float o2_1_5 = ((mode != 0) ? o1_5 : o1_5 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_1_5, w8[1], acc8[5]);
    float o1_6 = ys[14] * rs_1;
    float o2_1_6 = ((mode != 0) ? o1_6 : o1_6 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_1_6, w8[1], acc8[6]);
    float o1_7 = ys[15] * rs_1;
    float o2_1_7 = ((mode != 0) ? o1_7 : o1_7 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_1_7, w8[1], acc8[7]);
    float _sqrt_44;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_44) : "f"(__fmaf_rn(tot8[2], 0.00024414062f, eps)));
    float rs_2 = __frcp_rn(_sqrt_44);
    float o2_0 = ys[16] * rs_2;
    float o2_2_0 = ((mode != 0) ? o2_0 : o2_0 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_2_0, w8[2], acc8[0]);
    float o2_1 = ys[17] * rs_2;
    float o2_2_1 = ((mode != 0) ? o2_1 : o2_1 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_2_1, w8[2], acc8[1]);
    float o2_2 = ys[18] * rs_2;
    float o2_2_2 = ((mode != 0) ? o2_2 : o2_2 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_2_2, w8[2], acc8[2]);
    float o2_3 = ys[19] * rs_2;
    float o2_2_3 = ((mode != 0) ? o2_3 : o2_3 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_2_3, w8[2], acc8[3]);
    float o2_4 = ys[20] * rs_2;
    float o2_2_4 = ((mode != 0) ? o2_4 : o2_4 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_2_4, w8[2], acc8[4]);
    float o2_5 = ys[21] * rs_2;
    float o2_2_5 = ((mode != 0) ? o2_5 : o2_5 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_2_5, w8[2], acc8[5]);
    float o2_6 = ys[22] * rs_2;
    float o2_2_6 = ((mode != 0) ? o2_6 : o2_6 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_2_6, w8[2], acc8[6]);
    float o2_7 = ys[23] * rs_2;
    float o2_2_7 = ((mode != 0) ? o2_7 : o2_7 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_2_7, w8[2], acc8[7]);
    float _sqrt_45;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_45) : "f"(__fmaf_rn(tot8[3], 0.00024414062f, eps)));
    float rs_3 = __frcp_rn(_sqrt_45);
    float o3_0 = ys[24] * rs_3;
    float o2_3_0 = ((mode != 0) ? o3_0 : o3_0 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_3_0, w8[3], acc8[0]);
    float o3_1 = ys[25] * rs_3;
    float o2_3_1 = ((mode != 0) ? o3_1 : o3_1 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_3_1, w8[3], acc8[1]);
    float o3_2 = ys[26] * rs_3;
    float o2_3_2 = ((mode != 0) ? o3_2 : o3_2 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_3_2, w8[3], acc8[2]);
    float o3_3 = ys[27] * rs_3;
    float o2_3_3 = ((mode != 0) ? o3_3 : o3_3 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_3_3, w8[3], acc8[3]);
    float o3_4 = ys[28] * rs_3;
    float o2_3_4 = ((mode != 0) ? o3_4 : o3_4 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_3_4, w8[3], acc8[4]);
    float o3_5 = ys[29] * rs_3;
    float o2_3_5 = ((mode != 0) ? o3_5 : o3_5 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_3_5, w8[3], acc8[5]);
    float o3_6 = ys[30] * rs_3;
    float o2_3_6 = ((mode != 0) ? o3_6 : o3_6 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_3_6, w8[3], acc8[6]);
    float o3_7 = ys[31] * rs_3;
    float o2_3_7 = ((mode != 0) ? o3_7 : o3_7 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_3_7, w8[3], acc8[7]);
    float _sqrt_46;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_46) : "f"(__fmaf_rn(tot8[4], 0.00024414062f, eps)));
    float rs_4 = __frcp_rn(_sqrt_46);
    float o4_0 = ys[32] * rs_4;
    float o2_4_0 = ((mode != 0) ? o4_0 : o4_0 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_4_0, w8[4], acc8[0]);
    float o4_1 = ys[33] * rs_4;
    float o2_4_1 = ((mode != 0) ? o4_1 : o4_1 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_4_1, w8[4], acc8[1]);
    float o4_2 = ys[34] * rs_4;
    float o2_4_2 = ((mode != 0) ? o4_2 : o4_2 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_4_2, w8[4], acc8[2]);
    float o4_3 = ys[35] * rs_4;
    float o2_4_3 = ((mode != 0) ? o4_3 : o4_3 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_4_3, w8[4], acc8[3]);
    float o4_4 = ys[36] * rs_4;
    float o2_4_4 = ((mode != 0) ? o4_4 : o4_4 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_4_4, w8[4], acc8[4]);
    float o4_5 = ys[37] * rs_4;
    float o2_4_5 = ((mode != 0) ? o4_5 : o4_5 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_4_5, w8[4], acc8[5]);
    float o4_6 = ys[38] * rs_4;
    float o2_4_6 = ((mode != 0) ? o4_6 : o4_6 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_4_6, w8[4], acc8[6]);
    float o4_7 = ys[39] * rs_4;
    float o2_4_7 = ((mode != 0) ? o4_7 : o4_7 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_4_7, w8[4], acc8[7]);
    float _sqrt_47;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_47) : "f"(__fmaf_rn(tot8[5], 0.00024414062f, eps)));
    float rs_5 = __frcp_rn(_sqrt_47);
    float o5_0 = ys[40] * rs_5;
    float o2_5_0 = ((mode != 0) ? o5_0 : o5_0 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_5_0, w8[5], acc8[0]);
    float o5_1 = ys[41] * rs_5;
    float o2_5_1 = ((mode != 0) ? o5_1 : o5_1 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_5_1, w8[5], acc8[1]);
    float o5_2 = ys[42] * rs_5;
    float o2_5_2 = ((mode != 0) ? o5_2 : o5_2 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_5_2, w8[5], acc8[2]);
    float o5_3 = ys[43] * rs_5;
    float o2_5_3 = ((mode != 0) ? o5_3 : o5_3 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_5_3, w8[5], acc8[3]);
    float o5_4 = ys[44] * rs_5;
    float o2_5_4 = ((mode != 0) ? o5_4 : o5_4 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_5_4, w8[5], acc8[4]);
    float o5_5 = ys[45] * rs_5;
    float o2_5_5 = ((mode != 0) ? o5_5 : o5_5 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_5_5, w8[5], acc8[5]);
    float o5_6 = ys[46] * rs_5;
    float o2_5_6 = ((mode != 0) ? o5_6 : o5_6 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_5_6, w8[5], acc8[6]);
    float o5_7 = ys[47] * rs_5;
    float o2_5_7 = ((mode != 0) ? o5_7 : o5_7 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_5_7, w8[5], acc8[7]);
    float _sqrt_48;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_48) : "f"(__fmaf_rn(tot8[6], 0.00024414062f, eps)));
    float rs_6 = __frcp_rn(_sqrt_48);
    float o6_0 = ys[48] * rs_6;
    float o2_6_0 = ((mode != 0) ? o6_0 : o6_0 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_6_0, w8[6], acc8[0]);
    float o6_1 = ys[49] * rs_6;
    float o2_6_1 = ((mode != 0) ? o6_1 : o6_1 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_6_1, w8[6], acc8[1]);
    float o6_2 = ys[50] * rs_6;
    float o2_6_2 = ((mode != 0) ? o6_2 : o6_2 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_6_2, w8[6], acc8[2]);
    float o6_3 = ys[51] * rs_6;
    float o2_6_3 = ((mode != 0) ? o6_3 : o6_3 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_6_3, w8[6], acc8[3]);
    float o6_4 = ys[52] * rs_6;
    float o2_6_4 = ((mode != 0) ? o6_4 : o6_4 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_6_4, w8[6], acc8[4]);
    float o6_5 = ys[53] * rs_6;
    float o2_6_5 = ((mode != 0) ? o6_5 : o6_5 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_6_5, w8[6], acc8[5]);
    float o6_6 = ys[54] * rs_6;
    float o2_6_6 = ((mode != 0) ? o6_6 : o6_6 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_6_6, w8[6], acc8[6]);
    float o6_7 = ys[55] * rs_6;
    float o2_6_7 = ((mode != 0) ? o6_7 : o6_7 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_6_7, w8[6], acc8[7]);
    float _sqrt_49;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_49) : "f"(__fmaf_rn(tot8[7], 0.00024414062f, eps)));
    float rs_7 = __frcp_rn(_sqrt_49);
    float o7_0 = ys[56] * rs_7;
    float o2_7_0 = ((mode != 0) ? o7_0 : o7_0 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_7_0, w8[7], acc8[0]);
    float o7_1 = ys[57] * rs_7;
    float o2_7_1 = ((mode != 0) ? o7_1 : o7_1 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_7_1, w8[7], acc8[1]);
    float o7_2 = ys[58] * rs_7;
    float o2_7_2 = ((mode != 0) ? o7_2 : o7_2 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_7_2, w8[7], acc8[2]);
    float o7_3 = ys[59] * rs_7;
    float o2_7_3 = ((mode != 0) ? o7_3 : o7_3 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_7_3, w8[7], acc8[3]);
    float o7_4 = ys[60] * rs_7;
    float o2_7_4 = ((mode != 0) ? o7_4 : o7_4 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_7_4, w8[7], acc8[4]);
    float o7_5 = ys[61] * rs_7;
    float o2_7_5 = ((mode != 0) ? o7_5 : o7_5 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_7_5, w8[7], acc8[5]);
    float o7_6 = ys[62] * rs_7;
    float o2_7_6 = ((mode != 0) ? o7_6 : o7_6 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_7_6, w8[7], acc8[6]);
    float o7_7 = ys[63] * rs_7;
    float o2_7_7 = ((mode != 0) ? o7_7 : o7_7 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_7_7, w8[7], acc8[7]);
    unsigned int _packed_0[4];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc8[_lp*2 + 0], acc8[_lp*2+1 + 0]));
        _packed_0[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc8[_lp*2 + 2], acc8[_lp*2+1 + 2]));
        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc8[_lp*2 + 4], acc8[_lp*2+1 + 4]));
        _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc8[_lp*2 + 6], acc8[_lp*2+1 + 6]));
        _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(output + ((long long)slot * 4096 + col64))[0] = reinterpret_cast<int4*>(_packed_0)[0];
    int g_next = ((g == 2) ? 0 : g + 1);
    if (tid == 0 && cta_rank_0 == 0) {
        *(reinterpret_cast<int*>(gen + slot) + (0)) = g_next;
    }
}

} // extern "C"
