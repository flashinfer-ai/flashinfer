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
kernel_cake_tp_moe_finalize_rmsnorm_0357e2ae406a374a7aba(__nv_bfloat16* __restrict__ gemm2, int row_stride, int rows, int* __restrict__ idx, unsigned int* __restrict__ expert_weights, int weights_fp32, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ output, int n_tok, int slots, float eps, float gain_center, int mode, int rank, long long* __restrict__ peer_table, int* __restrict__ gen)
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
    long long gen_elems = (long long)slots * 131072;
    long long slot_base = (long long)slot * 131072 + col64;
    long long cur = gen_elems * (long long)g + slot_base;
    long long old = gen_elems * (long long)g_old + slot_base;
    long long mine = cur + (long long)rank * 4096;
    long long local_addr = peer_table[rank];
    __nv_bfloat16* local = reinterpret_cast<__nv_bfloat16*>(local_addr);
    long long peer_addr_0 = peer_table[0];
    long long peer_addr_1 = peer_table[1];
    long long peer_addr_2 = peer_table[2];
    long long peer_addr_3 = peer_table[3];
    __nv_bfloat16* peers_0 = reinterpret_cast<__nv_bfloat16*>(peer_addr_0);
    __nv_bfloat16* peers_1 = reinterpret_cast<__nv_bfloat16*>(peer_addr_1);
    __nv_bfloat16* peers_2 = reinterpret_cast<__nv_bfloat16*>(peer_addr_2);
    __nv_bfloat16* peers_3 = reinterpret_cast<__nv_bfloat16*>(peer_addr_3);
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
    {
        auto _gst_22 = local + (old + 65536);
        __builtin_assume(__isGlobal(_gst_22));
        reinterpret_cast<int4*>(_gst_22)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_23 = local + (old + 69632);
        __builtin_assume(__isGlobal(_gst_23));
        reinterpret_cast<int4*>(_gst_23)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_24 = local + (old + 73728);
        __builtin_assume(__isGlobal(_gst_24));
        reinterpret_cast<int4*>(_gst_24)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_25 = local + (old + 77824);
        __builtin_assume(__isGlobal(_gst_25));
        reinterpret_cast<int4*>(_gst_25)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_26 = local + (old + 81920);
        __builtin_assume(__isGlobal(_gst_26));
        reinterpret_cast<int4*>(_gst_26)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_27 = local + (old + 86016);
        __builtin_assume(__isGlobal(_gst_27));
        reinterpret_cast<int4*>(_gst_27)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_28 = local + (old + 90112);
        __builtin_assume(__isGlobal(_gst_28));
        reinterpret_cast<int4*>(_gst_28)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_29 = local + (old + 94208);
        __builtin_assume(__isGlobal(_gst_29));
        reinterpret_cast<int4*>(_gst_29)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_30 = local + (old + 98304);
        __builtin_assume(__isGlobal(_gst_30));
        reinterpret_cast<int4*>(_gst_30)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_31 = local + (old + 102400);
        __builtin_assume(__isGlobal(_gst_31));
        reinterpret_cast<int4*>(_gst_31)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_32 = local + (old + 106496);
        __builtin_assume(__isGlobal(_gst_32));
        reinterpret_cast<int4*>(_gst_32)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_33 = local + (old + 110592);
        __builtin_assume(__isGlobal(_gst_33));
        reinterpret_cast<int4*>(_gst_33)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_34 = local + (old + 114688);
        __builtin_assume(__isGlobal(_gst_34));
        reinterpret_cast<int4*>(_gst_34)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_35 = local + (old + 118784);
        __builtin_assume(__isGlobal(_gst_35));
        reinterpret_cast<int4*>(_gst_35)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_36 = local + (old + 122880);
        __builtin_assume(__isGlobal(_gst_36));
        reinterpret_cast<int4*>(_gst_36)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_37 = local + (old + 126976);
        __builtin_assume(__isGlobal(_gst_37));
        reinterpret_cast<int4*>(_gst_37)[0] = reinterpret_cast<int4*>(sent4)[0];
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
    int ridx_0 = idx8[0];
    if (ridx_0 >= 0 && ridx_0 < rows) {
        {
            uint4 _uv4_38 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_0 * (long long)row_stride + col64) + 0);
            packets[0 + 0] = _uv4_38.x;
            packets[0 + 1] = _uv4_38.y;
            packets[0 + 2] = _uv4_38.z;
            packets[0 + 3] = _uv4_38.w;
        }
    }
    int ridx_1 = idx8[1];
    if (ridx_1 >= 0 && ridx_1 < rows) {
        {
            uint4 _uv4_39 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_1 * (long long)row_stride + col64) + 0);
            packets[4 + 0] = _uv4_39.x;
            packets[4 + 1] = _uv4_39.y;
            packets[4 + 2] = _uv4_39.z;
            packets[4 + 3] = _uv4_39.w;
        }
    }
    int ridx_2 = idx8[2];
    if (ridx_2 >= 0 && ridx_2 < rows) {
        {
            uint4 _uv4_40 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_2 * (long long)row_stride + col64) + 0);
            packets[8 + 0] = _uv4_40.x;
            packets[8 + 1] = _uv4_40.y;
            packets[8 + 2] = _uv4_40.z;
            packets[8 + 3] = _uv4_40.w;
        }
    }
    int ridx_3 = idx8[3];
    if (ridx_3 >= 0 && ridx_3 < rows) {
        {
            uint4 _uv4_41 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_3 * (long long)row_stride + col64) + 0);
            packets[12 + 0] = _uv4_41.x;
            packets[12 + 1] = _uv4_41.y;
            packets[12 + 2] = _uv4_41.z;
            packets[12 + 3] = _uv4_41.w;
        }
    }
    int ridx_4 = idx8[4];
    if (ridx_4 >= 0 && ridx_4 < rows) {
        {
            uint4 _uv4_42 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_4 * (long long)row_stride + col64) + 0);
            packets[16 + 0] = _uv4_42.x;
            packets[16 + 1] = _uv4_42.y;
            packets[16 + 2] = _uv4_42.z;
            packets[16 + 3] = _uv4_42.w;
        }
    }
    int ridx_5 = idx8[5];
    if (ridx_5 >= 0 && ridx_5 < rows) {
        {
            uint4 _uv4_43 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_5 * (long long)row_stride + col64) + 0);
            packets[20 + 0] = _uv4_43.x;
            packets[20 + 1] = _uv4_43.y;
            packets[20 + 2] = _uv4_43.z;
            packets[20 + 3] = _uv4_43.w;
        }
    }
    int ridx_6 = idx8[6];
    if (ridx_6 >= 0 && ridx_6 < rows) {
        {
            uint4 _uv4_44 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_6 * (long long)row_stride + col64) + 0);
            packets[24 + 0] = _uv4_44.x;
            packets[24 + 1] = _uv4_44.y;
            packets[24 + 2] = _uv4_44.z;
            packets[24 + 3] = _uv4_44.w;
        }
    }
    int ridx_7 = idx8[7];
    if (ridx_7 >= 0 && ridx_7 < rows) {
        {
            uint4 _uv4_45 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx_7 * (long long)row_stride + col64) + 0);
            packets[28 + 0] = _uv4_45.x;
            packets[28 + 1] = _uv4_45.y;
            packets[28 + 2] = _uv4_45.z;
            packets[28 + 3] = _uv4_45.w;
        }
    }
    unsigned int raw0_1_0[4];
    raw0_1_0[0] = packets[0];
    raw0_1_0[1] = packets[1];
    raw0_1_0[2] = packets[2];
    raw0_1_0[3] = packets[3];
    uint32_t _packed16_sanitize_0[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_46 = static_cast<uint32_t>(raw0_1_0[_p16s_word]);
        uint32_t _p16s_lo_46 = _p16s_carrier_46 & 0x0000ffffu;
        uint32_t _p16s_hi_46 = (_p16s_carrier_46 >> 16) & 0x0000ffffu;
        if (_p16s_lo_46 == 0x00008000u) _p16s_lo_46 = 0u;
        if (_p16s_hi_46 == 0x00008000u) _p16s_hi_46 = 0u;
        _packed16_sanitize_0[_p16s_word] = _p16s_lo_46 | (_p16s_hi_46 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + mine), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + mine), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_2 + mine), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_3 + mine), "r"((_packed16_sanitize_0)[0]), "r"((_packed16_sanitize_0)[1]), "r"((_packed16_sanitize_0)[2]), "r"((_packed16_sanitize_0)[3]) : "memory");
    unsigned int raw0_1_1[4];
    raw0_1_1[0] = packets[4];
    raw0_1_1[1] = packets[5];
    raw0_1_1[2] = packets[6];
    raw0_1_1[3] = packets[7];
    uint32_t _packed16_sanitize_1[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_47 = static_cast<uint32_t>(raw0_1_1[_p16s_word]);
        uint32_t _p16s_lo_47 = _p16s_carrier_47 & 0x0000ffffu;
        uint32_t _p16s_hi_47 = (_p16s_carrier_47 >> 16) & 0x0000ffffu;
        if (_p16s_lo_47 == 0x00008000u) _p16s_lo_47 = 0u;
        if (_p16s_hi_47 == 0x00008000u) _p16s_hi_47 = 0u;
        _packed16_sanitize_1[_p16s_word] = _p16s_lo_47 | (_p16s_hi_47 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + (mine + 16384)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + (mine + 16384)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_2 + (mine + 16384)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_3 + (mine + 16384)), "r"((_packed16_sanitize_1)[0]), "r"((_packed16_sanitize_1)[1]), "r"((_packed16_sanitize_1)[2]), "r"((_packed16_sanitize_1)[3]) : "memory");
    unsigned int raw0_1_2[4];
    raw0_1_2[0] = packets[8];
    raw0_1_2[1] = packets[9];
    raw0_1_2[2] = packets[10];
    raw0_1_2[3] = packets[11];
    uint32_t _packed16_sanitize_2[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_48 = static_cast<uint32_t>(raw0_1_2[_p16s_word]);
        uint32_t _p16s_lo_48 = _p16s_carrier_48 & 0x0000ffffu;
        uint32_t _p16s_hi_48 = (_p16s_carrier_48 >> 16) & 0x0000ffffu;
        if (_p16s_lo_48 == 0x00008000u) _p16s_lo_48 = 0u;
        if (_p16s_hi_48 == 0x00008000u) _p16s_hi_48 = 0u;
        _packed16_sanitize_2[_p16s_word] = _p16s_lo_48 | (_p16s_hi_48 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + (mine + 32768)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + (mine + 32768)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_2 + (mine + 32768)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_3 + (mine + 32768)), "r"((_packed16_sanitize_2)[0]), "r"((_packed16_sanitize_2)[1]), "r"((_packed16_sanitize_2)[2]), "r"((_packed16_sanitize_2)[3]) : "memory");
    unsigned int raw0_1_3[4];
    raw0_1_3[0] = packets[12];
    raw0_1_3[1] = packets[13];
    raw0_1_3[2] = packets[14];
    raw0_1_3[3] = packets[15];
    uint32_t _packed16_sanitize_3[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_49 = static_cast<uint32_t>(raw0_1_3[_p16s_word]);
        uint32_t _p16s_lo_49 = _p16s_carrier_49 & 0x0000ffffu;
        uint32_t _p16s_hi_49 = (_p16s_carrier_49 >> 16) & 0x0000ffffu;
        if (_p16s_lo_49 == 0x00008000u) _p16s_lo_49 = 0u;
        if (_p16s_hi_49 == 0x00008000u) _p16s_hi_49 = 0u;
        _packed16_sanitize_3[_p16s_word] = _p16s_lo_49 | (_p16s_hi_49 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + (mine + 49152)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + (mine + 49152)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_2 + (mine + 49152)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_3 + (mine + 49152)), "r"((_packed16_sanitize_3)[0]), "r"((_packed16_sanitize_3)[1]), "r"((_packed16_sanitize_3)[2]), "r"((_packed16_sanitize_3)[3]) : "memory");
    unsigned int raw0_1_4[4];
    raw0_1_4[0] = packets[16];
    raw0_1_4[1] = packets[17];
    raw0_1_4[2] = packets[18];
    raw0_1_4[3] = packets[19];
    uint32_t _packed16_sanitize_4[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_50 = static_cast<uint32_t>(raw0_1_4[_p16s_word]);
        uint32_t _p16s_lo_50 = _p16s_carrier_50 & 0x0000ffffu;
        uint32_t _p16s_hi_50 = (_p16s_carrier_50 >> 16) & 0x0000ffffu;
        if (_p16s_lo_50 == 0x00008000u) _p16s_lo_50 = 0u;
        if (_p16s_hi_50 == 0x00008000u) _p16s_hi_50 = 0u;
        _packed16_sanitize_4[_p16s_word] = _p16s_lo_50 | (_p16s_hi_50 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + (mine + 65536)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + (mine + 65536)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_2 + (mine + 65536)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_3 + (mine + 65536)), "r"((_packed16_sanitize_4)[0]), "r"((_packed16_sanitize_4)[1]), "r"((_packed16_sanitize_4)[2]), "r"((_packed16_sanitize_4)[3]) : "memory");
    unsigned int raw0_1_5[4];
    raw0_1_5[0] = packets[20];
    raw0_1_5[1] = packets[21];
    raw0_1_5[2] = packets[22];
    raw0_1_5[3] = packets[23];
    uint32_t _packed16_sanitize_5[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_51 = static_cast<uint32_t>(raw0_1_5[_p16s_word]);
        uint32_t _p16s_lo_51 = _p16s_carrier_51 & 0x0000ffffu;
        uint32_t _p16s_hi_51 = (_p16s_carrier_51 >> 16) & 0x0000ffffu;
        if (_p16s_lo_51 == 0x00008000u) _p16s_lo_51 = 0u;
        if (_p16s_hi_51 == 0x00008000u) _p16s_hi_51 = 0u;
        _packed16_sanitize_5[_p16s_word] = _p16s_lo_51 | (_p16s_hi_51 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + (mine + 81920)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + (mine + 81920)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_2 + (mine + 81920)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_3 + (mine + 81920)), "r"((_packed16_sanitize_5)[0]), "r"((_packed16_sanitize_5)[1]), "r"((_packed16_sanitize_5)[2]), "r"((_packed16_sanitize_5)[3]) : "memory");
    unsigned int raw0_1_6[4];
    raw0_1_6[0] = packets[24];
    raw0_1_6[1] = packets[25];
    raw0_1_6[2] = packets[26];
    raw0_1_6[3] = packets[27];
    uint32_t _packed16_sanitize_6[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_52 = static_cast<uint32_t>(raw0_1_6[_p16s_word]);
        uint32_t _p16s_lo_52 = _p16s_carrier_52 & 0x0000ffffu;
        uint32_t _p16s_hi_52 = (_p16s_carrier_52 >> 16) & 0x0000ffffu;
        if (_p16s_lo_52 == 0x00008000u) _p16s_lo_52 = 0u;
        if (_p16s_hi_52 == 0x00008000u) _p16s_hi_52 = 0u;
        _packed16_sanitize_6[_p16s_word] = _p16s_lo_52 | (_p16s_hi_52 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + (mine + 98304)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + (mine + 98304)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_2 + (mine + 98304)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_3 + (mine + 98304)), "r"((_packed16_sanitize_6)[0]), "r"((_packed16_sanitize_6)[1]), "r"((_packed16_sanitize_6)[2]), "r"((_packed16_sanitize_6)[3]) : "memory");
    unsigned int raw0_1_7[4];
    raw0_1_7[0] = packets[28];
    raw0_1_7[1] = packets[29];
    raw0_1_7[2] = packets[30];
    raw0_1_7[3] = packets[31];
    uint32_t _packed16_sanitize_7[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_53 = static_cast<uint32_t>(raw0_1_7[_p16s_word]);
        uint32_t _p16s_lo_53 = _p16s_carrier_53 & 0x0000ffffu;
        uint32_t _p16s_hi_53 = (_p16s_carrier_53 >> 16) & 0x0000ffffu;
        if (_p16s_lo_53 == 0x00008000u) _p16s_lo_53 = 0u;
        if (_p16s_hi_53 == 0x00008000u) _p16s_hi_53 = 0u;
        _packed16_sanitize_7[_p16s_word] = _p16s_lo_53 | (_p16s_hi_53 << 16);
    }
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_0 + (mine + 114688)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_1 + (mine + 114688)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_2 + (mine + 114688)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peers_3 + (mine + 114688)), "r"((_packed16_sanitize_7)[0]), "r"((_packed16_sanitize_7)[1]), "r"((_packed16_sanitize_7)[2]), "r"((_packed16_sanitize_7)[3]) : "memory");
    float ys[64];
    float ssq8[8];
    uint32_t group_0[64];
    do {
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[0]), "=r"(group_0[1]), "=r"(group_0[2]), "=r"(group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + cur)) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[4]), "=r"(group_0[5]), "=r"(group_0[6]), "=r"(group_0[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 4096))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[8]), "=r"(group_0[9]), "=r"(group_0[10]), "=r"(group_0[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 8192))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[12]), "=r"(group_0[13]), "=r"(group_0[14]), "=r"(group_0[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 12288))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[16]), "=r"(group_0[17]), "=r"(group_0[18]), "=r"(group_0[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 16384))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[20]), "=r"(group_0[21]), "=r"(group_0[22]), "=r"(group_0[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 20480))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[24]), "=r"(group_0[25]), "=r"(group_0[26]), "=r"(group_0[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 24576))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[28]), "=r"(group_0[29]), "=r"(group_0[30]), "=r"(group_0[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 28672))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[32]), "=r"(group_0[33]), "=r"(group_0[34]), "=r"(group_0[35]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 32768))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[36]), "=r"(group_0[37]), "=r"(group_0[38]), "=r"(group_0[39]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 36864))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[40]), "=r"(group_0[41]), "=r"(group_0[42]), "=r"(group_0[43]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 40960))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[44]), "=r"(group_0[45]), "=r"(group_0[46]), "=r"(group_0[47]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 45056))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[48]), "=r"(group_0[49]), "=r"(group_0[50]), "=r"(group_0[51]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 49152))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[52]), "=r"(group_0[53]), "=r"(group_0[54]), "=r"(group_0[55]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 53248))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[56]), "=r"(group_0[57]), "=r"(group_0[58]), "=r"(group_0[59]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 57344))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_0[60]), "=r"(group_0[61]), "=r"(group_0[62]), "=r"(group_0[63]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 61440))) : "memory");
    } while (
        (((group_0[0] >> 0) & 0xffffu) == 0x8000u) || (((group_0[0] >> 16) & 0xffffu) == 0x8000u) || (((group_0[1] >> 0) & 0xffffu) == 0x8000u) || (((group_0[1] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[2] >> 0) & 0xffffu) == 0x8000u) || (((group_0[2] >> 16) & 0xffffu) == 0x8000u) || (((group_0[3] >> 0) & 0xffffu) == 0x8000u) || (((group_0[3] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[4] >> 0) & 0xffffu) == 0x8000u) || (((group_0[4] >> 16) & 0xffffu) == 0x8000u) || (((group_0[5] >> 0) & 0xffffu) == 0x8000u) || (((group_0[5] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[6] >> 0) & 0xffffu) == 0x8000u) || (((group_0[6] >> 16) & 0xffffu) == 0x8000u) || (((group_0[7] >> 0) & 0xffffu) == 0x8000u) || (((group_0[7] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[8] >> 0) & 0xffffu) == 0x8000u) || (((group_0[8] >> 16) & 0xffffu) == 0x8000u) || (((group_0[9] >> 0) & 0xffffu) == 0x8000u) || (((group_0[9] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[10] >> 0) & 0xffffu) == 0x8000u) || (((group_0[10] >> 16) & 0xffffu) == 0x8000u) || (((group_0[11] >> 0) & 0xffffu) == 0x8000u) || (((group_0[11] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[12] >> 0) & 0xffffu) == 0x8000u) || (((group_0[12] >> 16) & 0xffffu) == 0x8000u) || (((group_0[13] >> 0) & 0xffffu) == 0x8000u) || (((group_0[13] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[14] >> 0) & 0xffffu) == 0x8000u) || (((group_0[14] >> 16) & 0xffffu) == 0x8000u) || (((group_0[15] >> 0) & 0xffffu) == 0x8000u) || (((group_0[15] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[16] >> 0) & 0xffffu) == 0x8000u) || (((group_0[16] >> 16) & 0xffffu) == 0x8000u) || (((group_0[17] >> 0) & 0xffffu) == 0x8000u) || (((group_0[17] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[18] >> 0) & 0xffffu) == 0x8000u) || (((group_0[18] >> 16) & 0xffffu) == 0x8000u) || (((group_0[19] >> 0) & 0xffffu) == 0x8000u) || (((group_0[19] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[20] >> 0) & 0xffffu) == 0x8000u) || (((group_0[20] >> 16) & 0xffffu) == 0x8000u) || (((group_0[21] >> 0) & 0xffffu) == 0x8000u) || (((group_0[21] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[22] >> 0) & 0xffffu) == 0x8000u) || (((group_0[22] >> 16) & 0xffffu) == 0x8000u) || (((group_0[23] >> 0) & 0xffffu) == 0x8000u) || (((group_0[23] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[24] >> 0) & 0xffffu) == 0x8000u) || (((group_0[24] >> 16) & 0xffffu) == 0x8000u) || (((group_0[25] >> 0) & 0xffffu) == 0x8000u) || (((group_0[25] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[26] >> 0) & 0xffffu) == 0x8000u) || (((group_0[26] >> 16) & 0xffffu) == 0x8000u) || (((group_0[27] >> 0) & 0xffffu) == 0x8000u) || (((group_0[27] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[28] >> 0) & 0xffffu) == 0x8000u) || (((group_0[28] >> 16) & 0xffffu) == 0x8000u) || (((group_0[29] >> 0) & 0xffffu) == 0x8000u) || (((group_0[29] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[30] >> 0) & 0xffffu) == 0x8000u) || (((group_0[30] >> 16) & 0xffffu) == 0x8000u) || (((group_0[31] >> 0) & 0xffffu) == 0x8000u) || (((group_0[31] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[32] >> 0) & 0xffffu) == 0x8000u) || (((group_0[32] >> 16) & 0xffffu) == 0x8000u) || (((group_0[33] >> 0) & 0xffffu) == 0x8000u) || (((group_0[33] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[34] >> 0) & 0xffffu) == 0x8000u) || (((group_0[34] >> 16) & 0xffffu) == 0x8000u) || (((group_0[35] >> 0) & 0xffffu) == 0x8000u) || (((group_0[35] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[36] >> 0) & 0xffffu) == 0x8000u) || (((group_0[36] >> 16) & 0xffffu) == 0x8000u) || (((group_0[37] >> 0) & 0xffffu) == 0x8000u) || (((group_0[37] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[38] >> 0) & 0xffffu) == 0x8000u) || (((group_0[38] >> 16) & 0xffffu) == 0x8000u) || (((group_0[39] >> 0) & 0xffffu) == 0x8000u) || (((group_0[39] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[40] >> 0) & 0xffffu) == 0x8000u) || (((group_0[40] >> 16) & 0xffffu) == 0x8000u) || (((group_0[41] >> 0) & 0xffffu) == 0x8000u) || (((group_0[41] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[42] >> 0) & 0xffffu) == 0x8000u) || (((group_0[42] >> 16) & 0xffffu) == 0x8000u) || (((group_0[43] >> 0) & 0xffffu) == 0x8000u) || (((group_0[43] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[44] >> 0) & 0xffffu) == 0x8000u) || (((group_0[44] >> 16) & 0xffffu) == 0x8000u) || (((group_0[45] >> 0) & 0xffffu) == 0x8000u) || (((group_0[45] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[46] >> 0) & 0xffffu) == 0x8000u) || (((group_0[46] >> 16) & 0xffffu) == 0x8000u) || (((group_0[47] >> 0) & 0xffffu) == 0x8000u) || (((group_0[47] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[48] >> 0) & 0xffffu) == 0x8000u) || (((group_0[48] >> 16) & 0xffffu) == 0x8000u) || (((group_0[49] >> 0) & 0xffffu) == 0x8000u) || (((group_0[49] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[50] >> 0) & 0xffffu) == 0x8000u) || (((group_0[50] >> 16) & 0xffffu) == 0x8000u) || (((group_0[51] >> 0) & 0xffffu) == 0x8000u) || (((group_0[51] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[52] >> 0) & 0xffffu) == 0x8000u) || (((group_0[52] >> 16) & 0xffffu) == 0x8000u) || (((group_0[53] >> 0) & 0xffffu) == 0x8000u) || (((group_0[53] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[54] >> 0) & 0xffffu) == 0x8000u) || (((group_0[54] >> 16) & 0xffffu) == 0x8000u) || (((group_0[55] >> 0) & 0xffffu) == 0x8000u) || (((group_0[55] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[56] >> 0) & 0xffffu) == 0x8000u) || (((group_0[56] >> 16) & 0xffffu) == 0x8000u) || (((group_0[57] >> 0) & 0xffffu) == 0x8000u) || (((group_0[57] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[58] >> 0) & 0xffffu) == 0x8000u) || (((group_0[58] >> 16) & 0xffffu) == 0x8000u) || (((group_0[59] >> 0) & 0xffffu) == 0x8000u) || (((group_0[59] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[60] >> 0) & 0xffffu) == 0x8000u) || (((group_0[60] >> 16) & 0xffffu) == 0x8000u) || (((group_0[61] >> 0) & 0xffffu) == 0x8000u) || (((group_0[61] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_0[62] >> 0) & 0xffffu) == 0x8000u) || (((group_0[62] >> 16) & 0xffffu) == 0x8000u) || (((group_0[63] >> 0) & 0xffffu) == 0x8000u) || (((group_0[63] >> 16) & 0xffffu) == 0x8000u)
    );
    float s8_0[8];
    float stripe_0[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_0[_pair * 2])[0]), "=f"((&stripe_0[_pair * 2])[1])
            : "r"(group_0[_pair]));
    }
    s8_0[0] = stripe_0[0];
    s8_0[1] = stripe_0[1];
    s8_0[2] = stripe_0[2];
    s8_0[3] = stripe_0[3];
    s8_0[4] = stripe_0[4];
    s8_0[5] = stripe_0[5];
    s8_0[6] = stripe_0[6];
    s8_0[7] = stripe_0[7];
    float stripe_1[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_1[_pair * 2])[0]), "=f"((&stripe_1[_pair * 2])[1])
            : "r"(group_0[4 + _pair]));
    }
    s8_0[0] = s8_0[0] + stripe_1[0];
    s8_0[1] = s8_0[1] + stripe_1[1];
    s8_0[2] = s8_0[2] + stripe_1[2];
    s8_0[3] = s8_0[3] + stripe_1[3];
    s8_0[4] = s8_0[4] + stripe_1[4];
    s8_0[5] = s8_0[5] + stripe_1[5];
    s8_0[6] = s8_0[6] + stripe_1[6];
    s8_0[7] = s8_0[7] + stripe_1[7];
    float stripe_2[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_2[_pair * 2])[0]), "=f"((&stripe_2[_pair * 2])[1])
            : "r"(group_0[8 + _pair]));
    }
    s8_0[0] = s8_0[0] + stripe_2[0];
    s8_0[1] = s8_0[1] + stripe_2[1];
    s8_0[2] = s8_0[2] + stripe_2[2];
    s8_0[3] = s8_0[3] + stripe_2[3];
    s8_0[4] = s8_0[4] + stripe_2[4];
    s8_0[5] = s8_0[5] + stripe_2[5];
    s8_0[6] = s8_0[6] + stripe_2[6];
    s8_0[7] = s8_0[7] + stripe_2[7];
    float stripe_3[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_3[_pair * 2])[0]), "=f"((&stripe_3[_pair * 2])[1])
            : "r"(group_0[12 + _pair]));
    }
    s8_0[0] = s8_0[0] + stripe_3[0];
    s8_0[1] = s8_0[1] + stripe_3[1];
    s8_0[2] = s8_0[2] + stripe_3[2];
    s8_0[3] = s8_0[3] + stripe_3[3];
    s8_0[4] = s8_0[4] + stripe_3[4];
    s8_0[5] = s8_0[5] + stripe_3[5];
    s8_0[6] = s8_0[6] + stripe_3[6];
    s8_0[7] = s8_0[7] + stripe_3[7];
    unsigned int s8_bf16_0[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_0[_lp*2 + 0], s8_0[_lp*2+1 + 0]));
        s8_bf16_0[_lp] = *(uint32_t*)&_bf2;
    }
    float r8_0[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&r8_0[_pair * 2])[0]), "=f"((&r8_0[_pair * 2])[1])
            : "r"(s8_bf16_0[_pair]));
    }
    float ssq_0 = 0.0f;
    float y_0 = ((mode != 0) ? r8_0[0] * gw8[0] : r8_0[0]);
    ys[0] = y_0;
    ssq_0 = __fmaf_rn(y_0, y_0, ssq_0);
    float y_1 = ((mode != 0) ? r8_0[1] * gw8[1] : r8_0[1]);
    ys[1] = y_1;
    ssq_0 = __fmaf_rn(y_1, y_1, ssq_0);
    float y_2 = ((mode != 0) ? r8_0[2] * gw8[2] : r8_0[2]);
    ys[2] = y_2;
    ssq_0 = __fmaf_rn(y_2, y_2, ssq_0);
    float y_3 = ((mode != 0) ? r8_0[3] * gw8[3] : r8_0[3]);
    ys[3] = y_3;
    ssq_0 = __fmaf_rn(y_3, y_3, ssq_0);
    float y_4 = ((mode != 0) ? r8_0[4] * gw8[4] : r8_0[4]);
    ys[4] = y_4;
    ssq_0 = __fmaf_rn(y_4, y_4, ssq_0);
    float y_5 = ((mode != 0) ? r8_0[5] * gw8[5] : r8_0[5]);
    ys[5] = y_5;
    ssq_0 = __fmaf_rn(y_5, y_5, ssq_0);
    float y_6 = ((mode != 0) ? r8_0[6] * gw8[6] : r8_0[6]);
    ys[6] = y_6;
    ssq_0 = __fmaf_rn(y_6, y_6, ssq_0);
    float y_7 = ((mode != 0) ? r8_0[7] * gw8[7] : r8_0[7]);
    ys[7] = y_7;
    ssq_0 = __fmaf_rn(y_7, y_7, ssq_0);
    ssq8[0] = ssq_0;
    float s8_1[8];
    float stripe_4[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_4[_pair * 2])[0]), "=f"((&stripe_4[_pair * 2])[1])
            : "r"(group_0[16 + _pair]));
    }
    s8_1[0] = stripe_4[0];
    s8_1[1] = stripe_4[1];
    s8_1[2] = stripe_4[2];
    s8_1[3] = stripe_4[3];
    s8_1[4] = stripe_4[4];
    s8_1[5] = stripe_4[5];
    s8_1[6] = stripe_4[6];
    s8_1[7] = stripe_4[7];
    float stripe_5[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_5[_pair * 2])[0]), "=f"((&stripe_5[_pair * 2])[1])
            : "r"(group_0[20 + _pair]));
    }
    s8_1[0] = s8_1[0] + stripe_5[0];
    s8_1[1] = s8_1[1] + stripe_5[1];
    s8_1[2] = s8_1[2] + stripe_5[2];
    s8_1[3] = s8_1[3] + stripe_5[3];
    s8_1[4] = s8_1[4] + stripe_5[4];
    s8_1[5] = s8_1[5] + stripe_5[5];
    s8_1[6] = s8_1[6] + stripe_5[6];
    s8_1[7] = s8_1[7] + stripe_5[7];
    float stripe_6[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_6[_pair * 2])[0]), "=f"((&stripe_6[_pair * 2])[1])
            : "r"(group_0[24 + _pair]));
    }
    s8_1[0] = s8_1[0] + stripe_6[0];
    s8_1[1] = s8_1[1] + stripe_6[1];
    s8_1[2] = s8_1[2] + stripe_6[2];
    s8_1[3] = s8_1[3] + stripe_6[3];
    s8_1[4] = s8_1[4] + stripe_6[4];
    s8_1[5] = s8_1[5] + stripe_6[5];
    s8_1[6] = s8_1[6] + stripe_6[6];
    s8_1[7] = s8_1[7] + stripe_6[7];
    float stripe_7[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_7[_pair * 2])[0]), "=f"((&stripe_7[_pair * 2])[1])
            : "r"(group_0[28 + _pair]));
    }
    s8_1[0] = s8_1[0] + stripe_7[0];
    s8_1[1] = s8_1[1] + stripe_7[1];
    s8_1[2] = s8_1[2] + stripe_7[2];
    s8_1[3] = s8_1[3] + stripe_7[3];
    s8_1[4] = s8_1[4] + stripe_7[4];
    s8_1[5] = s8_1[5] + stripe_7[5];
    s8_1[6] = s8_1[6] + stripe_7[6];
    s8_1[7] = s8_1[7] + stripe_7[7];
    unsigned int s8_bf16_1[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_1[_lp*2 + 0], s8_1[_lp*2+1 + 0]));
        s8_bf16_1[_lp] = *(uint32_t*)&_bf2;
    }
    float r8_1[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&r8_1[_pair * 2])[0]), "=f"((&r8_1[_pair * 2])[1])
            : "r"(s8_bf16_1[_pair]));
    }
    float ssq_1 = 0.0f;
    float y_8 = ((mode != 0) ? r8_1[0] * gw8[0] : r8_1[0]);
    ys[8] = y_8;
    ssq_1 = __fmaf_rn(y_8, y_8, ssq_1);
    float y_9 = ((mode != 0) ? r8_1[1] * gw8[1] : r8_1[1]);
    ys[9] = y_9;
    ssq_1 = __fmaf_rn(y_9, y_9, ssq_1);
    float y_10 = ((mode != 0) ? r8_1[2] * gw8[2] : r8_1[2]);
    ys[10] = y_10;
    ssq_1 = __fmaf_rn(y_10, y_10, ssq_1);
    float y_11 = ((mode != 0) ? r8_1[3] * gw8[3] : r8_1[3]);
    ys[11] = y_11;
    ssq_1 = __fmaf_rn(y_11, y_11, ssq_1);
    float y_12 = ((mode != 0) ? r8_1[4] * gw8[4] : r8_1[4]);
    ys[12] = y_12;
    ssq_1 = __fmaf_rn(y_12, y_12, ssq_1);
    float y_13 = ((mode != 0) ? r8_1[5] * gw8[5] : r8_1[5]);
    ys[13] = y_13;
    ssq_1 = __fmaf_rn(y_13, y_13, ssq_1);
    float y_14 = ((mode != 0) ? r8_1[6] * gw8[6] : r8_1[6]);
    ys[14] = y_14;
    ssq_1 = __fmaf_rn(y_14, y_14, ssq_1);
    float y_15 = ((mode != 0) ? r8_1[7] * gw8[7] : r8_1[7]);
    ys[15] = y_15;
    ssq_1 = __fmaf_rn(y_15, y_15, ssq_1);
    ssq8[1] = ssq_1;
    float s8_2[8];
    float stripe_8[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_8[_pair * 2])[0]), "=f"((&stripe_8[_pair * 2])[1])
            : "r"(group_0[32 + _pair]));
    }
    s8_2[0] = stripe_8[0];
    s8_2[1] = stripe_8[1];
    s8_2[2] = stripe_8[2];
    s8_2[3] = stripe_8[3];
    s8_2[4] = stripe_8[4];
    s8_2[5] = stripe_8[5];
    s8_2[6] = stripe_8[6];
    s8_2[7] = stripe_8[7];
    float stripe_9[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_9[_pair * 2])[0]), "=f"((&stripe_9[_pair * 2])[1])
            : "r"(group_0[36 + _pair]));
    }
    s8_2[0] = s8_2[0] + stripe_9[0];
    s8_2[1] = s8_2[1] + stripe_9[1];
    s8_2[2] = s8_2[2] + stripe_9[2];
    s8_2[3] = s8_2[3] + stripe_9[3];
    s8_2[4] = s8_2[4] + stripe_9[4];
    s8_2[5] = s8_2[5] + stripe_9[5];
    s8_2[6] = s8_2[6] + stripe_9[6];
    s8_2[7] = s8_2[7] + stripe_9[7];
    float stripe_10[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_10[_pair * 2])[0]), "=f"((&stripe_10[_pair * 2])[1])
            : "r"(group_0[40 + _pair]));
    }
    s8_2[0] = s8_2[0] + stripe_10[0];
    s8_2[1] = s8_2[1] + stripe_10[1];
    s8_2[2] = s8_2[2] + stripe_10[2];
    s8_2[3] = s8_2[3] + stripe_10[3];
    s8_2[4] = s8_2[4] + stripe_10[4];
    s8_2[5] = s8_2[5] + stripe_10[5];
    s8_2[6] = s8_2[6] + stripe_10[6];
    s8_2[7] = s8_2[7] + stripe_10[7];
    float stripe_11[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_11[_pair * 2])[0]), "=f"((&stripe_11[_pair * 2])[1])
            : "r"(group_0[44 + _pair]));
    }
    s8_2[0] = s8_2[0] + stripe_11[0];
    s8_2[1] = s8_2[1] + stripe_11[1];
    s8_2[2] = s8_2[2] + stripe_11[2];
    s8_2[3] = s8_2[3] + stripe_11[3];
    s8_2[4] = s8_2[4] + stripe_11[4];
    s8_2[5] = s8_2[5] + stripe_11[5];
    s8_2[6] = s8_2[6] + stripe_11[6];
    s8_2[7] = s8_2[7] + stripe_11[7];
    unsigned int s8_bf16_2[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_2[_lp*2 + 0], s8_2[_lp*2+1 + 0]));
        s8_bf16_2[_lp] = *(uint32_t*)&_bf2;
    }
    float r8_2[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&r8_2[_pair * 2])[0]), "=f"((&r8_2[_pair * 2])[1])
            : "r"(s8_bf16_2[_pair]));
    }
    float ssq_2 = 0.0f;
    float y_16 = ((mode != 0) ? r8_2[0] * gw8[0] : r8_2[0]);
    ys[16] = y_16;
    ssq_2 = __fmaf_rn(y_16, y_16, ssq_2);
    float y_17 = ((mode != 0) ? r8_2[1] * gw8[1] : r8_2[1]);
    ys[17] = y_17;
    ssq_2 = __fmaf_rn(y_17, y_17, ssq_2);
    float y_18 = ((mode != 0) ? r8_2[2] * gw8[2] : r8_2[2]);
    ys[18] = y_18;
    ssq_2 = __fmaf_rn(y_18, y_18, ssq_2);
    float y_19 = ((mode != 0) ? r8_2[3] * gw8[3] : r8_2[3]);
    ys[19] = y_19;
    ssq_2 = __fmaf_rn(y_19, y_19, ssq_2);
    float y_20 = ((mode != 0) ? r8_2[4] * gw8[4] : r8_2[4]);
    ys[20] = y_20;
    ssq_2 = __fmaf_rn(y_20, y_20, ssq_2);
    float y_21 = ((mode != 0) ? r8_2[5] * gw8[5] : r8_2[5]);
    ys[21] = y_21;
    ssq_2 = __fmaf_rn(y_21, y_21, ssq_2);
    float y_22 = ((mode != 0) ? r8_2[6] * gw8[6] : r8_2[6]);
    ys[22] = y_22;
    ssq_2 = __fmaf_rn(y_22, y_22, ssq_2);
    float y_23 = ((mode != 0) ? r8_2[7] * gw8[7] : r8_2[7]);
    ys[23] = y_23;
    ssq_2 = __fmaf_rn(y_23, y_23, ssq_2);
    ssq8[2] = ssq_2;
    float s8_3[8];
    float stripe_12[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_12[_pair * 2])[0]), "=f"((&stripe_12[_pair * 2])[1])
            : "r"(group_0[48 + _pair]));
    }
    s8_3[0] = stripe_12[0];
    s8_3[1] = stripe_12[1];
    s8_3[2] = stripe_12[2];
    s8_3[3] = stripe_12[3];
    s8_3[4] = stripe_12[4];
    s8_3[5] = stripe_12[5];
    s8_3[6] = stripe_12[6];
    s8_3[7] = stripe_12[7];
    float stripe_13[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_13[_pair * 2])[0]), "=f"((&stripe_13[_pair * 2])[1])
            : "r"(group_0[52 + _pair]));
    }
    s8_3[0] = s8_3[0] + stripe_13[0];
    s8_3[1] = s8_3[1] + stripe_13[1];
    s8_3[2] = s8_3[2] + stripe_13[2];
    s8_3[3] = s8_3[3] + stripe_13[3];
    s8_3[4] = s8_3[4] + stripe_13[4];
    s8_3[5] = s8_3[5] + stripe_13[5];
    s8_3[6] = s8_3[6] + stripe_13[6];
    s8_3[7] = s8_3[7] + stripe_13[7];
    float stripe_14[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_14[_pair * 2])[0]), "=f"((&stripe_14[_pair * 2])[1])
            : "r"(group_0[56 + _pair]));
    }
    s8_3[0] = s8_3[0] + stripe_14[0];
    s8_3[1] = s8_3[1] + stripe_14[1];
    s8_3[2] = s8_3[2] + stripe_14[2];
    s8_3[3] = s8_3[3] + stripe_14[3];
    s8_3[4] = s8_3[4] + stripe_14[4];
    s8_3[5] = s8_3[5] + stripe_14[5];
    s8_3[6] = s8_3[6] + stripe_14[6];
    s8_3[7] = s8_3[7] + stripe_14[7];
    float stripe_15[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_15[_pair * 2])[0]), "=f"((&stripe_15[_pair * 2])[1])
            : "r"(group_0[60 + _pair]));
    }
    s8_3[0] = s8_3[0] + stripe_15[0];
    s8_3[1] = s8_3[1] + stripe_15[1];
    s8_3[2] = s8_3[2] + stripe_15[2];
    s8_3[3] = s8_3[3] + stripe_15[3];
    s8_3[4] = s8_3[4] + stripe_15[4];
    s8_3[5] = s8_3[5] + stripe_15[5];
    s8_3[6] = s8_3[6] + stripe_15[6];
    s8_3[7] = s8_3[7] + stripe_15[7];
    unsigned int s8_bf16_3[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_3[_lp*2 + 0], s8_3[_lp*2+1 + 0]));
        s8_bf16_3[_lp] = *(uint32_t*)&_bf2;
    }
    float r8_3[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&r8_3[_pair * 2])[0]), "=f"((&r8_3[_pair * 2])[1])
            : "r"(s8_bf16_3[_pair]));
    }
    float ssq_3 = 0.0f;
    float y_24 = ((mode != 0) ? r8_3[0] * gw8[0] : r8_3[0]);
    ys[24] = y_24;
    ssq_3 = __fmaf_rn(y_24, y_24, ssq_3);
    float y_25 = ((mode != 0) ? r8_3[1] * gw8[1] : r8_3[1]);
    ys[25] = y_25;
    ssq_3 = __fmaf_rn(y_25, y_25, ssq_3);
    float y_26 = ((mode != 0) ? r8_3[2] * gw8[2] : r8_3[2]);
    ys[26] = y_26;
    ssq_3 = __fmaf_rn(y_26, y_26, ssq_3);
    float y_27 = ((mode != 0) ? r8_3[3] * gw8[3] : r8_3[3]);
    ys[27] = y_27;
    ssq_3 = __fmaf_rn(y_27, y_27, ssq_3);
    float y_28 = ((mode != 0) ? r8_3[4] * gw8[4] : r8_3[4]);
    ys[28] = y_28;
    ssq_3 = __fmaf_rn(y_28, y_28, ssq_3);
    float y_29 = ((mode != 0) ? r8_3[5] * gw8[5] : r8_3[5]);
    ys[29] = y_29;
    ssq_3 = __fmaf_rn(y_29, y_29, ssq_3);
    float y_30 = ((mode != 0) ? r8_3[6] * gw8[6] : r8_3[6]);
    ys[30] = y_30;
    ssq_3 = __fmaf_rn(y_30, y_30, ssq_3);
    float y_31 = ((mode != 0) ? r8_3[7] * gw8[7] : r8_3[7]);
    ys[31] = y_31;
    ssq_3 = __fmaf_rn(y_31, y_31, ssq_3);
    ssq8[3] = ssq_3;
    uint32_t group_1[64];
    do {
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[0]), "=r"(group_1[1]), "=r"(group_1[2]), "=r"(group_1[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 65536))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[4]), "=r"(group_1[5]), "=r"(group_1[6]), "=r"(group_1[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 69632))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[8]), "=r"(group_1[9]), "=r"(group_1[10]), "=r"(group_1[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 73728))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[12]), "=r"(group_1[13]), "=r"(group_1[14]), "=r"(group_1[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 77824))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[16]), "=r"(group_1[17]), "=r"(group_1[18]), "=r"(group_1[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 81920))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[20]), "=r"(group_1[21]), "=r"(group_1[22]), "=r"(group_1[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 86016))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[24]), "=r"(group_1[25]), "=r"(group_1[26]), "=r"(group_1[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 90112))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[28]), "=r"(group_1[29]), "=r"(group_1[30]), "=r"(group_1[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 94208))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[32]), "=r"(group_1[33]), "=r"(group_1[34]), "=r"(group_1[35]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 98304))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[36]), "=r"(group_1[37]), "=r"(group_1[38]), "=r"(group_1[39]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 102400))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[40]), "=r"(group_1[41]), "=r"(group_1[42]), "=r"(group_1[43]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 106496))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[44]), "=r"(group_1[45]), "=r"(group_1[46]), "=r"(group_1[47]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 110592))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[48]), "=r"(group_1[49]), "=r"(group_1[50]), "=r"(group_1[51]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 114688))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[52]), "=r"(group_1[53]), "=r"(group_1[54]), "=r"(group_1[55]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 118784))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[56]), "=r"(group_1[57]), "=r"(group_1[58]), "=r"(group_1[59]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 122880))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(group_1[60]), "=r"(group_1[61]), "=r"(group_1[62]), "=r"(group_1[63]) : "l"(reinterpret_cast<__nv_bfloat16*>(local + (cur + 126976))) : "memory");
    } while (
        (((group_1[0] >> 0) & 0xffffu) == 0x8000u) || (((group_1[0] >> 16) & 0xffffu) == 0x8000u) || (((group_1[1] >> 0) & 0xffffu) == 0x8000u) || (((group_1[1] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[2] >> 0) & 0xffffu) == 0x8000u) || (((group_1[2] >> 16) & 0xffffu) == 0x8000u) || (((group_1[3] >> 0) & 0xffffu) == 0x8000u) || (((group_1[3] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[4] >> 0) & 0xffffu) == 0x8000u) || (((group_1[4] >> 16) & 0xffffu) == 0x8000u) || (((group_1[5] >> 0) & 0xffffu) == 0x8000u) || (((group_1[5] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[6] >> 0) & 0xffffu) == 0x8000u) || (((group_1[6] >> 16) & 0xffffu) == 0x8000u) || (((group_1[7] >> 0) & 0xffffu) == 0x8000u) || (((group_1[7] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[8] >> 0) & 0xffffu) == 0x8000u) || (((group_1[8] >> 16) & 0xffffu) == 0x8000u) || (((group_1[9] >> 0) & 0xffffu) == 0x8000u) || (((group_1[9] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[10] >> 0) & 0xffffu) == 0x8000u) || (((group_1[10] >> 16) & 0xffffu) == 0x8000u) || (((group_1[11] >> 0) & 0xffffu) == 0x8000u) || (((group_1[11] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[12] >> 0) & 0xffffu) == 0x8000u) || (((group_1[12] >> 16) & 0xffffu) == 0x8000u) || (((group_1[13] >> 0) & 0xffffu) == 0x8000u) || (((group_1[13] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[14] >> 0) & 0xffffu) == 0x8000u) || (((group_1[14] >> 16) & 0xffffu) == 0x8000u) || (((group_1[15] >> 0) & 0xffffu) == 0x8000u) || (((group_1[15] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[16] >> 0) & 0xffffu) == 0x8000u) || (((group_1[16] >> 16) & 0xffffu) == 0x8000u) || (((group_1[17] >> 0) & 0xffffu) == 0x8000u) || (((group_1[17] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[18] >> 0) & 0xffffu) == 0x8000u) || (((group_1[18] >> 16) & 0xffffu) == 0x8000u) || (((group_1[19] >> 0) & 0xffffu) == 0x8000u) || (((group_1[19] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[20] >> 0) & 0xffffu) == 0x8000u) || (((group_1[20] >> 16) & 0xffffu) == 0x8000u) || (((group_1[21] >> 0) & 0xffffu) == 0x8000u) || (((group_1[21] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[22] >> 0) & 0xffffu) == 0x8000u) || (((group_1[22] >> 16) & 0xffffu) == 0x8000u) || (((group_1[23] >> 0) & 0xffffu) == 0x8000u) || (((group_1[23] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[24] >> 0) & 0xffffu) == 0x8000u) || (((group_1[24] >> 16) & 0xffffu) == 0x8000u) || (((group_1[25] >> 0) & 0xffffu) == 0x8000u) || (((group_1[25] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[26] >> 0) & 0xffffu) == 0x8000u) || (((group_1[26] >> 16) & 0xffffu) == 0x8000u) || (((group_1[27] >> 0) & 0xffffu) == 0x8000u) || (((group_1[27] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[28] >> 0) & 0xffffu) == 0x8000u) || (((group_1[28] >> 16) & 0xffffu) == 0x8000u) || (((group_1[29] >> 0) & 0xffffu) == 0x8000u) || (((group_1[29] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[30] >> 0) & 0xffffu) == 0x8000u) || (((group_1[30] >> 16) & 0xffffu) == 0x8000u) || (((group_1[31] >> 0) & 0xffffu) == 0x8000u) || (((group_1[31] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[32] >> 0) & 0xffffu) == 0x8000u) || (((group_1[32] >> 16) & 0xffffu) == 0x8000u) || (((group_1[33] >> 0) & 0xffffu) == 0x8000u) || (((group_1[33] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[34] >> 0) & 0xffffu) == 0x8000u) || (((group_1[34] >> 16) & 0xffffu) == 0x8000u) || (((group_1[35] >> 0) & 0xffffu) == 0x8000u) || (((group_1[35] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[36] >> 0) & 0xffffu) == 0x8000u) || (((group_1[36] >> 16) & 0xffffu) == 0x8000u) || (((group_1[37] >> 0) & 0xffffu) == 0x8000u) || (((group_1[37] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[38] >> 0) & 0xffffu) == 0x8000u) || (((group_1[38] >> 16) & 0xffffu) == 0x8000u) || (((group_1[39] >> 0) & 0xffffu) == 0x8000u) || (((group_1[39] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[40] >> 0) & 0xffffu) == 0x8000u) || (((group_1[40] >> 16) & 0xffffu) == 0x8000u) || (((group_1[41] >> 0) & 0xffffu) == 0x8000u) || (((group_1[41] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[42] >> 0) & 0xffffu) == 0x8000u) || (((group_1[42] >> 16) & 0xffffu) == 0x8000u) || (((group_1[43] >> 0) & 0xffffu) == 0x8000u) || (((group_1[43] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[44] >> 0) & 0xffffu) == 0x8000u) || (((group_1[44] >> 16) & 0xffffu) == 0x8000u) || (((group_1[45] >> 0) & 0xffffu) == 0x8000u) || (((group_1[45] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[46] >> 0) & 0xffffu) == 0x8000u) || (((group_1[46] >> 16) & 0xffffu) == 0x8000u) || (((group_1[47] >> 0) & 0xffffu) == 0x8000u) || (((group_1[47] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[48] >> 0) & 0xffffu) == 0x8000u) || (((group_1[48] >> 16) & 0xffffu) == 0x8000u) || (((group_1[49] >> 0) & 0xffffu) == 0x8000u) || (((group_1[49] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[50] >> 0) & 0xffffu) == 0x8000u) || (((group_1[50] >> 16) & 0xffffu) == 0x8000u) || (((group_1[51] >> 0) & 0xffffu) == 0x8000u) || (((group_1[51] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[52] >> 0) & 0xffffu) == 0x8000u) || (((group_1[52] >> 16) & 0xffffu) == 0x8000u) || (((group_1[53] >> 0) & 0xffffu) == 0x8000u) || (((group_1[53] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[54] >> 0) & 0xffffu) == 0x8000u) || (((group_1[54] >> 16) & 0xffffu) == 0x8000u) || (((group_1[55] >> 0) & 0xffffu) == 0x8000u) || (((group_1[55] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[56] >> 0) & 0xffffu) == 0x8000u) || (((group_1[56] >> 16) & 0xffffu) == 0x8000u) || (((group_1[57] >> 0) & 0xffffu) == 0x8000u) || (((group_1[57] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[58] >> 0) & 0xffffu) == 0x8000u) || (((group_1[58] >> 16) & 0xffffu) == 0x8000u) || (((group_1[59] >> 0) & 0xffffu) == 0x8000u) || (((group_1[59] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[60] >> 0) & 0xffffu) == 0x8000u) || (((group_1[60] >> 16) & 0xffffu) == 0x8000u) || (((group_1[61] >> 0) & 0xffffu) == 0x8000u) || (((group_1[61] >> 16) & 0xffffu) == 0x8000u) ||
        (((group_1[62] >> 0) & 0xffffu) == 0x8000u) || (((group_1[62] >> 16) & 0xffffu) == 0x8000u) || (((group_1[63] >> 0) & 0xffffu) == 0x8000u) || (((group_1[63] >> 16) & 0xffffu) == 0x8000u)
    );
    float s8_4[8];
    float stripe_16[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_16[_pair * 2])[0]), "=f"((&stripe_16[_pair * 2])[1])
            : "r"(group_1[_pair]));
    }
    s8_4[0] = stripe_16[0];
    s8_4[1] = stripe_16[1];
    s8_4[2] = stripe_16[2];
    s8_4[3] = stripe_16[3];
    s8_4[4] = stripe_16[4];
    s8_4[5] = stripe_16[5];
    s8_4[6] = stripe_16[6];
    s8_4[7] = stripe_16[7];
    float stripe_17[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_17[_pair * 2])[0]), "=f"((&stripe_17[_pair * 2])[1])
            : "r"(group_1[4 + _pair]));
    }
    s8_4[0] = s8_4[0] + stripe_17[0];
    s8_4[1] = s8_4[1] + stripe_17[1];
    s8_4[2] = s8_4[2] + stripe_17[2];
    s8_4[3] = s8_4[3] + stripe_17[3];
    s8_4[4] = s8_4[4] + stripe_17[4];
    s8_4[5] = s8_4[5] + stripe_17[5];
    s8_4[6] = s8_4[6] + stripe_17[6];
    s8_4[7] = s8_4[7] + stripe_17[7];
    float stripe_18[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_18[_pair * 2])[0]), "=f"((&stripe_18[_pair * 2])[1])
            : "r"(group_1[8 + _pair]));
    }
    s8_4[0] = s8_4[0] + stripe_18[0];
    s8_4[1] = s8_4[1] + stripe_18[1];
    s8_4[2] = s8_4[2] + stripe_18[2];
    s8_4[3] = s8_4[3] + stripe_18[3];
    s8_4[4] = s8_4[4] + stripe_18[4];
    s8_4[5] = s8_4[5] + stripe_18[5];
    s8_4[6] = s8_4[6] + stripe_18[6];
    s8_4[7] = s8_4[7] + stripe_18[7];
    float stripe_19[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_19[_pair * 2])[0]), "=f"((&stripe_19[_pair * 2])[1])
            : "r"(group_1[12 + _pair]));
    }
    s8_4[0] = s8_4[0] + stripe_19[0];
    s8_4[1] = s8_4[1] + stripe_19[1];
    s8_4[2] = s8_4[2] + stripe_19[2];
    s8_4[3] = s8_4[3] + stripe_19[3];
    s8_4[4] = s8_4[4] + stripe_19[4];
    s8_4[5] = s8_4[5] + stripe_19[5];
    s8_4[6] = s8_4[6] + stripe_19[6];
    s8_4[7] = s8_4[7] + stripe_19[7];
    unsigned int s8_bf16_4[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_4[_lp*2 + 0], s8_4[_lp*2+1 + 0]));
        s8_bf16_4[_lp] = *(uint32_t*)&_bf2;
    }
    float r8_4[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&r8_4[_pair * 2])[0]), "=f"((&r8_4[_pair * 2])[1])
            : "r"(s8_bf16_4[_pair]));
    }
    float ssq_4 = 0.0f;
    float y_32 = ((mode != 0) ? r8_4[0] * gw8[0] : r8_4[0]);
    ys[32] = y_32;
    ssq_4 = __fmaf_rn(y_32, y_32, ssq_4);
    float y_33 = ((mode != 0) ? r8_4[1] * gw8[1] : r8_4[1]);
    ys[33] = y_33;
    ssq_4 = __fmaf_rn(y_33, y_33, ssq_4);
    float y_34 = ((mode != 0) ? r8_4[2] * gw8[2] : r8_4[2]);
    ys[34] = y_34;
    ssq_4 = __fmaf_rn(y_34, y_34, ssq_4);
    float y_35 = ((mode != 0) ? r8_4[3] * gw8[3] : r8_4[3]);
    ys[35] = y_35;
    ssq_4 = __fmaf_rn(y_35, y_35, ssq_4);
    float y_36 = ((mode != 0) ? r8_4[4] * gw8[4] : r8_4[4]);
    ys[36] = y_36;
    ssq_4 = __fmaf_rn(y_36, y_36, ssq_4);
    float y_37 = ((mode != 0) ? r8_4[5] * gw8[5] : r8_4[5]);
    ys[37] = y_37;
    ssq_4 = __fmaf_rn(y_37, y_37, ssq_4);
    float y_38 = ((mode != 0) ? r8_4[6] * gw8[6] : r8_4[6]);
    ys[38] = y_38;
    ssq_4 = __fmaf_rn(y_38, y_38, ssq_4);
    float y_39 = ((mode != 0) ? r8_4[7] * gw8[7] : r8_4[7]);
    ys[39] = y_39;
    ssq_4 = __fmaf_rn(y_39, y_39, ssq_4);
    ssq8[4] = ssq_4;
    float s8_5[8];
    float stripe_20[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_20[_pair * 2])[0]), "=f"((&stripe_20[_pair * 2])[1])
            : "r"(group_1[16 + _pair]));
    }
    s8_5[0] = stripe_20[0];
    s8_5[1] = stripe_20[1];
    s8_5[2] = stripe_20[2];
    s8_5[3] = stripe_20[3];
    s8_5[4] = stripe_20[4];
    s8_5[5] = stripe_20[5];
    s8_5[6] = stripe_20[6];
    s8_5[7] = stripe_20[7];
    float stripe_21[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_21[_pair * 2])[0]), "=f"((&stripe_21[_pair * 2])[1])
            : "r"(group_1[20 + _pair]));
    }
    s8_5[0] = s8_5[0] + stripe_21[0];
    s8_5[1] = s8_5[1] + stripe_21[1];
    s8_5[2] = s8_5[2] + stripe_21[2];
    s8_5[3] = s8_5[3] + stripe_21[3];
    s8_5[4] = s8_5[4] + stripe_21[4];
    s8_5[5] = s8_5[5] + stripe_21[5];
    s8_5[6] = s8_5[6] + stripe_21[6];
    s8_5[7] = s8_5[7] + stripe_21[7];
    float stripe_22[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_22[_pair * 2])[0]), "=f"((&stripe_22[_pair * 2])[1])
            : "r"(group_1[24 + _pair]));
    }
    s8_5[0] = s8_5[0] + stripe_22[0];
    s8_5[1] = s8_5[1] + stripe_22[1];
    s8_5[2] = s8_5[2] + stripe_22[2];
    s8_5[3] = s8_5[3] + stripe_22[3];
    s8_5[4] = s8_5[4] + stripe_22[4];
    s8_5[5] = s8_5[5] + stripe_22[5];
    s8_5[6] = s8_5[6] + stripe_22[6];
    s8_5[7] = s8_5[7] + stripe_22[7];
    float stripe_23[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_23[_pair * 2])[0]), "=f"((&stripe_23[_pair * 2])[1])
            : "r"(group_1[28 + _pair]));
    }
    s8_5[0] = s8_5[0] + stripe_23[0];
    s8_5[1] = s8_5[1] + stripe_23[1];
    s8_5[2] = s8_5[2] + stripe_23[2];
    s8_5[3] = s8_5[3] + stripe_23[3];
    s8_5[4] = s8_5[4] + stripe_23[4];
    s8_5[5] = s8_5[5] + stripe_23[5];
    s8_5[6] = s8_5[6] + stripe_23[6];
    s8_5[7] = s8_5[7] + stripe_23[7];
    unsigned int s8_bf16_5[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_5[_lp*2 + 0], s8_5[_lp*2+1 + 0]));
        s8_bf16_5[_lp] = *(uint32_t*)&_bf2;
    }
    float r8_5[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&r8_5[_pair * 2])[0]), "=f"((&r8_5[_pair * 2])[1])
            : "r"(s8_bf16_5[_pair]));
    }
    float ssq_5 = 0.0f;
    float y_40 = ((mode != 0) ? r8_5[0] * gw8[0] : r8_5[0]);
    ys[40] = y_40;
    ssq_5 = __fmaf_rn(y_40, y_40, ssq_5);
    float y_41 = ((mode != 0) ? r8_5[1] * gw8[1] : r8_5[1]);
    ys[41] = y_41;
    ssq_5 = __fmaf_rn(y_41, y_41, ssq_5);
    float y_42 = ((mode != 0) ? r8_5[2] * gw8[2] : r8_5[2]);
    ys[42] = y_42;
    ssq_5 = __fmaf_rn(y_42, y_42, ssq_5);
    float y_43 = ((mode != 0) ? r8_5[3] * gw8[3] : r8_5[3]);
    ys[43] = y_43;
    ssq_5 = __fmaf_rn(y_43, y_43, ssq_5);
    float y_44 = ((mode != 0) ? r8_5[4] * gw8[4] : r8_5[4]);
    ys[44] = y_44;
    ssq_5 = __fmaf_rn(y_44, y_44, ssq_5);
    float y_45 = ((mode != 0) ? r8_5[5] * gw8[5] : r8_5[5]);
    ys[45] = y_45;
    ssq_5 = __fmaf_rn(y_45, y_45, ssq_5);
    float y_46 = ((mode != 0) ? r8_5[6] * gw8[6] : r8_5[6]);
    ys[46] = y_46;
    ssq_5 = __fmaf_rn(y_46, y_46, ssq_5);
    float y_47 = ((mode != 0) ? r8_5[7] * gw8[7] : r8_5[7]);
    ys[47] = y_47;
    ssq_5 = __fmaf_rn(y_47, y_47, ssq_5);
    ssq8[5] = ssq_5;
    float s8_6[8];
    float stripe_24[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_24[_pair * 2])[0]), "=f"((&stripe_24[_pair * 2])[1])
            : "r"(group_1[32 + _pair]));
    }
    s8_6[0] = stripe_24[0];
    s8_6[1] = stripe_24[1];
    s8_6[2] = stripe_24[2];
    s8_6[3] = stripe_24[3];
    s8_6[4] = stripe_24[4];
    s8_6[5] = stripe_24[5];
    s8_6[6] = stripe_24[6];
    s8_6[7] = stripe_24[7];
    float stripe_25[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_25[_pair * 2])[0]), "=f"((&stripe_25[_pair * 2])[1])
            : "r"(group_1[36 + _pair]));
    }
    s8_6[0] = s8_6[0] + stripe_25[0];
    s8_6[1] = s8_6[1] + stripe_25[1];
    s8_6[2] = s8_6[2] + stripe_25[2];
    s8_6[3] = s8_6[3] + stripe_25[3];
    s8_6[4] = s8_6[4] + stripe_25[4];
    s8_6[5] = s8_6[5] + stripe_25[5];
    s8_6[6] = s8_6[6] + stripe_25[6];
    s8_6[7] = s8_6[7] + stripe_25[7];
    float stripe_26[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_26[_pair * 2])[0]), "=f"((&stripe_26[_pair * 2])[1])
            : "r"(group_1[40 + _pair]));
    }
    s8_6[0] = s8_6[0] + stripe_26[0];
    s8_6[1] = s8_6[1] + stripe_26[1];
    s8_6[2] = s8_6[2] + stripe_26[2];
    s8_6[3] = s8_6[3] + stripe_26[3];
    s8_6[4] = s8_6[4] + stripe_26[4];
    s8_6[5] = s8_6[5] + stripe_26[5];
    s8_6[6] = s8_6[6] + stripe_26[6];
    s8_6[7] = s8_6[7] + stripe_26[7];
    float stripe_27[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_27[_pair * 2])[0]), "=f"((&stripe_27[_pair * 2])[1])
            : "r"(group_1[44 + _pair]));
    }
    s8_6[0] = s8_6[0] + stripe_27[0];
    s8_6[1] = s8_6[1] + stripe_27[1];
    s8_6[2] = s8_6[2] + stripe_27[2];
    s8_6[3] = s8_6[3] + stripe_27[3];
    s8_6[4] = s8_6[4] + stripe_27[4];
    s8_6[5] = s8_6[5] + stripe_27[5];
    s8_6[6] = s8_6[6] + stripe_27[6];
    s8_6[7] = s8_6[7] + stripe_27[7];
    unsigned int s8_bf16_6[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_6[_lp*2 + 0], s8_6[_lp*2+1 + 0]));
        s8_bf16_6[_lp] = *(uint32_t*)&_bf2;
    }
    float r8_6[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&r8_6[_pair * 2])[0]), "=f"((&r8_6[_pair * 2])[1])
            : "r"(s8_bf16_6[_pair]));
    }
    float ssq_6 = 0.0f;
    float y_48 = ((mode != 0) ? r8_6[0] * gw8[0] : r8_6[0]);
    ys[48] = y_48;
    ssq_6 = __fmaf_rn(y_48, y_48, ssq_6);
    float y_49 = ((mode != 0) ? r8_6[1] * gw8[1] : r8_6[1]);
    ys[49] = y_49;
    ssq_6 = __fmaf_rn(y_49, y_49, ssq_6);
    float y_50 = ((mode != 0) ? r8_6[2] * gw8[2] : r8_6[2]);
    ys[50] = y_50;
    ssq_6 = __fmaf_rn(y_50, y_50, ssq_6);
    float y_51 = ((mode != 0) ? r8_6[3] * gw8[3] : r8_6[3]);
    ys[51] = y_51;
    ssq_6 = __fmaf_rn(y_51, y_51, ssq_6);
    float y_52 = ((mode != 0) ? r8_6[4] * gw8[4] : r8_6[4]);
    ys[52] = y_52;
    ssq_6 = __fmaf_rn(y_52, y_52, ssq_6);
    float y_53 = ((mode != 0) ? r8_6[5] * gw8[5] : r8_6[5]);
    ys[53] = y_53;
    ssq_6 = __fmaf_rn(y_53, y_53, ssq_6);
    float y_54 = ((mode != 0) ? r8_6[6] * gw8[6] : r8_6[6]);
    ys[54] = y_54;
    ssq_6 = __fmaf_rn(y_54, y_54, ssq_6);
    float y_55 = ((mode != 0) ? r8_6[7] * gw8[7] : r8_6[7]);
    ys[55] = y_55;
    ssq_6 = __fmaf_rn(y_55, y_55, ssq_6);
    ssq8[6] = ssq_6;
    float s8_7[8];
    float stripe_28[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_28[_pair * 2])[0]), "=f"((&stripe_28[_pair * 2])[1])
            : "r"(group_1[48 + _pair]));
    }
    s8_7[0] = stripe_28[0];
    s8_7[1] = stripe_28[1];
    s8_7[2] = stripe_28[2];
    s8_7[3] = stripe_28[3];
    s8_7[4] = stripe_28[4];
    s8_7[5] = stripe_28[5];
    s8_7[6] = stripe_28[6];
    s8_7[7] = stripe_28[7];
    float stripe_29[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_29[_pair * 2])[0]), "=f"((&stripe_29[_pair * 2])[1])
            : "r"(group_1[52 + _pair]));
    }
    s8_7[0] = s8_7[0] + stripe_29[0];
    s8_7[1] = s8_7[1] + stripe_29[1];
    s8_7[2] = s8_7[2] + stripe_29[2];
    s8_7[3] = s8_7[3] + stripe_29[3];
    s8_7[4] = s8_7[4] + stripe_29[4];
    s8_7[5] = s8_7[5] + stripe_29[5];
    s8_7[6] = s8_7[6] + stripe_29[6];
    s8_7[7] = s8_7[7] + stripe_29[7];
    float stripe_30[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_30[_pair * 2])[0]), "=f"((&stripe_30[_pair * 2])[1])
            : "r"(group_1[56 + _pair]));
    }
    s8_7[0] = s8_7[0] + stripe_30[0];
    s8_7[1] = s8_7[1] + stripe_30[1];
    s8_7[2] = s8_7[2] + stripe_30[2];
    s8_7[3] = s8_7[3] + stripe_30[3];
    s8_7[4] = s8_7[4] + stripe_30[4];
    s8_7[5] = s8_7[5] + stripe_30[5];
    s8_7[6] = s8_7[6] + stripe_30[6];
    s8_7[7] = s8_7[7] + stripe_30[7];
    float stripe_31[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&stripe_31[_pair * 2])[0]), "=f"((&stripe_31[_pair * 2])[1])
            : "r"(group_1[60 + _pair]));
    }
    s8_7[0] = s8_7[0] + stripe_31[0];
    s8_7[1] = s8_7[1] + stripe_31[1];
    s8_7[2] = s8_7[2] + stripe_31[2];
    s8_7[3] = s8_7[3] + stripe_31[3];
    s8_7[4] = s8_7[4] + stripe_31[4];
    s8_7[5] = s8_7[5] + stripe_31[5];
    s8_7[6] = s8_7[6] + stripe_31[6];
    s8_7[7] = s8_7[7] + stripe_31[7];
    unsigned int s8_bf16_7[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_7[_lp*2 + 0], s8_7[_lp*2+1 + 0]));
        s8_bf16_7[_lp] = *(uint32_t*)&_bf2;
    }
    float r8_7[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&r8_7[_pair * 2])[0]), "=f"((&r8_7[_pair * 2])[1])
            : "r"(s8_bf16_7[_pair]));
    }
    float ssq_7 = 0.0f;
    float y_56 = ((mode != 0) ? r8_7[0] * gw8[0] : r8_7[0]);
    ys[56] = y_56;
    ssq_7 = __fmaf_rn(y_56, y_56, ssq_7);
    float y_57 = ((mode != 0) ? r8_7[1] * gw8[1] : r8_7[1]);
    ys[57] = y_57;
    ssq_7 = __fmaf_rn(y_57, y_57, ssq_7);
    float y_58 = ((mode != 0) ? r8_7[2] * gw8[2] : r8_7[2]);
    ys[58] = y_58;
    ssq_7 = __fmaf_rn(y_58, y_58, ssq_7);
    float y_59 = ((mode != 0) ? r8_7[3] * gw8[3] : r8_7[3]);
    ys[59] = y_59;
    ssq_7 = __fmaf_rn(y_59, y_59, ssq_7);
    float y_60 = ((mode != 0) ? r8_7[4] * gw8[4] : r8_7[4]);
    ys[60] = y_60;
    ssq_7 = __fmaf_rn(y_60, y_60, ssq_7);
    float y_61 = ((mode != 0) ? r8_7[5] * gw8[5] : r8_7[5]);
    ys[61] = y_61;
    ssq_7 = __fmaf_rn(y_61, y_61, ssq_7);
    float y_62 = ((mode != 0) ? r8_7[6] * gw8[6] : r8_7[6]);
    ys[62] = y_62;
    ssq_7 = __fmaf_rn(y_62, y_62, ssq_7);
    float y_63 = ((mode != 0) ? r8_7[7] * gw8[7] : r8_7[7]);
    ys[63] = y_63;
    ssq_7 = __fmaf_rn(y_63, y_63, ssq_7);
    ssq8[7] = ssq_7;
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
        uint32_t _cluster_mapa_54;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_54) : "r"(slot_addr), "r"(0));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_54), "f"(part) : "memory");
        uint32_t _cluster_mapa_55;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_55) : "r"(slot_addr), "r"(1));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_55), "f"(part) : "memory");
        uint32_t _cluster_mapa_56;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_56) : "r"(slot_addr), "r"(2));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_56), "f"(part) : "memory");
        uint32_t _cluster_mapa_57;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_57) : "r"(slot_addr), "r"(3));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_57), "f"(part) : "memory");
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
    float _sqrt_58;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_58) : "f"(__fmaf_rn(tot8[0], 0.00024414062f, eps)));
    float rs_0 = __frcp_rn(_sqrt_58);
    float o_0 = ys[0] * rs_0;
    float o2_0 = ((mode != 0) ? o_0 : o_0 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_0, w8[0], acc8[0]);
    float o_1 = ys[1] * rs_0;
    float o2_1 = ((mode != 0) ? o_1 : o_1 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_1, w8[0], acc8[1]);
    float o_2 = ys[2] * rs_0;
    float o2_2 = ((mode != 0) ? o_2 : o_2 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_2, w8[0], acc8[2]);
    float o_3 = ys[3] * rs_0;
    float o2_3 = ((mode != 0) ? o_3 : o_3 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_3, w8[0], acc8[3]);
    float o_4 = ys[4] * rs_0;
    float o2_4 = ((mode != 0) ? o_4 : o_4 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_4, w8[0], acc8[4]);
    float o_5 = ys[5] * rs_0;
    float o2_5 = ((mode != 0) ? o_5 : o_5 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_5, w8[0], acc8[5]);
    float o_6 = ys[6] * rs_0;
    float o2_6 = ((mode != 0) ? o_6 : o_6 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_6, w8[0], acc8[6]);
    float o_7 = ys[7] * rs_0;
    float o2_7 = ((mode != 0) ? o_7 : o_7 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_7, w8[0], acc8[7]);
    float _sqrt_59;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_59) : "f"(__fmaf_rn(tot8[1], 0.00024414062f, eps)));
    float rs_1 = __frcp_rn(_sqrt_59);
    float o_8 = ys[8] * rs_1;
    float o2_8 = ((mode != 0) ? o_8 : o_8 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_8, w8[1], acc8[0]);
    float o_9 = ys[9] * rs_1;
    float o2_9 = ((mode != 0) ? o_9 : o_9 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_9, w8[1], acc8[1]);
    float o_10 = ys[10] * rs_1;
    float o2_10 = ((mode != 0) ? o_10 : o_10 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_10, w8[1], acc8[2]);
    float o_11 = ys[11] * rs_1;
    float o2_11 = ((mode != 0) ? o_11 : o_11 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_11, w8[1], acc8[3]);
    float o_12 = ys[12] * rs_1;
    float o2_12 = ((mode != 0) ? o_12 : o_12 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_12, w8[1], acc8[4]);
    float o_13 = ys[13] * rs_1;
    float o2_13 = ((mode != 0) ? o_13 : o_13 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_13, w8[1], acc8[5]);
    float o_14 = ys[14] * rs_1;
    float o2_14 = ((mode != 0) ? o_14 : o_14 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_14, w8[1], acc8[6]);
    float o_15 = ys[15] * rs_1;
    float o2_15 = ((mode != 0) ? o_15 : o_15 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_15, w8[1], acc8[7]);
    float _sqrt_60;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_60) : "f"(__fmaf_rn(tot8[2], 0.00024414062f, eps)));
    float rs_2 = __frcp_rn(_sqrt_60);
    float o_16 = ys[16] * rs_2;
    float o2_16 = ((mode != 0) ? o_16 : o_16 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_16, w8[2], acc8[0]);
    float o_17 = ys[17] * rs_2;
    float o2_17 = ((mode != 0) ? o_17 : o_17 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_17, w8[2], acc8[1]);
    float o_18 = ys[18] * rs_2;
    float o2_18 = ((mode != 0) ? o_18 : o_18 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_18, w8[2], acc8[2]);
    float o_19 = ys[19] * rs_2;
    float o2_19 = ((mode != 0) ? o_19 : o_19 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_19, w8[2], acc8[3]);
    float o_20 = ys[20] * rs_2;
    float o2_20 = ((mode != 0) ? o_20 : o_20 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_20, w8[2], acc8[4]);
    float o_21 = ys[21] * rs_2;
    float o2_21 = ((mode != 0) ? o_21 : o_21 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_21, w8[2], acc8[5]);
    float o_22 = ys[22] * rs_2;
    float o2_22 = ((mode != 0) ? o_22 : o_22 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_22, w8[2], acc8[6]);
    float o_23 = ys[23] * rs_2;
    float o2_23 = ((mode != 0) ? o_23 : o_23 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_23, w8[2], acc8[7]);
    float _sqrt_61;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_61) : "f"(__fmaf_rn(tot8[3], 0.00024414062f, eps)));
    float rs_3 = __frcp_rn(_sqrt_61);
    float o_24 = ys[24] * rs_3;
    float o2_24 = ((mode != 0) ? o_24 : o_24 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_24, w8[3], acc8[0]);
    float o_25 = ys[25] * rs_3;
    float o2_25 = ((mode != 0) ? o_25 : o_25 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_25, w8[3], acc8[1]);
    float o_26 = ys[26] * rs_3;
    float o2_26 = ((mode != 0) ? o_26 : o_26 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_26, w8[3], acc8[2]);
    float o_27 = ys[27] * rs_3;
    float o2_27 = ((mode != 0) ? o_27 : o_27 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_27, w8[3], acc8[3]);
    float o_28 = ys[28] * rs_3;
    float o2_28 = ((mode != 0) ? o_28 : o_28 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_28, w8[3], acc8[4]);
    float o_29 = ys[29] * rs_3;
    float o2_29 = ((mode != 0) ? o_29 : o_29 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_29, w8[3], acc8[5]);
    float o_30 = ys[30] * rs_3;
    float o2_30 = ((mode != 0) ? o_30 : o_30 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_30, w8[3], acc8[6]);
    float o_31 = ys[31] * rs_3;
    float o2_31 = ((mode != 0) ? o_31 : o_31 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_31, w8[3], acc8[7]);
    float _sqrt_62;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_62) : "f"(__fmaf_rn(tot8[4], 0.00024414062f, eps)));
    float rs_4 = __frcp_rn(_sqrt_62);
    float o_32 = ys[32] * rs_4;
    float o2_32 = ((mode != 0) ? o_32 : o_32 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_32, w8[4], acc8[0]);
    float o_33 = ys[33] * rs_4;
    float o2_33 = ((mode != 0) ? o_33 : o_33 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_33, w8[4], acc8[1]);
    float o_34 = ys[34] * rs_4;
    float o2_34 = ((mode != 0) ? o_34 : o_34 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_34, w8[4], acc8[2]);
    float o_35 = ys[35] * rs_4;
    float o2_35 = ((mode != 0) ? o_35 : o_35 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_35, w8[4], acc8[3]);
    float o_36 = ys[36] * rs_4;
    float o2_36 = ((mode != 0) ? o_36 : o_36 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_36, w8[4], acc8[4]);
    float o_37 = ys[37] * rs_4;
    float o2_37 = ((mode != 0) ? o_37 : o_37 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_37, w8[4], acc8[5]);
    float o_38 = ys[38] * rs_4;
    float o2_38 = ((mode != 0) ? o_38 : o_38 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_38, w8[4], acc8[6]);
    float o_39 = ys[39] * rs_4;
    float o2_39 = ((mode != 0) ? o_39 : o_39 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_39, w8[4], acc8[7]);
    float _sqrt_63;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_63) : "f"(__fmaf_rn(tot8[5], 0.00024414062f, eps)));
    float rs_5 = __frcp_rn(_sqrt_63);
    float o_40 = ys[40] * rs_5;
    float o2_40 = ((mode != 0) ? o_40 : o_40 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_40, w8[5], acc8[0]);
    float o_41 = ys[41] * rs_5;
    float o2_41 = ((mode != 0) ? o_41 : o_41 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_41, w8[5], acc8[1]);
    float o_42 = ys[42] * rs_5;
    float o2_42 = ((mode != 0) ? o_42 : o_42 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_42, w8[5], acc8[2]);
    float o_43 = ys[43] * rs_5;
    float o2_43 = ((mode != 0) ? o_43 : o_43 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_43, w8[5], acc8[3]);
    float o_44 = ys[44] * rs_5;
    float o2_44 = ((mode != 0) ? o_44 : o_44 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_44, w8[5], acc8[4]);
    float o_45 = ys[45] * rs_5;
    float o2_45 = ((mode != 0) ? o_45 : o_45 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_45, w8[5], acc8[5]);
    float o_46 = ys[46] * rs_5;
    float o2_46 = ((mode != 0) ? o_46 : o_46 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_46, w8[5], acc8[6]);
    float o_47 = ys[47] * rs_5;
    float o2_47 = ((mode != 0) ? o_47 : o_47 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_47, w8[5], acc8[7]);
    float _sqrt_64;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_64) : "f"(__fmaf_rn(tot8[6], 0.00024414062f, eps)));
    float rs_6 = __frcp_rn(_sqrt_64);
    float o_48 = ys[48] * rs_6;
    float o2_48 = ((mode != 0) ? o_48 : o_48 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_48, w8[6], acc8[0]);
    float o_49 = ys[49] * rs_6;
    float o2_49 = ((mode != 0) ? o_49 : o_49 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_49, w8[6], acc8[1]);
    float o_50 = ys[50] * rs_6;
    float o2_50 = ((mode != 0) ? o_50 : o_50 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_50, w8[6], acc8[2]);
    float o_51 = ys[51] * rs_6;
    float o2_51 = ((mode != 0) ? o_51 : o_51 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_51, w8[6], acc8[3]);
    float o_52 = ys[52] * rs_6;
    float o2_52 = ((mode != 0) ? o_52 : o_52 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_52, w8[6], acc8[4]);
    float o_53 = ys[53] * rs_6;
    float o2_53 = ((mode != 0) ? o_53 : o_53 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_53, w8[6], acc8[5]);
    float o_54 = ys[54] * rs_6;
    float o2_54 = ((mode != 0) ? o_54 : o_54 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_54, w8[6], acc8[6]);
    float o_55 = ys[55] * rs_6;
    float o2_55 = ((mode != 0) ? o_55 : o_55 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_55, w8[6], acc8[7]);
    float _sqrt_65;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_65) : "f"(__fmaf_rn(tot8[7], 0.00024414062f, eps)));
    float rs_7 = __frcp_rn(_sqrt_65);
    float o_56 = ys[56] * rs_7;
    float o2_56 = ((mode != 0) ? o_56 : o_56 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_56, w8[7], acc8[0]);
    float o_57 = ys[57] * rs_7;
    float o2_57 = ((mode != 0) ? o_57 : o_57 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_57, w8[7], acc8[1]);
    float o_58 = ys[58] * rs_7;
    float o2_58 = ((mode != 0) ? o_58 : o_58 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_58, w8[7], acc8[2]);
    float o_59 = ys[59] * rs_7;
    float o2_59 = ((mode != 0) ? o_59 : o_59 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_59, w8[7], acc8[3]);
    float o_60 = ys[60] * rs_7;
    float o2_60 = ((mode != 0) ? o_60 : o_60 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_60, w8[7], acc8[4]);
    float o_61 = ys[61] * rs_7;
    float o2_61 = ((mode != 0) ? o_61 : o_61 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_61, w8[7], acc8[5]);
    float o_62 = ys[62] * rs_7;
    float o2_62 = ((mode != 0) ? o_62 : o_62 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_62, w8[7], acc8[6]);
    float o_63 = ys[63] * rs_7;
    float o2_63 = ((mode != 0) ? o_63 : o_63 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_63, w8[7], acc8[7]);
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
