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
#define SMEM_WARP_TAB_STAGE_BYTES 4
#define SMEM_WARP_TAB_STRIDE 4
#define SMEM_CTA_TAB_OFF 4
#define SMEM_CTA_TAB_STAGE_BYTES 64
#define SMEM_CTA_TAB_STRIDE 64
#define SMEM_TOTAL 128
#define THREADS 32
#define LAUNCH_MIN_BLOCKS 1

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

__global__ __launch_bounds__(32, LAUNCH_MIN_BLOCKS) __cluster_dims__(16,1,1) void
kernel_cake_tp_moe_finalize_rmsnorm_6b793225024afe0c9590(__nv_bfloat16* __restrict__ gemm2, int row_stride, int rows, int* __restrict__ idx, unsigned int* __restrict__ expert_weights, int weights_fp32, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ output, int n_tok, int slots, float eps, float gain_center, int mode, int rank, long long* __restrict__ peer_table, int* __restrict__ gen, unsigned long long b_mcast)
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
    const unsigned int clusters_x = gridDim.x / 16;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 16;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* warp_tab = reinterpret_cast<float*>(smem_raw + 0);
    const int warp_tab_addr = smem + 0;
    float* cta_tab = reinterpret_cast<float*>(smem_raw + 4);
    const int cta_tab_addr = smem + 4;

    // === Task calls (declaration order) ===
    int slot = cluster_id;
    int cta_rank_0 = cta_rank;
    if (__syncthreads_and(slot >= n_tok)) return;
    uint32_t _smid_read_0;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(_smid_read_0));
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    int col = (cta_rank_0 * 32 + tid) * 8;
    long long col64 = (long long)col;
    long long slot64 = (long long)slot;
    long long rank64 = (long long)rank;
    int rank_i = rank;
    int g = gen[slot];
    int g_old = ((g == 0) ? 2 : g - 1);
    long long cur_slot = (long long)g * (long long)slots + slot64;
    long long old_slot = (long long)g_old * (long long)slots + slot64;
    long long a_cur = cur_slot * 32768 + col64;
    long long a_old = old_slot * 32768 + col64;
    long long b_cur = cur_slot * 32768 + col64;
    long long b_old = old_slot * 32768 + col64;
    long long a_local_addr = peer_table[rank];
    long long b_local_addr = peer_table[8 + rank];
    __nv_bfloat16* a_local = reinterpret_cast<__nv_bfloat16*>(a_local_addr);
    unsigned int* b_local = reinterpret_cast<unsigned int*>(b_local_addr);
    long long a_peer_addr0 = peer_table[0];
    long long a_peer_addr1 = peer_table[1];
    long long a_peer_addr2 = peer_table[2];
    long long a_peer_addr3 = peer_table[3];
    long long a_peer_addr4 = peer_table[4];
    long long a_peer_addr5 = peer_table[5];
    long long a_peer_addr6 = peer_table[6];
    long long a_peer_addr7 = peer_table[7];
    __nv_bfloat16* a_peer0 = reinterpret_cast<__nv_bfloat16*>(a_peer_addr0);
    __nv_bfloat16* a_peer1 = reinterpret_cast<__nv_bfloat16*>(a_peer_addr1);
    __nv_bfloat16* a_peer2 = reinterpret_cast<__nv_bfloat16*>(a_peer_addr2);
    __nv_bfloat16* a_peer3 = reinterpret_cast<__nv_bfloat16*>(a_peer_addr3);
    __nv_bfloat16* a_peer4 = reinterpret_cast<__nv_bfloat16*>(a_peer_addr4);
    __nv_bfloat16* a_peer5 = reinterpret_cast<__nv_bfloat16*>(a_peer_addr5);
    __nv_bfloat16* a_peer6 = reinterpret_cast<__nv_bfloat16*>(a_peer_addr6);
    __nv_bfloat16* a_peer7 = reinterpret_cast<__nv_bfloat16*>(a_peer_addr7);
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
    unsigned int sentf4[4];
    sent4[0] = 2147516416;
    sentf4[0] = 2147483648;
    sent4[1] = 2147516416;
    sentf4[1] = 2147483648;
    sent4[2] = 2147516416;
    sentf4[2] = 2147483648;
    sent4[3] = 2147516416;
    sentf4[3] = 2147483648;
    {
        auto _gst_6 = a_local + a_old;
        __builtin_assume(__isGlobal(_gst_6));
        reinterpret_cast<int4*>(_gst_6)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_7 = a_local + (a_old + 4096);
        __builtin_assume(__isGlobal(_gst_7));
        reinterpret_cast<int4*>(_gst_7)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_8 = a_local + (a_old + 8192);
        __builtin_assume(__isGlobal(_gst_8));
        reinterpret_cast<int4*>(_gst_8)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_9 = a_local + (a_old + 12288);
        __builtin_assume(__isGlobal(_gst_9));
        reinterpret_cast<int4*>(_gst_9)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_10 = a_local + (a_old + 16384);
        __builtin_assume(__isGlobal(_gst_10));
        reinterpret_cast<int4*>(_gst_10)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_11 = a_local + (a_old + 20480);
        __builtin_assume(__isGlobal(_gst_11));
        reinterpret_cast<int4*>(_gst_11)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_12 = a_local + (a_old + 24576);
        __builtin_assume(__isGlobal(_gst_12));
        reinterpret_cast<int4*>(_gst_12)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_13 = a_local + (a_old + 28672);
        __builtin_assume(__isGlobal(_gst_13));
        reinterpret_cast<int4*>(_gst_13)[0] = reinterpret_cast<int4*>(sent4)[0];
    }
    {
        auto _gst_14 = b_local + b_old;
        __builtin_assume(__isGlobal(_gst_14));
        reinterpret_cast<int4*>(_gst_14)[0] = reinterpret_cast<int4*>(sentf4)[0];
    }
    {
        auto _gst_15 = b_local + (b_old + 4);
        __builtin_assume(__isGlobal(_gst_15));
        reinterpret_cast<int4*>(_gst_15)[0] = reinterpret_cast<int4*>(sentf4)[0];
    }
    {
        auto _gst_16 = b_local + (b_old + 4096);
        __builtin_assume(__isGlobal(_gst_16));
        reinterpret_cast<int4*>(_gst_16)[0] = reinterpret_cast<int4*>(sentf4)[0];
    }
    {
        auto _gst_17 = b_local + (b_old + 4100);
        __builtin_assume(__isGlobal(_gst_17));
        reinterpret_cast<int4*>(_gst_17)[0] = reinterpret_cast<int4*>(sentf4)[0];
    }
    {
        auto _gst_18 = b_local + (b_old + 8192);
        __builtin_assume(__isGlobal(_gst_18));
        reinterpret_cast<int4*>(_gst_18)[0] = reinterpret_cast<int4*>(sentf4)[0];
    }
    {
        auto _gst_19 = b_local + (b_old + 8196);
        __builtin_assume(__isGlobal(_gst_19));
        reinterpret_cast<int4*>(_gst_19)[0] = reinterpret_cast<int4*>(sentf4)[0];
    }
    {
        auto _gst_20 = b_local + (b_old + 12288);
        __builtin_assume(__isGlobal(_gst_20));
        reinterpret_cast<int4*>(_gst_20)[0] = reinterpret_cast<int4*>(sentf4)[0];
    }
    {
        auto _gst_21 = b_local + (b_old + 12292);
        __builtin_assume(__isGlobal(_gst_21));
        reinterpret_cast<int4*>(_gst_21)[0] = reinterpret_cast<int4*>(sentf4)[0];
    }
    {
        auto _gst_22 = b_local + (b_old + 16384);
        __builtin_assume(__isGlobal(_gst_22));
        reinterpret_cast<int4*>(_gst_22)[0] = reinterpret_cast<int4*>(sentf4)[0];
    }
    {
        auto _gst_23 = b_local + (b_old + 16388);
        __builtin_assume(__isGlobal(_gst_23));
        reinterpret_cast<int4*>(_gst_23)[0] = reinterpret_cast<int4*>(sentf4)[0];
    }
    {
        auto _gst_24 = b_local + (b_old + 20480);
        __builtin_assume(__isGlobal(_gst_24));
        reinterpret_cast<int4*>(_gst_24)[0] = reinterpret_cast<int4*>(sentf4)[0];
    }
    {
        auto _gst_25 = b_local + (b_old + 20484);
        __builtin_assume(__isGlobal(_gst_25));
        reinterpret_cast<int4*>(_gst_25)[0] = reinterpret_cast<int4*>(sentf4)[0];
    }
    {
        auto _gst_26 = b_local + (b_old + 24576);
        __builtin_assume(__isGlobal(_gst_26));
        reinterpret_cast<int4*>(_gst_26)[0] = reinterpret_cast<int4*>(sentf4)[0];
    }
    {
        auto _gst_27 = b_local + (b_old + 24580);
        __builtin_assume(__isGlobal(_gst_27));
        reinterpret_cast<int4*>(_gst_27)[0] = reinterpret_cast<int4*>(sentf4)[0];
    }
    {
        auto _gst_28 = b_local + (b_old + 28672);
        __builtin_assume(__isGlobal(_gst_28));
        reinterpret_cast<int4*>(_gst_28)[0] = reinterpret_cast<int4*>(sentf4)[0];
    }
    {
        auto _gst_29 = b_local + (b_old + 28676);
        __builtin_assume(__isGlobal(_gst_29));
        reinterpret_cast<int4*>(_gst_29)[0] = reinterpret_cast<int4*>(sentf4)[0];
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
            uint4 _uv4_30 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx0 * (long long)row_stride + col64) + 0);
            packets[0 + 0] = _uv4_30.x;
            packets[0 + 1] = _uv4_30.y;
            packets[0 + 2] = _uv4_30.z;
            packets[0 + 3] = _uv4_30.w;
        }
    }
    int ridx1 = idx8[1];
    if (ridx1 >= 0 && ridx1 < rows) {
        {
            uint4 _uv4_31 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx1 * (long long)row_stride + col64) + 0);
            packets[4 + 0] = _uv4_31.x;
            packets[4 + 1] = _uv4_31.y;
            packets[4 + 2] = _uv4_31.z;
            packets[4 + 3] = _uv4_31.w;
        }
    }
    int ridx2 = idx8[2];
    if (ridx2 >= 0 && ridx2 < rows) {
        {
            uint4 _uv4_32 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx2 * (long long)row_stride + col64) + 0);
            packets[8 + 0] = _uv4_32.x;
            packets[8 + 1] = _uv4_32.y;
            packets[8 + 2] = _uv4_32.z;
            packets[8 + 3] = _uv4_32.w;
        }
    }
    int ridx3 = idx8[3];
    if (ridx3 >= 0 && ridx3 < rows) {
        {
            uint4 _uv4_33 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx3 * (long long)row_stride + col64) + 0);
            packets[12 + 0] = _uv4_33.x;
            packets[12 + 1] = _uv4_33.y;
            packets[12 + 2] = _uv4_33.z;
            packets[12 + 3] = _uv4_33.w;
        }
    }
    int ridx4 = idx8[4];
    if (ridx4 >= 0 && ridx4 < rows) {
        {
            uint4 _uv4_34 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx4 * (long long)row_stride + col64) + 0);
            packets[16 + 0] = _uv4_34.x;
            packets[16 + 1] = _uv4_34.y;
            packets[16 + 2] = _uv4_34.z;
            packets[16 + 3] = _uv4_34.w;
        }
    }
    int ridx5 = idx8[5];
    if (ridx5 >= 0 && ridx5 < rows) {
        {
            uint4 _uv4_35 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx5 * (long long)row_stride + col64) + 0);
            packets[20 + 0] = _uv4_35.x;
            packets[20 + 1] = _uv4_35.y;
            packets[20 + 2] = _uv4_35.z;
            packets[20 + 3] = _uv4_35.w;
        }
    }
    int ridx6 = idx8[6];
    if (ridx6 >= 0 && ridx6 < rows) {
        {
            uint4 _uv4_36 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx6 * (long long)row_stride + col64) + 0);
            packets[24 + 0] = _uv4_36.x;
            packets[24 + 1] = _uv4_36.y;
            packets[24 + 2] = _uv4_36.z;
            packets[24 + 3] = _uv4_36.w;
        }
    }
    int ridx7 = idx8[7];
    if (ridx7 >= 0 && ridx7 < rows) {
        {
            uint4 _uv4_37 = *reinterpret_cast<const uint4*>(gemm2 + ((long long)ridx7 * (long long)row_stride + col64) + 0);
            packets[28 + 0] = _uv4_37.x;
            packets[28 + 1] = _uv4_37.y;
            packets[28 + 2] = _uv4_37.z;
            packets[28 + 3] = _uv4_37.w;
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
        uint32_t _p16s_carrier_38 = static_cast<uint32_t>(raw0[_p16s_word]);
        uint32_t _p16s_lo_38 = _p16s_carrier_38 & 0x0000ffffu;
        uint32_t _p16s_hi_38 = (_p16s_carrier_38 >> 16) & 0x0000ffffu;
        if (_p16s_lo_38 == 0x00008000u) _p16s_lo_38 = 0u;
        if (_p16s_hi_38 == 0x00008000u) _p16s_hi_38 = 0u;
        _packed16_sanitize_0[_p16s_word] = _p16s_lo_38 | (_p16s_hi_38 << 16);
    }
    {
        auto _gst_39 = a_peer0 + (a_cur + rank64 * 4096);
        __builtin_assume(__isGlobal(_gst_39));
        reinterpret_cast<int4*>(_gst_39)[0] = reinterpret_cast<int4*>(_packed16_sanitize_0)[0];
    }
    unsigned int raw1[4];
    raw1[0] = packets[4];
    raw1[1] = packets[5];
    raw1[2] = packets[6];
    raw1[3] = packets[7];
    uint32_t _packed16_sanitize_1[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_40 = static_cast<uint32_t>(raw1[_p16s_word]);
        uint32_t _p16s_lo_40 = _p16s_carrier_40 & 0x0000ffffu;
        uint32_t _p16s_hi_40 = (_p16s_carrier_40 >> 16) & 0x0000ffffu;
        if (_p16s_lo_40 == 0x00008000u) _p16s_lo_40 = 0u;
        if (_p16s_hi_40 == 0x00008000u) _p16s_hi_40 = 0u;
        _packed16_sanitize_1[_p16s_word] = _p16s_lo_40 | (_p16s_hi_40 << 16);
    }
    {
        auto _gst_41 = a_peer1 + (a_cur + rank64 * 4096);
        __builtin_assume(__isGlobal(_gst_41));
        reinterpret_cast<int4*>(_gst_41)[0] = reinterpret_cast<int4*>(_packed16_sanitize_1)[0];
    }
    unsigned int raw2[4];
    raw2[0] = packets[8];
    raw2[1] = packets[9];
    raw2[2] = packets[10];
    raw2[3] = packets[11];
    uint32_t _packed16_sanitize_2[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_42 = static_cast<uint32_t>(raw2[_p16s_word]);
        uint32_t _p16s_lo_42 = _p16s_carrier_42 & 0x0000ffffu;
        uint32_t _p16s_hi_42 = (_p16s_carrier_42 >> 16) & 0x0000ffffu;
        if (_p16s_lo_42 == 0x00008000u) _p16s_lo_42 = 0u;
        if (_p16s_hi_42 == 0x00008000u) _p16s_hi_42 = 0u;
        _packed16_sanitize_2[_p16s_word] = _p16s_lo_42 | (_p16s_hi_42 << 16);
    }
    {
        auto _gst_43 = a_peer2 + (a_cur + rank64 * 4096);
        __builtin_assume(__isGlobal(_gst_43));
        reinterpret_cast<int4*>(_gst_43)[0] = reinterpret_cast<int4*>(_packed16_sanitize_2)[0];
    }
    unsigned int raw3[4];
    raw3[0] = packets[12];
    raw3[1] = packets[13];
    raw3[2] = packets[14];
    raw3[3] = packets[15];
    uint32_t _packed16_sanitize_3[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_44 = static_cast<uint32_t>(raw3[_p16s_word]);
        uint32_t _p16s_lo_44 = _p16s_carrier_44 & 0x0000ffffu;
        uint32_t _p16s_hi_44 = (_p16s_carrier_44 >> 16) & 0x0000ffffu;
        if (_p16s_lo_44 == 0x00008000u) _p16s_lo_44 = 0u;
        if (_p16s_hi_44 == 0x00008000u) _p16s_hi_44 = 0u;
        _packed16_sanitize_3[_p16s_word] = _p16s_lo_44 | (_p16s_hi_44 << 16);
    }
    {
        auto _gst_45 = a_peer3 + (a_cur + rank64 * 4096);
        __builtin_assume(__isGlobal(_gst_45));
        reinterpret_cast<int4*>(_gst_45)[0] = reinterpret_cast<int4*>(_packed16_sanitize_3)[0];
    }
    unsigned int raw4[4];
    raw4[0] = packets[16];
    raw4[1] = packets[17];
    raw4[2] = packets[18];
    raw4[3] = packets[19];
    uint32_t _packed16_sanitize_4[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_46 = static_cast<uint32_t>(raw4[_p16s_word]);
        uint32_t _p16s_lo_46 = _p16s_carrier_46 & 0x0000ffffu;
        uint32_t _p16s_hi_46 = (_p16s_carrier_46 >> 16) & 0x0000ffffu;
        if (_p16s_lo_46 == 0x00008000u) _p16s_lo_46 = 0u;
        if (_p16s_hi_46 == 0x00008000u) _p16s_hi_46 = 0u;
        _packed16_sanitize_4[_p16s_word] = _p16s_lo_46 | (_p16s_hi_46 << 16);
    }
    {
        auto _gst_47 = a_peer4 + (a_cur + rank64 * 4096);
        __builtin_assume(__isGlobal(_gst_47));
        reinterpret_cast<int4*>(_gst_47)[0] = reinterpret_cast<int4*>(_packed16_sanitize_4)[0];
    }
    unsigned int raw5[4];
    raw5[0] = packets[20];
    raw5[1] = packets[21];
    raw5[2] = packets[22];
    raw5[3] = packets[23];
    uint32_t _packed16_sanitize_5[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_48 = static_cast<uint32_t>(raw5[_p16s_word]);
        uint32_t _p16s_lo_48 = _p16s_carrier_48 & 0x0000ffffu;
        uint32_t _p16s_hi_48 = (_p16s_carrier_48 >> 16) & 0x0000ffffu;
        if (_p16s_lo_48 == 0x00008000u) _p16s_lo_48 = 0u;
        if (_p16s_hi_48 == 0x00008000u) _p16s_hi_48 = 0u;
        _packed16_sanitize_5[_p16s_word] = _p16s_lo_48 | (_p16s_hi_48 << 16);
    }
    {
        auto _gst_49 = a_peer5 + (a_cur + rank64 * 4096);
        __builtin_assume(__isGlobal(_gst_49));
        reinterpret_cast<int4*>(_gst_49)[0] = reinterpret_cast<int4*>(_packed16_sanitize_5)[0];
    }
    unsigned int raw6[4];
    raw6[0] = packets[24];
    raw6[1] = packets[25];
    raw6[2] = packets[26];
    raw6[3] = packets[27];
    uint32_t _packed16_sanitize_6[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_50 = static_cast<uint32_t>(raw6[_p16s_word]);
        uint32_t _p16s_lo_50 = _p16s_carrier_50 & 0x0000ffffu;
        uint32_t _p16s_hi_50 = (_p16s_carrier_50 >> 16) & 0x0000ffffu;
        if (_p16s_lo_50 == 0x00008000u) _p16s_lo_50 = 0u;
        if (_p16s_hi_50 == 0x00008000u) _p16s_hi_50 = 0u;
        _packed16_sanitize_6[_p16s_word] = _p16s_lo_50 | (_p16s_hi_50 << 16);
    }
    {
        auto _gst_51 = a_peer6 + (a_cur + rank64 * 4096);
        __builtin_assume(__isGlobal(_gst_51));
        reinterpret_cast<int4*>(_gst_51)[0] = reinterpret_cast<int4*>(_packed16_sanitize_6)[0];
    }
    unsigned int raw7[4];
    raw7[0] = packets[28];
    raw7[1] = packets[29];
    raw7[2] = packets[30];
    raw7[3] = packets[31];
    uint32_t _packed16_sanitize_7[4];
    #pragma unroll
    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
        uint32_t _p16s_carrier_52 = static_cast<uint32_t>(raw7[_p16s_word]);
        uint32_t _p16s_lo_52 = _p16s_carrier_52 & 0x0000ffffu;
        uint32_t _p16s_hi_52 = (_p16s_carrier_52 >> 16) & 0x0000ffffu;
        if (_p16s_lo_52 == 0x00008000u) _p16s_lo_52 = 0u;
        if (_p16s_hi_52 == 0x00008000u) _p16s_hi_52 = 0u;
        _packed16_sanitize_7[_p16s_word] = _p16s_lo_52 | (_p16s_hi_52 << 16);
    }
    {
        auto _gst_53 = a_peer7 + (a_cur + rank64 * 4096);
        __builtin_assume(__isGlobal(_gst_53));
        reinterpret_cast<int4*>(_gst_53)[0] = reinterpret_cast<int4*>(_packed16_sanitize_7)[0];
    }
    float ys[8];
    float ssq_k[1];
    uint32_t _sysv_poll_group_0[32];
    do {
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + a_cur)) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4]), "=r"(_sysv_poll_group_0[5]), "=r"(_sysv_poll_group_0[6]), "=r"(_sysv_poll_group_0[7]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + 4096))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[8]), "=r"(_sysv_poll_group_0[9]), "=r"(_sysv_poll_group_0[10]), "=r"(_sysv_poll_group_0[11]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + 8192))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[12]), "=r"(_sysv_poll_group_0[13]), "=r"(_sysv_poll_group_0[14]), "=r"(_sysv_poll_group_0[15]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + 12288))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[16]), "=r"(_sysv_poll_group_0[17]), "=r"(_sysv_poll_group_0[18]), "=r"(_sysv_poll_group_0[19]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + 16384))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[20]), "=r"(_sysv_poll_group_0[21]), "=r"(_sysv_poll_group_0[22]), "=r"(_sysv_poll_group_0[23]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + 20480))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[24]), "=r"(_sysv_poll_group_0[25]), "=r"(_sysv_poll_group_0[26]), "=r"(_sysv_poll_group_0[27]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + 24576))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[28]), "=r"(_sysv_poll_group_0[29]), "=r"(_sysv_poll_group_0[30]), "=r"(_sysv_poll_group_0[31]) : "l"(reinterpret_cast<__nv_bfloat16*>(a_local + (a_cur + 28672))) : "memory");
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
    float _unpack_bf16_4[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_4[_pair * 2])[0]), "=f"((&_unpack_bf16_4[_pair * 2])[1])
            : "r"(_sysv_poll_group_0[8 + _pair]));
    }
    s8_0[0] = s8_0[0] + _unpack_bf16_4[0];
    s8_0[1] = s8_0[1] + _unpack_bf16_4[1];
    s8_0[2] = s8_0[2] + _unpack_bf16_4[2];
    s8_0[3] = s8_0[3] + _unpack_bf16_4[3];
    s8_0[4] = s8_0[4] + _unpack_bf16_4[4];
    s8_0[5] = s8_0[5] + _unpack_bf16_4[5];
    s8_0[6] = s8_0[6] + _unpack_bf16_4[6];
    s8_0[7] = s8_0[7] + _unpack_bf16_4[7];
    float _unpack_bf16_5[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_5[_pair * 2])[0]), "=f"((&_unpack_bf16_5[_pair * 2])[1])
            : "r"(_sysv_poll_group_0[12 + _pair]));
    }
    s8_0[0] = s8_0[0] + _unpack_bf16_5[0];
    s8_0[1] = s8_0[1] + _unpack_bf16_5[1];
    s8_0[2] = s8_0[2] + _unpack_bf16_5[2];
    s8_0[3] = s8_0[3] + _unpack_bf16_5[3];
    s8_0[4] = s8_0[4] + _unpack_bf16_5[4];
    s8_0[5] = s8_0[5] + _unpack_bf16_5[5];
    s8_0[6] = s8_0[6] + _unpack_bf16_5[6];
    s8_0[7] = s8_0[7] + _unpack_bf16_5[7];
    float _unpack_bf16_6[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_6[_pair * 2])[0]), "=f"((&_unpack_bf16_6[_pair * 2])[1])
            : "r"(_sysv_poll_group_0[16 + _pair]));
    }
    s8_0[0] = s8_0[0] + _unpack_bf16_6[0];
    s8_0[1] = s8_0[1] + _unpack_bf16_6[1];
    s8_0[2] = s8_0[2] + _unpack_bf16_6[2];
    s8_0[3] = s8_0[3] + _unpack_bf16_6[3];
    s8_0[4] = s8_0[4] + _unpack_bf16_6[4];
    s8_0[5] = s8_0[5] + _unpack_bf16_6[5];
    s8_0[6] = s8_0[6] + _unpack_bf16_6[6];
    s8_0[7] = s8_0[7] + _unpack_bf16_6[7];
    float _unpack_bf16_7[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_7[_pair * 2])[0]), "=f"((&_unpack_bf16_7[_pair * 2])[1])
            : "r"(_sysv_poll_group_0[20 + _pair]));
    }
    s8_0[0] = s8_0[0] + _unpack_bf16_7[0];
    s8_0[1] = s8_0[1] + _unpack_bf16_7[1];
    s8_0[2] = s8_0[2] + _unpack_bf16_7[2];
    s8_0[3] = s8_0[3] + _unpack_bf16_7[3];
    s8_0[4] = s8_0[4] + _unpack_bf16_7[4];
    s8_0[5] = s8_0[5] + _unpack_bf16_7[5];
    s8_0[6] = s8_0[6] + _unpack_bf16_7[6];
    s8_0[7] = s8_0[7] + _unpack_bf16_7[7];
    float _unpack_bf16_8[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_8[_pair * 2])[0]), "=f"((&_unpack_bf16_8[_pair * 2])[1])
            : "r"(_sysv_poll_group_0[24 + _pair]));
    }
    s8_0[0] = s8_0[0] + _unpack_bf16_8[0];
    s8_0[1] = s8_0[1] + _unpack_bf16_8[1];
    s8_0[2] = s8_0[2] + _unpack_bf16_8[2];
    s8_0[3] = s8_0[3] + _unpack_bf16_8[3];
    s8_0[4] = s8_0[4] + _unpack_bf16_8[4];
    s8_0[5] = s8_0[5] + _unpack_bf16_8[5];
    s8_0[6] = s8_0[6] + _unpack_bf16_8[6];
    s8_0[7] = s8_0[7] + _unpack_bf16_8[7];
    float _unpack_bf16_9[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_unpack_bf16_9[_pair * 2])[0]), "=f"((&_unpack_bf16_9[_pair * 2])[1])
            : "r"(_sysv_poll_group_0[28 + _pair]));
    }
    s8_0[0] = s8_0[0] + _unpack_bf16_9[0];
    s8_0[1] = s8_0[1] + _unpack_bf16_9[1];
    s8_0[2] = s8_0[2] + _unpack_bf16_9[2];
    s8_0[3] = s8_0[3] + _unpack_bf16_9[3];
    s8_0[4] = s8_0[4] + _unpack_bf16_9[4];
    s8_0[5] = s8_0[5] + _unpack_bf16_9[5];
    s8_0[6] = s8_0[6] + _unpack_bf16_9[6];
    s8_0[7] = s8_0[7] + _unpack_bf16_9[7];
    unsigned int s8_0_bf16[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s8_0[_lp*2 + 0], s8_0[_lp*2+1 + 0]));
        s8_0_bf16[_lp] = *(uint32_t*)&_bf2;
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
            : "r"(s8_0_bf16[_pair]));
    }
    float ssq0 = 0.0f;
    float y0_0 = ((mode != 0) ? _unpack_bf16_10[0] * gw8[0] : _unpack_bf16_10[0]);
    ys[0] = y0_0;
    ssq0 = __fmaf_rn(y0_0, y0_0, ssq0);
    float y0_1 = ((mode != 0) ? _unpack_bf16_10[1] * gw8[1] : _unpack_bf16_10[1]);
    ys[1] = y0_1;
    ssq0 = __fmaf_rn(y0_1, y0_1, ssq0);
    float y0_2 = ((mode != 0) ? _unpack_bf16_10[2] * gw8[2] : _unpack_bf16_10[2]);
    ys[2] = y0_2;
    ssq0 = __fmaf_rn(y0_2, y0_2, ssq0);
    float y0_3 = ((mode != 0) ? _unpack_bf16_10[3] * gw8[3] : _unpack_bf16_10[3]);
    ys[3] = y0_3;
    ssq0 = __fmaf_rn(y0_3, y0_3, ssq0);
    float y0_4 = ((mode != 0) ? _unpack_bf16_10[4] * gw8[4] : _unpack_bf16_10[4]);
    ys[4] = y0_4;
    ssq0 = __fmaf_rn(y0_4, y0_4, ssq0);
    float y0_5 = ((mode != 0) ? _unpack_bf16_10[5] * gw8[5] : _unpack_bf16_10[5]);
    ys[5] = y0_5;
    ssq0 = __fmaf_rn(y0_5, y0_5, ssq0);
    float y0_6 = ((mode != 0) ? _unpack_bf16_10[6] * gw8[6] : _unpack_bf16_10[6]);
    ys[6] = y0_6;
    ssq0 = __fmaf_rn(y0_6, y0_6, ssq0);
    float y0_7 = ((mode != 0) ? _unpack_bf16_10[7] * gw8[7] : _unpack_bf16_10[7]);
    ys[7] = y0_7;
    ssq0 = __fmaf_rn(y0_7, y0_7, ssq0);
    ssq_k[0] = ssq0;
    float _warp_reduce_0 = ssq_k[0];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    if (lane == 0) {
        warp_tab[warp] = _warp_reduce_0;
    }
    __syncthreads();
    float tot_k[1];
    int kk = tid & 0;
    float part = warp_tab[kk];
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (tid < 1) {
        unsigned int slot_addr = cta_tab_addr + (unsigned int)((cta_rank_0 + tid) * 4);
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
        uint32_t _cluster_mapa_58;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_58) : "r"(slot_addr), "r"(4));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_58), "f"(part) : "memory");
        uint32_t _cluster_mapa_59;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_59) : "r"(slot_addr), "r"(5));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_59), "f"(part) : "memory");
        uint32_t _cluster_mapa_60;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_60) : "r"(slot_addr), "r"(6));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_60), "f"(part) : "memory");
        uint32_t _cluster_mapa_61;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_61) : "r"(slot_addr), "r"(7));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_61), "f"(part) : "memory");
        uint32_t _cluster_mapa_62;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_62) : "r"(slot_addr), "r"(8));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_62), "f"(part) : "memory");
        uint32_t _cluster_mapa_63;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_63) : "r"(slot_addr), "r"(9));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_63), "f"(part) : "memory");
        uint32_t _cluster_mapa_64;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_64) : "r"(slot_addr), "r"(10));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_64), "f"(part) : "memory");
        uint32_t _cluster_mapa_65;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_65) : "r"(slot_addr), "r"(11));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_65), "f"(part) : "memory");
        uint32_t _cluster_mapa_66;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_66) : "r"(slot_addr), "r"(12));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_66), "f"(part) : "memory");
        uint32_t _cluster_mapa_67;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_67) : "r"(slot_addr), "r"(13));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_67), "f"(part) : "memory");
        uint32_t _cluster_mapa_68;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_68) : "r"(slot_addr), "r"(14));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_68), "f"(part) : "memory");
        uint32_t _cluster_mapa_69;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_69) : "r"(slot_addr), "r"(15));
        asm volatile(
            "st.shared::cluster.f32 [%0], %1;"
            :: "r"(_cluster_mapa_69), "f"(part) : "memory");
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    tot_k[0] = cta_tab[0];
    tot_k[0] = tot_k[0] + cta_tab[1];
    tot_k[0] = tot_k[0] + cta_tab[2];
    tot_k[0] = tot_k[0] + cta_tab[3];
    tot_k[0] = tot_k[0] + cta_tab[4];
    tot_k[0] = tot_k[0] + cta_tab[5];
    tot_k[0] = tot_k[0] + cta_tab[6];
    tot_k[0] = tot_k[0] + cta_tab[7];
    tot_k[0] = tot_k[0] + cta_tab[8];
    tot_k[0] = tot_k[0] + cta_tab[9];
    tot_k[0] = tot_k[0] + cta_tab[10];
    tot_k[0] = tot_k[0] + cta_tab[11];
    tot_k[0] = tot_k[0] + cta_tab[12];
    tot_k[0] = tot_k[0] + cta_tab[13];
    tot_k[0] = tot_k[0] + cta_tab[14];
    tot_k[0] = tot_k[0] + cta_tab[15];
    float acc8[8];
    acc8[0] = 0.0f;
    acc8[1] = 0.0f;
    acc8[2] = 0.0f;
    acc8[3] = 0.0f;
    acc8[4] = 0.0f;
    acc8[5] = 0.0f;
    acc8[6] = 0.0f;
    acc8[7] = 0.0f;
    float wk0 = ((rank_i == 7) ? w8[7] : ((rank_i == 6) ? w8[6] : ((rank_i == 5) ? w8[5] : ((rank_i == 4) ? w8[4] : ((rank_i == 3) ? w8[3] : ((rank_i == 2) ? w8[2] : ((rank_i == 1) ? w8[1] : w8[0])))))));
    float _sqrt_70;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_70) : "f"(__fmaf_rn(tot_k[0], 0.00024414062f, eps)));
    float rs0 = __frcp_rn(_sqrt_70);
    float o0_0 = ys[0] * rs0;
    float o2_0_0 = ((mode != 0) ? o0_0 : o0_0 * gw8[0]);
    acc8[0] = __fmaf_rn(o2_0_0, wk0, acc8[0]);
    float o0_1 = ys[1] * rs0;
    float o2_0_1 = ((mode != 0) ? o0_1 : o0_1 * gw8[1]);
    acc8[1] = __fmaf_rn(o2_0_1, wk0, acc8[1]);
    float o0_2 = ys[2] * rs0;
    float o2_0_2 = ((mode != 0) ? o0_2 : o0_2 * gw8[2]);
    acc8[2] = __fmaf_rn(o2_0_2, wk0, acc8[2]);
    float o0_3 = ys[3] * rs0;
    float o2_0_3 = ((mode != 0) ? o0_3 : o0_3 * gw8[3]);
    acc8[3] = __fmaf_rn(o2_0_3, wk0, acc8[3]);
    float o0_4 = ys[4] * rs0;
    float o2_0_4 = ((mode != 0) ? o0_4 : o0_4 * gw8[4]);
    acc8[4] = __fmaf_rn(o2_0_4, wk0, acc8[4]);
    float o0_5 = ys[5] * rs0;
    float o2_0_5 = ((mode != 0) ? o0_5 : o0_5 * gw8[5]);
    acc8[5] = __fmaf_rn(o2_0_5, wk0, acc8[5]);
    float o0_6 = ys[6] * rs0;
    float o2_0_6 = ((mode != 0) ? o0_6 : o0_6 * gw8[6]);
    acc8[6] = __fmaf_rn(o2_0_6, wk0, acc8[6]);
    float o0_7 = ys[7] * rs0;
    float o2_0_7 = ((mode != 0) ? o0_7 : o0_7 * gw8[7]);
    acc8[7] = __fmaf_rn(o2_0_7, wk0, acc8[7]);
    unsigned int lo4[4];
    unsigned int hi4[4];
    unsigned int blo0 = __as_u32(acc8[0]);
    unsigned int bhi0 = __as_u32(acc8[4]);
    unsigned int slo0 = ((blo0 == 2147483648u) ? 0 : blo0);
    unsigned int shi0 = ((bhi0 == 2147483648u) ? 0 : bhi0);
    lo4[0] = slo0;
    hi4[0] = shi0;
    unsigned int blo1 = __as_u32(acc8[1]);
    unsigned int bhi1 = __as_u32(acc8[5]);
    unsigned int slo1 = ((blo1 == 2147483648u) ? 0 : blo1);
    unsigned int shi1 = ((bhi1 == 2147483648u) ? 0 : bhi1);
    lo4[1] = slo1;
    hi4[1] = shi1;
    unsigned int blo2 = __as_u32(acc8[2]);
    unsigned int bhi2 = __as_u32(acc8[6]);
    unsigned int slo2 = ((blo2 == 2147483648u) ? 0 : blo2);
    unsigned int shi2 = ((bhi2 == 2147483648u) ? 0 : bhi2);
    lo4[2] = slo2;
    hi4[2] = shi2;
    unsigned int blo3 = __as_u32(acc8[3]);
    unsigned int bhi3 = __as_u32(acc8[7]);
    unsigned int slo3 = ((blo3 == 2147483648u) ? 0 : blo3);
    unsigned int shi3 = ((bhi3 == 2147483648u) ? 0 : bhi3);
    lo4[3] = slo3;
    hi4[3] = shi3;
    unsigned long long mc_lo = b_mcast + (unsigned long long)((b_cur + rank64 * 4096) * 4);
    int lane_odd = lane & 1;
    long long mc_i = (long long)mc_lo - (long long)lane * 16;
    int src_lane0 = lane >> 1;
    unsigned int q0[4];
    unsigned int q_lo0_0 = __shfl_sync(0xFFFFFFFF, lo4[0], src_lane0);
    unsigned int q_hi0_0 = __shfl_sync(0xFFFFFFFF, hi4[0], src_lane0);
    q0[0] = ((lane_odd == 0) ? q_lo0_0 : q_hi0_0);
    unsigned int q_lo0_1 = __shfl_sync(0xFFFFFFFF, lo4[1], src_lane0);
    unsigned int q_hi0_1 = __shfl_sync(0xFFFFFFFF, hi4[1], src_lane0);
    q0[1] = ((lane_odd == 0) ? q_lo0_1 : q_hi0_1);
    unsigned int q_lo0_2 = __shfl_sync(0xFFFFFFFF, lo4[2], src_lane0);
    unsigned int q_hi0_2 = __shfl_sync(0xFFFFFFFF, hi4[2], src_lane0);
    q0[2] = ((lane_odd == 0) ? q_lo0_2 : q_hi0_2);
    unsigned int q_lo0_3 = __shfl_sync(0xFFFFFFFF, lo4[3], src_lane0);
    unsigned int q_hi0_3 = __shfl_sync(0xFFFFFFFF, hi4[3], src_lane0);
    q0[3] = ((lane_odd == 0) ? q_lo0_3 : q_hi0_3);
    unsigned long long mc_sec0 = (unsigned long long)mc_i;
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<float*>(mc_sec0)), "r"(q0[0]), "r"(q0[1]), "r"(q0[2]), "r"(q0[3]) : "memory");
    int src_lane1 = (lane >> 1) + 16;
    unsigned int q1[4];
    unsigned int q_lo1_0 = __shfl_sync(0xFFFFFFFF, lo4[0], src_lane1);
    unsigned int q_hi1_0 = __shfl_sync(0xFFFFFFFF, hi4[0], src_lane1);
    q1[0] = ((lane_odd == 0) ? q_lo1_0 : q_hi1_0);
    unsigned int q_lo1_1 = __shfl_sync(0xFFFFFFFF, lo4[1], src_lane1);
    unsigned int q_hi1_1 = __shfl_sync(0xFFFFFFFF, hi4[1], src_lane1);
    q1[1] = ((lane_odd == 0) ? q_lo1_1 : q_hi1_1);
    unsigned int q_lo1_2 = __shfl_sync(0xFFFFFFFF, lo4[2], src_lane1);
    unsigned int q_hi1_2 = __shfl_sync(0xFFFFFFFF, hi4[2], src_lane1);
    q1[2] = ((lane_odd == 0) ? q_lo1_2 : q_hi1_2);
    unsigned int q_lo1_3 = __shfl_sync(0xFFFFFFFF, lo4[3], src_lane1);
    unsigned int q_hi1_3 = __shfl_sync(0xFFFFFFFF, hi4[3], src_lane1);
    q1[3] = ((lane_odd == 0) ? q_lo1_3 : q_hi1_3);
    unsigned long long mc_sec1 = (unsigned long long)(mc_i + 512);
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<float*>(mc_sec1)), "r"(q1[0]), "r"(q1[1]), "r"(q1[2]), "r"(q1[3]) : "memory");
    asm volatile("fence.proxy.alias;" ::: "memory");
    uint32_t _sysv_poll_group_1[64];
    do {
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[0]), "=r"(_sysv_poll_group_1[1]), "=r"(_sysv_poll_group_1[2]), "=r"(_sysv_poll_group_1[3]) : "l"(reinterpret_cast<unsigned int*>(b_local + b_cur)) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[4]), "=r"(_sysv_poll_group_1[5]), "=r"(_sysv_poll_group_1[6]), "=r"(_sysv_poll_group_1[7]) : "l"(reinterpret_cast<unsigned int*>(b_local + (b_cur + 4))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[8]), "=r"(_sysv_poll_group_1[9]), "=r"(_sysv_poll_group_1[10]), "=r"(_sysv_poll_group_1[11]) : "l"(reinterpret_cast<unsigned int*>(b_local + (b_cur + 4096))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[12]), "=r"(_sysv_poll_group_1[13]), "=r"(_sysv_poll_group_1[14]), "=r"(_sysv_poll_group_1[15]) : "l"(reinterpret_cast<unsigned int*>(b_local + (b_cur + 4100))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[16]), "=r"(_sysv_poll_group_1[17]), "=r"(_sysv_poll_group_1[18]), "=r"(_sysv_poll_group_1[19]) : "l"(reinterpret_cast<unsigned int*>(b_local + (b_cur + 8192))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[20]), "=r"(_sysv_poll_group_1[21]), "=r"(_sysv_poll_group_1[22]), "=r"(_sysv_poll_group_1[23]) : "l"(reinterpret_cast<unsigned int*>(b_local + (b_cur + 8196))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[24]), "=r"(_sysv_poll_group_1[25]), "=r"(_sysv_poll_group_1[26]), "=r"(_sysv_poll_group_1[27]) : "l"(reinterpret_cast<unsigned int*>(b_local + (b_cur + 12288))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[28]), "=r"(_sysv_poll_group_1[29]), "=r"(_sysv_poll_group_1[30]), "=r"(_sysv_poll_group_1[31]) : "l"(reinterpret_cast<unsigned int*>(b_local + (b_cur + 12292))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[32]), "=r"(_sysv_poll_group_1[33]), "=r"(_sysv_poll_group_1[34]), "=r"(_sysv_poll_group_1[35]) : "l"(reinterpret_cast<unsigned int*>(b_local + (b_cur + 16384))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[36]), "=r"(_sysv_poll_group_1[37]), "=r"(_sysv_poll_group_1[38]), "=r"(_sysv_poll_group_1[39]) : "l"(reinterpret_cast<unsigned int*>(b_local + (b_cur + 16388))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[40]), "=r"(_sysv_poll_group_1[41]), "=r"(_sysv_poll_group_1[42]), "=r"(_sysv_poll_group_1[43]) : "l"(reinterpret_cast<unsigned int*>(b_local + (b_cur + 20480))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[44]), "=r"(_sysv_poll_group_1[45]), "=r"(_sysv_poll_group_1[46]), "=r"(_sysv_poll_group_1[47]) : "l"(reinterpret_cast<unsigned int*>(b_local + (b_cur + 20484))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[48]), "=r"(_sysv_poll_group_1[49]), "=r"(_sysv_poll_group_1[50]), "=r"(_sysv_poll_group_1[51]) : "l"(reinterpret_cast<unsigned int*>(b_local + (b_cur + 24576))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[52]), "=r"(_sysv_poll_group_1[53]), "=r"(_sysv_poll_group_1[54]), "=r"(_sysv_poll_group_1[55]) : "l"(reinterpret_cast<unsigned int*>(b_local + (b_cur + 24580))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[56]), "=r"(_sysv_poll_group_1[57]), "=r"(_sysv_poll_group_1[58]), "=r"(_sysv_poll_group_1[59]) : "l"(reinterpret_cast<unsigned int*>(b_local + (b_cur + 28672))) : "memory");
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_1[60]), "=r"(_sysv_poll_group_1[61]), "=r"(_sysv_poll_group_1[62]), "=r"(_sysv_poll_group_1[63]) : "l"(reinterpret_cast<unsigned int*>(b_local + (b_cur + 28676))) : "memory");
    } while (
        (_sysv_poll_group_1[0] == 0x80000000u) || (_sysv_poll_group_1[1] == 0x80000000u) || (_sysv_poll_group_1[2] == 0x80000000u) || (_sysv_poll_group_1[3] == 0x80000000u) ||
        (_sysv_poll_group_1[4] == 0x80000000u) || (_sysv_poll_group_1[5] == 0x80000000u) || (_sysv_poll_group_1[6] == 0x80000000u) || (_sysv_poll_group_1[7] == 0x80000000u) ||
        (_sysv_poll_group_1[8] == 0x80000000u) || (_sysv_poll_group_1[9] == 0x80000000u) || (_sysv_poll_group_1[10] == 0x80000000u) || (_sysv_poll_group_1[11] == 0x80000000u) ||
        (_sysv_poll_group_1[12] == 0x80000000u) || (_sysv_poll_group_1[13] == 0x80000000u) || (_sysv_poll_group_1[14] == 0x80000000u) || (_sysv_poll_group_1[15] == 0x80000000u) ||
        (_sysv_poll_group_1[16] == 0x80000000u) || (_sysv_poll_group_1[17] == 0x80000000u) || (_sysv_poll_group_1[18] == 0x80000000u) || (_sysv_poll_group_1[19] == 0x80000000u) ||
        (_sysv_poll_group_1[20] == 0x80000000u) || (_sysv_poll_group_1[21] == 0x80000000u) || (_sysv_poll_group_1[22] == 0x80000000u) || (_sysv_poll_group_1[23] == 0x80000000u) ||
        (_sysv_poll_group_1[24] == 0x80000000u) || (_sysv_poll_group_1[25] == 0x80000000u) || (_sysv_poll_group_1[26] == 0x80000000u) || (_sysv_poll_group_1[27] == 0x80000000u) ||
        (_sysv_poll_group_1[28] == 0x80000000u) || (_sysv_poll_group_1[29] == 0x80000000u) || (_sysv_poll_group_1[30] == 0x80000000u) || (_sysv_poll_group_1[31] == 0x80000000u) ||
        (_sysv_poll_group_1[32] == 0x80000000u) || (_sysv_poll_group_1[33] == 0x80000000u) || (_sysv_poll_group_1[34] == 0x80000000u) || (_sysv_poll_group_1[35] == 0x80000000u) ||
        (_sysv_poll_group_1[36] == 0x80000000u) || (_sysv_poll_group_1[37] == 0x80000000u) || (_sysv_poll_group_1[38] == 0x80000000u) || (_sysv_poll_group_1[39] == 0x80000000u) ||
        (_sysv_poll_group_1[40] == 0x80000000u) || (_sysv_poll_group_1[41] == 0x80000000u) || (_sysv_poll_group_1[42] == 0x80000000u) || (_sysv_poll_group_1[43] == 0x80000000u) ||
        (_sysv_poll_group_1[44] == 0x80000000u) || (_sysv_poll_group_1[45] == 0x80000000u) || (_sysv_poll_group_1[46] == 0x80000000u) || (_sysv_poll_group_1[47] == 0x80000000u) ||
        (_sysv_poll_group_1[48] == 0x80000000u) || (_sysv_poll_group_1[49] == 0x80000000u) || (_sysv_poll_group_1[50] == 0x80000000u) || (_sysv_poll_group_1[51] == 0x80000000u) ||
        (_sysv_poll_group_1[52] == 0x80000000u) || (_sysv_poll_group_1[53] == 0x80000000u) || (_sysv_poll_group_1[54] == 0x80000000u) || (_sysv_poll_group_1[55] == 0x80000000u) ||
        (_sysv_poll_group_1[56] == 0x80000000u) || (_sysv_poll_group_1[57] == 0x80000000u) || (_sysv_poll_group_1[58] == 0x80000000u) || (_sysv_poll_group_1[59] == 0x80000000u) ||
        (_sysv_poll_group_1[60] == 0x80000000u) || (_sysv_poll_group_1[61] == 0x80000000u) || (_sysv_poll_group_1[62] == 0x80000000u) || (_sysv_poll_group_1[63] == 0x80000000u)
    );
    float out8[8];
    out8[0] = __uint_as_float(_sysv_poll_group_1[0]);
    out8[0] = out8[0] + __uint_as_float(_sysv_poll_group_1[8]);
    out8[0] = out8[0] + __uint_as_float(_sysv_poll_group_1[16]);
    out8[0] = out8[0] + __uint_as_float(_sysv_poll_group_1[24]);
    out8[0] = out8[0] + __uint_as_float(_sysv_poll_group_1[32]);
    out8[0] = out8[0] + __uint_as_float(_sysv_poll_group_1[40]);
    out8[0] = out8[0] + __uint_as_float(_sysv_poll_group_1[48]);
    out8[0] = out8[0] + __uint_as_float(_sysv_poll_group_1[56]);
    out8[1] = __uint_as_float(_sysv_poll_group_1[1]);
    out8[1] = out8[1] + __uint_as_float(_sysv_poll_group_1[9]);
    out8[1] = out8[1] + __uint_as_float(_sysv_poll_group_1[17]);
    out8[1] = out8[1] + __uint_as_float(_sysv_poll_group_1[25]);
    out8[1] = out8[1] + __uint_as_float(_sysv_poll_group_1[33]);
    out8[1] = out8[1] + __uint_as_float(_sysv_poll_group_1[41]);
    out8[1] = out8[1] + __uint_as_float(_sysv_poll_group_1[49]);
    out8[1] = out8[1] + __uint_as_float(_sysv_poll_group_1[57]);
    out8[2] = __uint_as_float(_sysv_poll_group_1[2]);
    out8[2] = out8[2] + __uint_as_float(_sysv_poll_group_1[10]);
    out8[2] = out8[2] + __uint_as_float(_sysv_poll_group_1[18]);
    out8[2] = out8[2] + __uint_as_float(_sysv_poll_group_1[26]);
    out8[2] = out8[2] + __uint_as_float(_sysv_poll_group_1[34]);
    out8[2] = out8[2] + __uint_as_float(_sysv_poll_group_1[42]);
    out8[2] = out8[2] + __uint_as_float(_sysv_poll_group_1[50]);
    out8[2] = out8[2] + __uint_as_float(_sysv_poll_group_1[58]);
    out8[3] = __uint_as_float(_sysv_poll_group_1[3]);
    out8[3] = out8[3] + __uint_as_float(_sysv_poll_group_1[11]);
    out8[3] = out8[3] + __uint_as_float(_sysv_poll_group_1[19]);
    out8[3] = out8[3] + __uint_as_float(_sysv_poll_group_1[27]);
    out8[3] = out8[3] + __uint_as_float(_sysv_poll_group_1[35]);
    out8[3] = out8[3] + __uint_as_float(_sysv_poll_group_1[43]);
    out8[3] = out8[3] + __uint_as_float(_sysv_poll_group_1[51]);
    out8[3] = out8[3] + __uint_as_float(_sysv_poll_group_1[59]);
    out8[4] = __uint_as_float(_sysv_poll_group_1[4]);
    out8[4] = out8[4] + __uint_as_float(_sysv_poll_group_1[12]);
    out8[4] = out8[4] + __uint_as_float(_sysv_poll_group_1[20]);
    out8[4] = out8[4] + __uint_as_float(_sysv_poll_group_1[28]);
    out8[4] = out8[4] + __uint_as_float(_sysv_poll_group_1[36]);
    out8[4] = out8[4] + __uint_as_float(_sysv_poll_group_1[44]);
    out8[4] = out8[4] + __uint_as_float(_sysv_poll_group_1[52]);
    out8[4] = out8[4] + __uint_as_float(_sysv_poll_group_1[60]);
    out8[5] = __uint_as_float(_sysv_poll_group_1[5]);
    out8[5] = out8[5] + __uint_as_float(_sysv_poll_group_1[13]);
    out8[5] = out8[5] + __uint_as_float(_sysv_poll_group_1[21]);
    out8[5] = out8[5] + __uint_as_float(_sysv_poll_group_1[29]);
    out8[5] = out8[5] + __uint_as_float(_sysv_poll_group_1[37]);
    out8[5] = out8[5] + __uint_as_float(_sysv_poll_group_1[45]);
    out8[5] = out8[5] + __uint_as_float(_sysv_poll_group_1[53]);
    out8[5] = out8[5] + __uint_as_float(_sysv_poll_group_1[61]);
    out8[6] = __uint_as_float(_sysv_poll_group_1[6]);
    out8[6] = out8[6] + __uint_as_float(_sysv_poll_group_1[14]);
    out8[6] = out8[6] + __uint_as_float(_sysv_poll_group_1[22]);
    out8[6] = out8[6] + __uint_as_float(_sysv_poll_group_1[30]);
    out8[6] = out8[6] + __uint_as_float(_sysv_poll_group_1[38]);
    out8[6] = out8[6] + __uint_as_float(_sysv_poll_group_1[46]);
    out8[6] = out8[6] + __uint_as_float(_sysv_poll_group_1[54]);
    out8[6] = out8[6] + __uint_as_float(_sysv_poll_group_1[62]);
    out8[7] = __uint_as_float(_sysv_poll_group_1[7]);
    out8[7] = out8[7] + __uint_as_float(_sysv_poll_group_1[15]);
    out8[7] = out8[7] + __uint_as_float(_sysv_poll_group_1[23]);
    out8[7] = out8[7] + __uint_as_float(_sysv_poll_group_1[31]);
    out8[7] = out8[7] + __uint_as_float(_sysv_poll_group_1[39]);
    out8[7] = out8[7] + __uint_as_float(_sysv_poll_group_1[47]);
    out8[7] = out8[7] + __uint_as_float(_sysv_poll_group_1[55]);
    out8[7] = out8[7] + __uint_as_float(_sysv_poll_group_1[63]);
    unsigned int _packed_0[4];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out8[_lp*2 + 0], out8[_lp*2+1 + 0]));
        _packed_0[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out8[_lp*2 + 2], out8[_lp*2+1 + 2]));
        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out8[_lp*2 + 4], out8[_lp*2+1 + 4]));
        _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out8[_lp*2 + 6], out8[_lp*2+1 + 6]));
        _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(output + (slot64 * 4096 + col64))[0] = reinterpret_cast<int4*>(_packed_0)[0];
    int g_next = ((g == 2) ? 0 : g + 1);
    if (tid == 0 && cta_rank_0 == 0) {
        *(reinterpret_cast<int*>(gen + slot) + (0)) = g_next;
    }
}

} // extern "C"
