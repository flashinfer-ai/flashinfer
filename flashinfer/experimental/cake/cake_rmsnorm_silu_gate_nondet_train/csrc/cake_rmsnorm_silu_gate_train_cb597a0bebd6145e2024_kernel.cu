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
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

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

#define LAUNCH_MIN_BLOCKS 4

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


extern "C" __device__ float __nv_expf(float);


__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
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

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(16,1,1) void
kernel_cake_rmsnorm_silu_gate_train_cb597a0bebd6145e2024(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dz, __nv_bfloat16* __restrict__ dw, int rows, CakeFastDivmod heads, long long dy_tok, long long dy_head, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps, int accumulate, int weight_grad)
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
    #define recv_full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 16;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 16;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    float* fold32 = reinterpret_cast<float*>(smem_raw + 1024);
    const int fold32_addr = smem + 1024;
    float* part = reinterpret_cast<float*>(smem_raw + 5120);
    const int part_addr = smem + 5120;
    float* recv = reinterpret_cast<float*>(smem_raw + 5632);
    const int recv_addr = smem + 5632;

    // Mbarrier init (1 barriers at smem_raw[0..8))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // recv_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // === Task calls (declaration order) ===
    int rank = cta_rank;
    int group = tid >> 4;
    int col = (tid & 15) * 8;
    int nblocks = (rows + 15) / 16;
    unsigned int wbits[4];
    {
        uint4 _uv4_0 = *reinterpret_cast<const uint4*>(w + col + 0);
        wbits[0 + 0] = _uv4_0.x;
        wbits[0 + 1] = _uv4_0.y;
        wbits[0 + 2] = _uv4_0.z;
        wbits[0 + 3] = _uv4_0.w;
    }
    float acc[8];
    acc[0] = 0.0f;
    acc[1] = 0.0f;
    acc[2] = 0.0f;
    acc[3] = 0.0f;
    acc[4] = 0.0f;
    acc[5] = 0.0f;
    acc[6] = 0.0f;
    acc[7] = 0.0f;
    #pragma unroll 1
    for (int u = rank; u < nblocks; u += 16) {
        int row_0 = u * 16 + group;
        int src_0 = ((row_0) < (rows - 1) ? (row_0) : (rows - 1));
        uint32_t tq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_0);
        uint32_t th_0 = (uint32_t)((unsigned int)src_0) - tq_0 * (uint32_t)(heads.divisor);
        long long goff_0 = (long long)tq_0 * dy_tok + (long long)th_0 * dy_head + (long long)col;
        long long xoff_0 = (long long)tq_0 * x_tok + (long long)th_0 * x_head + (long long)col;
        long long zoff_0 = (long long)tq_0 * z_tok + (long long)th_0 * z_head + (long long)col;
        float gs_0[8];
        float xs_0[8];
        float zs_0[8];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(dy + goff_0 + 0);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&gs_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&gs_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(x + xoff_0 + 0);
            uint4 _vld_2[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_2[_blk] = _vptr_2[_blk];
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&xs_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                    (&xs_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                }
            }
        }
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(z + zoff_0 + 0);
            uint4 _vld_3[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_3[_blk] = _vptr_3[_blk];
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&zs_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                    (&zs_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                }
            }
        }
        float wv[8];
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&wv[_pair * 2])[0]), "=f"((&wv[_pair * 2])[1])
                : "r"(wbits[_pair]));
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&wv[2 + _pair * 2])[0]), "=f"((&wv[2 + _pair * 2])[1])
                : "r"(wbits[1 + _pair]));
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&wv[4 + _pair * 2])[0]), "=f"((&wv[4 + _pair * 2])[1])
                : "r"(wbits[2 + _pair]));
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&wv[6 + _pair * 2])[0]), "=f"((&wv[6 + _pair * 2])[1])
                : "r"(wbits[3 + _pair]));
        }
        float total_0 = 0.0f;
        total_0 = __fmaf_rn(xs_0[0], xs_0[0], total_0);
        total_0 = __fmaf_rn(xs_0[1], xs_0[1], total_0);
        total_0 = __fmaf_rn(xs_0[2], xs_0[2], total_0);
        total_0 = __fmaf_rn(xs_0[3], xs_0[3], total_0);
        total_0 = __fmaf_rn(xs_0[4], xs_0[4], total_0);
        total_0 = __fmaf_rn(xs_0[5], xs_0[5], total_0);
        total_0 = __fmaf_rn(xs_0[6], xs_0[6], total_0);
        total_0 = __fmaf_rn(xs_0[7], xs_0[7], total_0);
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, total_0, 1);
        total_0 = total_0 + _shfl_xor_sync_4;
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, total_0, 2);
        total_0 = total_0 + _shfl_xor_sync_5;
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, total_0, 4);
        total_0 = total_0 + _shfl_xor_sync_6;
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, total_0, 8);
        total_0 = total_0 + _shfl_xor_sync_7;
        float _sqrt_8;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_8) : "f"(total_0 * 0.0078125f + eps));
        float rstd_0 = __frcp_rn(_sqrt_8);
        unsigned int keep_0 = 4294967295u;
        unsigned int sign_0 = 2147483648u;
        keep_0 = ((row_0 < rows) ? keep_0 : sign_0);
        float dot_0 = 0.0f;
        int fast_0 = 0;
        float _neg_9;
        asm("neg.f32 %0, %1;" : "=f"(_neg_9) : "f"(zs_0[0]));
        float _neg_10;
        asm("neg.f32 %0, %1;" : "=f"(_neg_10) : "f"(zs_0[1]));
        float _neg_11;
        asm("neg.f32 %0, %1;" : "=f"(_neg_11) : "f"(zs_0[2]));
        float mz_0;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz_0) : "f"(_neg_9), "f"(_neg_10), "f"(_neg_11));
        float _abs_13;
        asm("abs.f32 %0, %1;" : "=f"(_abs_13) : "f"(zs_0[0]));
        float _neg_12;
        asm("neg.f32 %0, %1;" : "=f"(_neg_12) : "f"(_abs_13));
        float _abs_15;
        asm("abs.f32 %0, %1;" : "=f"(_abs_15) : "f"(zs_0[1]));
        float _neg_14;
        asm("neg.f32 %0, %1;" : "=f"(_neg_14) : "f"(_abs_15));
        float _abs_17;
        asm("abs.f32 %0, %1;" : "=f"(_abs_17) : "f"(zs_0[2]));
        float _neg_16;
        asm("neg.f32 %0, %1;" : "=f"(_neg_16) : "f"(_abs_17));
        float ma_0;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma_0) : "f"(_neg_12), "f"(_neg_14), "f"(_neg_16));
        float _neg_19;
        asm("neg.f32 %0, %1;" : "=f"(_neg_19) : "f"(zs_0[3]));
        float _neg_20;
        asm("neg.f32 %0, %1;" : "=f"(_neg_20) : "f"(zs_0[4]));
        float _max3_18;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_18) : "f"(mz_0), "f"(_neg_19), "f"(_neg_20));
        mz_0 = _max3_18;
        float _abs_23;
        asm("abs.f32 %0, %1;" : "=f"(_abs_23) : "f"(zs_0[3]));
        float _neg_22;
        asm("neg.f32 %0, %1;" : "=f"(_neg_22) : "f"(_abs_23));
        float _abs_25;
        asm("abs.f32 %0, %1;" : "=f"(_abs_25) : "f"(zs_0[4]));
        float _neg_24;
        asm("neg.f32 %0, %1;" : "=f"(_neg_24) : "f"(_abs_25));
        float _max3_21;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_21) : "f"(ma_0), "f"(_neg_22), "f"(_neg_24));
        ma_0 = _max3_21;
        float _neg_27;
        asm("neg.f32 %0, %1;" : "=f"(_neg_27) : "f"(zs_0[5]));
        float _neg_28;
        asm("neg.f32 %0, %1;" : "=f"(_neg_28) : "f"(zs_0[6]));
        float _max3_26;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_26) : "f"(mz_0), "f"(_neg_27), "f"(_neg_28));
        mz_0 = _max3_26;
        float _abs_31;
        asm("abs.f32 %0, %1;" : "=f"(_abs_31) : "f"(zs_0[5]));
        float _neg_30;
        asm("neg.f32 %0, %1;" : "=f"(_neg_30) : "f"(_abs_31));
        float _abs_33;
        asm("abs.f32 %0, %1;" : "=f"(_abs_33) : "f"(zs_0[6]));
        float _neg_32;
        asm("neg.f32 %0, %1;" : "=f"(_neg_32) : "f"(_abs_33));
        float _max3_29;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_29) : "f"(ma_0), "f"(_neg_30), "f"(_neg_32));
        ma_0 = _max3_29;
        float _neg_35;
        asm("neg.f32 %0, %1;" : "=f"(_neg_35) : "f"(zs_0[7]));
        float _max_34;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_34) : "f"(mz_0), "f"(_neg_35));
        mz_0 = _max_34;
        float _abs_38;
        asm("abs.f32 %0, %1;" : "=f"(_abs_38) : "f"(zs_0[7]));
        float _neg_37;
        asm("neg.f32 %0, %1;" : "=f"(_neg_37) : "f"(_abs_38));
        float _max_36;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_36) : "f"(ma_0), "f"(_neg_37));
        ma_0 = _max_36;
        float _abs_39;
        asm("abs.f32 %0, %1;" : "=f"(_abs_39) : "f"(zs_0[0]));
        float _abs_40;
        asm("abs.f32 %0, %1;" : "=f"(_abs_40) : "f"(zs_0[1]));
        float _abs_41;
        asm("abs.f32 %0, %1;" : "=f"(_abs_41) : "f"(zs_0[2]));
        float mp_0;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp_0) : "f"(_abs_39), "f"(_abs_40), "f"(_abs_41));
        float _abs_43;
        asm("abs.f32 %0, %1;" : "=f"(_abs_43) : "f"(zs_0[3]));
        float _abs_44;
        asm("abs.f32 %0, %1;" : "=f"(_abs_44) : "f"(zs_0[4]));
        float _max3_42;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_42) : "f"(mp_0), "f"(_abs_43), "f"(_abs_44));
        mp_0 = _max3_42;
        float _abs_46;
        asm("abs.f32 %0, %1;" : "=f"(_abs_46) : "f"(zs_0[5]));
        float _abs_47;
        asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(zs_0[6]));
        float _max3_45;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_45) : "f"(mp_0), "f"(_abs_46), "f"(_abs_47));
        mp_0 = _max3_45;
        float _abs_49;
        asm("abs.f32 %0, %1;" : "=f"(_abs_49) : "f"(zs_0[7]));
        float _max_48;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_48) : "f"(mp_0), "f"(_abs_49));
        mp_0 = _max_48;
        if (mz_0 <= 41.0f) {
            if (ma_0 <= -1.1754944e-38f) {
                if (mp_0 <= 3.4028235e+38f) {
                    fast_0 = 1;
                }
            }
        }
        if (fast_0 != 0) {
            float2 rstd2_0 = make_float2(rstd_0, rstd_0);
            float2 z2_0 = make_float2(zs_0[0], zs_0[1]);
            float ta_0;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_0) : "f"(zs_0[0]), "f"(-0.00572498f), "f"(0.5f));
            float tb_0;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_0) : "f"(zs_0[1]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_50;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(ta_0), "f"(252.0f), "f"(1.2582913e+07f));
            float _fma_51;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(tb_0), "f"(252.0f), "f"(1.2582913e+07f));
            float2 t2_0 = make_float2(_fma_50, _fma_51);
            float2 r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_0));
            r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.925963e-08f, -1.925963e-08f), r2_0);
            float2 nscale_0 = make_float2(__uint_as_float((__as_u32(t2_0.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_0.y) << 23) + 2147483648u));
            float _exp2_52 = approx_exp2(r2_0.x);
            float _exp2_53 = approx_exp2(r2_0.y);
            float2 nden_0 = fma_f32x2_rn_noftz(nscale_0, make_float2(_exp2_52, _exp2_53), make_float2(-1.0f, -1.0f));
            float _neg_55;
            asm("neg.f32 %0, %1;" : "=f"(_neg_55) : "f"(nden_0.x));
            float _rcp_54 = approx_rcp(_neg_55);
            float _neg_57;
            asm("neg.f32 %0, %1;" : "=f"(_neg_57) : "f"(nden_0.y));
            float _rcp_56 = approx_rcp(_neg_57);
            float2 r0_0 = make_float2(_rcp_54, _rcp_56);
            float2 r1_0 = fma_f32x2_rn_noftz(r0_0, fma_f32x2_rn_noftz(nden_0, r0_0, make_float2(1.0f, 1.0f)), r0_0);
            float2 sig2_0 = fma_f32x2_rn_noftz(r1_0, fma_f32x2_rn_noftz(nden_0, r1_0, make_float2(1.0f, 1.0f)), r1_0);
            float2 q0_0 = fma_f32x2_rn_noftz(z2_0, r1_0, make_float2(0.0f, 0.0f));
            float2 gate2_0 = fma_f32x2_rn_noftz(r1_0, fma_f32x2_rn_noftz(nden_0, q0_0, z2_0), q0_0);
            float2 g2_0 = make_float2(gs_0[0], gs_0[1]);
            float2 w2_0 = make_float2(wv[0], wv[1]);
            float2 _f32x2_mul_a_58 = make_float2(xs_0[0], xs_0[1]);
            float2 n2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&rstd2_0));
            float2 _f32x2_mul_a_59 = mul_f32x2_noftz(mul_f32x2_noftz(g2_0, mul_f32x2_noftz(n2_0, w2_0)), sig2_0);
            float2 _f32x2_mul_b_60 = fma_f32x2_rn_noftz(z2_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_0), make_float2(1.0f, 1.0f));
            float2 dz2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&_f32x2_mul_b_60));
            float2 a2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_0) : "l"(*(const unsigned long long*)&g2_0), "l"(*(const unsigned long long*)&gate2_0));
            float2 dn2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_0) : "l"(*(const unsigned long long*)&a2_0), "l"(*(const unsigned long long*)&w2_0));
            float2 acc2_0 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_0.x) & keep_0), __uint_as_float(__as_u32(a2_0.y) & keep_0)), make_float2(__uint_as_float(__as_u32(n2_0.x) & keep_0), __uint_as_float(__as_u32(n2_0.y) & keep_0)), make_float2(acc[0], acc[1]));
            acc[0] = acc2_0.x;
            acc[1] = acc2_0.y;
            zs_0[0] = dz2_0.x;
            zs_0[1] = dz2_0.y;
            gs_0[0] = dn2_0.x;
            gs_0[1] = dn2_0.y;
            float2 z2_1 = make_float2(zs_0[2], zs_0[3]);
            float ta_1;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_1) : "f"(zs_0[2]), "f"(-0.00572498f), "f"(0.5f));
            float tb_1;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_1) : "f"(zs_0[3]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_61;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(ta_1), "f"(252.0f), "f"(1.2582913e+07f));
            float _fma_62;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(tb_1), "f"(252.0f), "f"(1.2582913e+07f));
            float2 t2_1 = make_float2(_fma_61, _fma_62);
            float2 r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_1));
            r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.925963e-08f, -1.925963e-08f), r2_1);
            float2 nscale_1 = make_float2(__uint_as_float((__as_u32(t2_1.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_1.y) << 23) + 2147483648u));
            float _exp2_63 = approx_exp2(r2_1.x);
            float _exp2_64 = approx_exp2(r2_1.y);
            float2 nden_1 = fma_f32x2_rn_noftz(nscale_1, make_float2(_exp2_63, _exp2_64), make_float2(-1.0f, -1.0f));
            float _neg_66;
            asm("neg.f32 %0, %1;" : "=f"(_neg_66) : "f"(nden_1.x));
            float _rcp_65 = approx_rcp(_neg_66);
            float _neg_68;
            asm("neg.f32 %0, %1;" : "=f"(_neg_68) : "f"(nden_1.y));
            float _rcp_67 = approx_rcp(_neg_68);
            float2 r0_1 = make_float2(_rcp_65, _rcp_67);
            float2 r1_1 = fma_f32x2_rn_noftz(r0_1, fma_f32x2_rn_noftz(nden_1, r0_1, make_float2(1.0f, 1.0f)), r0_1);
            float2 sig2_1 = fma_f32x2_rn_noftz(r1_1, fma_f32x2_rn_noftz(nden_1, r1_1, make_float2(1.0f, 1.0f)), r1_1);
            float2 q0_1 = fma_f32x2_rn_noftz(z2_1, r1_1, make_float2(0.0f, 0.0f));
            float2 gate2_1 = fma_f32x2_rn_noftz(r1_1, fma_f32x2_rn_noftz(nden_1, q0_1, z2_1), q0_1);
            float2 g2_1 = make_float2(gs_0[2], gs_0[3]);
            float2 w2_1 = make_float2(wv[2], wv[3]);
            float2 _f32x2_mul_a_69 = make_float2(xs_0[2], xs_0[3]);
            float2 n2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&rstd2_0));
            float2 _f32x2_mul_a_70 = mul_f32x2_noftz(mul_f32x2_noftz(g2_1, mul_f32x2_noftz(n2_1, w2_1)), sig2_1);
            float2 _f32x2_mul_b_71 = fma_f32x2_rn_noftz(z2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_1), make_float2(1.0f, 1.0f));
            float2 dz2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
            float2 a2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_1) : "l"(*(const unsigned long long*)&g2_1), "l"(*(const unsigned long long*)&gate2_1));
            float2 dn2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_1) : "l"(*(const unsigned long long*)&a2_1), "l"(*(const unsigned long long*)&w2_1));
            float2 acc2_1 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_1.x) & keep_0), __uint_as_float(__as_u32(a2_1.y) & keep_0)), make_float2(__uint_as_float(__as_u32(n2_1.x) & keep_0), __uint_as_float(__as_u32(n2_1.y) & keep_0)), make_float2(acc[2], acc[3]));
            acc[2] = acc2_1.x;
            acc[3] = acc2_1.y;
            zs_0[2] = dz2_1.x;
            zs_0[3] = dz2_1.y;
            gs_0[2] = dn2_1.x;
            gs_0[3] = dn2_1.y;
            float2 z2_2 = make_float2(zs_0[4], zs_0[5]);
            float ta_2;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_2) : "f"(zs_0[4]), "f"(-0.00572498f), "f"(0.5f));
            float tb_2;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_2) : "f"(zs_0[5]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_72;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(ta_2), "f"(252.0f), "f"(1.2582913e+07f));
            float _fma_73;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(tb_2), "f"(252.0f), "f"(1.2582913e+07f));
            float2 t2_2 = make_float2(_fma_72, _fma_73);
            float2 r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_2));
            r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.925963e-08f, -1.925963e-08f), r2_2);
            float2 nscale_2 = make_float2(__uint_as_float((__as_u32(t2_2.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_2.y) << 23) + 2147483648u));
            float _exp2_74 = approx_exp2(r2_2.x);
            float _exp2_75 = approx_exp2(r2_2.y);
            float2 nden_2 = fma_f32x2_rn_noftz(nscale_2, make_float2(_exp2_74, _exp2_75), make_float2(-1.0f, -1.0f));
            float _neg_77;
            asm("neg.f32 %0, %1;" : "=f"(_neg_77) : "f"(nden_2.x));
            float _rcp_76 = approx_rcp(_neg_77);
            float _neg_79;
            asm("neg.f32 %0, %1;" : "=f"(_neg_79) : "f"(nden_2.y));
            float _rcp_78 = approx_rcp(_neg_79);
            float2 r0_2 = make_float2(_rcp_76, _rcp_78);
            float2 r1_2 = fma_f32x2_rn_noftz(r0_2, fma_f32x2_rn_noftz(nden_2, r0_2, make_float2(1.0f, 1.0f)), r0_2);
            float2 sig2_2 = fma_f32x2_rn_noftz(r1_2, fma_f32x2_rn_noftz(nden_2, r1_2, make_float2(1.0f, 1.0f)), r1_2);
            float2 q0_2 = fma_f32x2_rn_noftz(z2_2, r1_2, make_float2(0.0f, 0.0f));
            float2 gate2_2 = fma_f32x2_rn_noftz(r1_2, fma_f32x2_rn_noftz(nden_2, q0_2, z2_2), q0_2);
            float2 g2_2 = make_float2(gs_0[4], gs_0[5]);
            float2 w2_2 = make_float2(wv[4], wv[5]);
            float2 _f32x2_mul_a_80 = make_float2(xs_0[4], xs_0[5]);
            float2 n2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&rstd2_0));
            float2 _f32x2_mul_a_81 = mul_f32x2_noftz(mul_f32x2_noftz(g2_2, mul_f32x2_noftz(n2_2, w2_2)), sig2_2);
            float2 _f32x2_mul_b_82 = fma_f32x2_rn_noftz(z2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_2), make_float2(1.0f, 1.0f));
            float2 dz2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&_f32x2_mul_b_82));
            float2 a2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_2) : "l"(*(const unsigned long long*)&g2_2), "l"(*(const unsigned long long*)&gate2_2));
            float2 dn2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_2) : "l"(*(const unsigned long long*)&a2_2), "l"(*(const unsigned long long*)&w2_2));
            float2 acc2_2 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_2.x) & keep_0), __uint_as_float(__as_u32(a2_2.y) & keep_0)), make_float2(__uint_as_float(__as_u32(n2_2.x) & keep_0), __uint_as_float(__as_u32(n2_2.y) & keep_0)), make_float2(acc[4], acc[5]));
            acc[4] = acc2_2.x;
            acc[5] = acc2_2.y;
            zs_0[4] = dz2_2.x;
            zs_0[5] = dz2_2.y;
            gs_0[4] = dn2_2.x;
            gs_0[5] = dn2_2.y;
            float2 z2_3 = make_float2(zs_0[6], zs_0[7]);
            float ta_3;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_3) : "f"(zs_0[6]), "f"(-0.00572498f), "f"(0.5f));
            float tb_3;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_3) : "f"(zs_0[7]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_83;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(ta_3), "f"(252.0f), "f"(1.2582913e+07f));
            float _fma_84;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(tb_3), "f"(252.0f), "f"(1.2582913e+07f));
            float2 t2_3 = make_float2(_fma_83, _fma_84);
            float2 r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_3));
            r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.925963e-08f, -1.925963e-08f), r2_3);
            float2 nscale_3 = make_float2(__uint_as_float((__as_u32(t2_3.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_3.y) << 23) + 2147483648u));
            float _exp2_85 = approx_exp2(r2_3.x);
            float _exp2_86 = approx_exp2(r2_3.y);
            float2 nden_3 = fma_f32x2_rn_noftz(nscale_3, make_float2(_exp2_85, _exp2_86), make_float2(-1.0f, -1.0f));
            float _neg_88;
            asm("neg.f32 %0, %1;" : "=f"(_neg_88) : "f"(nden_3.x));
            float _rcp_87 = approx_rcp(_neg_88);
            float _neg_90;
            asm("neg.f32 %0, %1;" : "=f"(_neg_90) : "f"(nden_3.y));
            float _rcp_89 = approx_rcp(_neg_90);
            float2 r0_3 = make_float2(_rcp_87, _rcp_89);
            float2 r1_3 = fma_f32x2_rn_noftz(r0_3, fma_f32x2_rn_noftz(nden_3, r0_3, make_float2(1.0f, 1.0f)), r0_3);
            float2 sig2_3 = fma_f32x2_rn_noftz(r1_3, fma_f32x2_rn_noftz(nden_3, r1_3, make_float2(1.0f, 1.0f)), r1_3);
            float2 q0_3 = fma_f32x2_rn_noftz(z2_3, r1_3, make_float2(0.0f, 0.0f));
            float2 gate2_3 = fma_f32x2_rn_noftz(r1_3, fma_f32x2_rn_noftz(nden_3, q0_3, z2_3), q0_3);
            float2 g2_3 = make_float2(gs_0[6], gs_0[7]);
            float2 w2_3 = make_float2(wv[6], wv[7]);
            float2 _f32x2_mul_a_91 = make_float2(xs_0[6], xs_0[7]);
            float2 n2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_91), "l"(*(const unsigned long long*)&rstd2_0));
            float2 _f32x2_mul_a_92 = mul_f32x2_noftz(mul_f32x2_noftz(g2_3, mul_f32x2_noftz(n2_3, w2_3)), sig2_3);
            float2 _f32x2_mul_b_93 = fma_f32x2_rn_noftz(z2_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_3), make_float2(1.0f, 1.0f));
            float2 dz2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_92), "l"(*(const unsigned long long*)&_f32x2_mul_b_93));
            float2 a2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_3) : "l"(*(const unsigned long long*)&g2_3), "l"(*(const unsigned long long*)&gate2_3));
            float2 dn2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_3) : "l"(*(const unsigned long long*)&a2_3), "l"(*(const unsigned long long*)&w2_3));
            float2 acc2_3 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_3.x) & keep_0), __uint_as_float(__as_u32(a2_3.y) & keep_0)), make_float2(__uint_as_float(__as_u32(n2_3.x) & keep_0), __uint_as_float(__as_u32(n2_3.y) & keep_0)), make_float2(acc[6], acc[7]));
            acc[6] = acc2_3.x;
            acc[7] = acc2_3.y;
            zs_0[6] = dz2_3.x;
            zs_0[7] = dz2_3.y;
            gs_0[6] = dn2_3.x;
            gs_0[7] = dn2_3.y;
            dot_0 = __fmaf_rn(gs_0[0], xs_0[0], dot_0);
            dot_0 = __fmaf_rn(gs_0[1], xs_0[1], dot_0);
            dot_0 = __fmaf_rn(gs_0[2], xs_0[2], dot_0);
            dot_0 = __fmaf_rn(gs_0[3], xs_0[3], dot_0);
            dot_0 = __fmaf_rn(gs_0[4], xs_0[4], dot_0);
            dot_0 = __fmaf_rn(gs_0[5], xs_0[5], dot_0);
            dot_0 = __fmaf_rn(gs_0[6], xs_0[6], dot_0);
            dot_0 = __fmaf_rn(gs_0[7], xs_0[7], dot_0);
        } else {
            float _exp_94 = __nv_expf(zs_0[0] * -1.0f);
            float den_0 = 1.0f + _exp_94;
            float sig_0 = __fdiv_rn(1.0f, den_0);
            float gate_0 = __fdiv_rn(zs_0[0], den_0);
            float nj_0 = xs_0[0] * rstd_0;
            float aj_0 = gs_0[0] * gate_0;
            float dzj_0 = gs_0[0] * (nj_0 * wv[0]) * sig_0 * __fmaf_rn(zs_0[0], 1.0f - sig_0, 1.0f);
            float dnj_0 = aj_0 * wv[0];
            dot_0 = __fmaf_rn(dnj_0, xs_0[0], dot_0);
            acc[0] = __fmaf_rn(__uint_as_float(__as_u32(aj_0) & keep_0), __uint_as_float(__as_u32(nj_0) & keep_0), acc[0]);
            zs_0[0] = dzj_0;
            gs_0[0] = dnj_0;
            float _exp_95 = __nv_expf(zs_0[1] * -1.0f);
            float den_1 = 1.0f + _exp_95;
            float sig_1 = __fdiv_rn(1.0f, den_1);
            float gate_1 = __fdiv_rn(zs_0[1], den_1);
            float nj_1 = xs_0[1] * rstd_0;
            float aj_1 = gs_0[1] * gate_1;
            float dzj_1 = gs_0[1] * (nj_1 * wv[1]) * sig_1 * __fmaf_rn(zs_0[1], 1.0f - sig_1, 1.0f);
            float dnj_1 = aj_1 * wv[1];
            dot_0 = __fmaf_rn(dnj_1, xs_0[1], dot_0);
            acc[1] = __fmaf_rn(__uint_as_float(__as_u32(aj_1) & keep_0), __uint_as_float(__as_u32(nj_1) & keep_0), acc[1]);
            zs_0[1] = dzj_1;
            gs_0[1] = dnj_1;
            float _exp_96 = __nv_expf(zs_0[2] * -1.0f);
            float den_2 = 1.0f + _exp_96;
            float sig_2 = __fdiv_rn(1.0f, den_2);
            float gate_2 = __fdiv_rn(zs_0[2], den_2);
            float nj_2 = xs_0[2] * rstd_0;
            float aj_2 = gs_0[2] * gate_2;
            float dzj_2 = gs_0[2] * (nj_2 * wv[2]) * sig_2 * __fmaf_rn(zs_0[2], 1.0f - sig_2, 1.0f);
            float dnj_2 = aj_2 * wv[2];
            dot_0 = __fmaf_rn(dnj_2, xs_0[2], dot_0);
            acc[2] = __fmaf_rn(__uint_as_float(__as_u32(aj_2) & keep_0), __uint_as_float(__as_u32(nj_2) & keep_0), acc[2]);
            zs_0[2] = dzj_2;
            gs_0[2] = dnj_2;
            float _exp_97 = __nv_expf(zs_0[3] * -1.0f);
            float den_3 = 1.0f + _exp_97;
            float sig_3 = __fdiv_rn(1.0f, den_3);
            float gate_3 = __fdiv_rn(zs_0[3], den_3);
            float nj_3 = xs_0[3] * rstd_0;
            float aj_3 = gs_0[3] * gate_3;
            float dzj_3 = gs_0[3] * (nj_3 * wv[3]) * sig_3 * __fmaf_rn(zs_0[3], 1.0f - sig_3, 1.0f);
            float dnj_3 = aj_3 * wv[3];
            dot_0 = __fmaf_rn(dnj_3, xs_0[3], dot_0);
            acc[3] = __fmaf_rn(__uint_as_float(__as_u32(aj_3) & keep_0), __uint_as_float(__as_u32(nj_3) & keep_0), acc[3]);
            zs_0[3] = dzj_3;
            gs_0[3] = dnj_3;
            float _exp_98 = __nv_expf(zs_0[4] * -1.0f);
            float den_4 = 1.0f + _exp_98;
            float sig_4 = __fdiv_rn(1.0f, den_4);
            float gate_4 = __fdiv_rn(zs_0[4], den_4);
            float nj_4 = xs_0[4] * rstd_0;
            float aj_4 = gs_0[4] * gate_4;
            float dzj_4 = gs_0[4] * (nj_4 * wv[4]) * sig_4 * __fmaf_rn(zs_0[4], 1.0f - sig_4, 1.0f);
            float dnj_4 = aj_4 * wv[4];
            dot_0 = __fmaf_rn(dnj_4, xs_0[4], dot_0);
            acc[4] = __fmaf_rn(__uint_as_float(__as_u32(aj_4) & keep_0), __uint_as_float(__as_u32(nj_4) & keep_0), acc[4]);
            zs_0[4] = dzj_4;
            gs_0[4] = dnj_4;
            float _exp_99 = __nv_expf(zs_0[5] * -1.0f);
            float den_5 = 1.0f + _exp_99;
            float sig_5 = __fdiv_rn(1.0f, den_5);
            float gate_5 = __fdiv_rn(zs_0[5], den_5);
            float nj_5 = xs_0[5] * rstd_0;
            float aj_5 = gs_0[5] * gate_5;
            float dzj_5 = gs_0[5] * (nj_5 * wv[5]) * sig_5 * __fmaf_rn(zs_0[5], 1.0f - sig_5, 1.0f);
            float dnj_5 = aj_5 * wv[5];
            dot_0 = __fmaf_rn(dnj_5, xs_0[5], dot_0);
            acc[5] = __fmaf_rn(__uint_as_float(__as_u32(aj_5) & keep_0), __uint_as_float(__as_u32(nj_5) & keep_0), acc[5]);
            zs_0[5] = dzj_5;
            gs_0[5] = dnj_5;
            float _exp_100 = __nv_expf(zs_0[6] * -1.0f);
            float den_6 = 1.0f + _exp_100;
            float sig_6 = __fdiv_rn(1.0f, den_6);
            float gate_6 = __fdiv_rn(zs_0[6], den_6);
            float nj_6 = xs_0[6] * rstd_0;
            float aj_6 = gs_0[6] * gate_6;
            float dzj_6 = gs_0[6] * (nj_6 * wv[6]) * sig_6 * __fmaf_rn(zs_0[6], 1.0f - sig_6, 1.0f);
            float dnj_6 = aj_6 * wv[6];
            dot_0 = __fmaf_rn(dnj_6, xs_0[6], dot_0);
            acc[6] = __fmaf_rn(__uint_as_float(__as_u32(aj_6) & keep_0), __uint_as_float(__as_u32(nj_6) & keep_0), acc[6]);
            zs_0[6] = dzj_6;
            gs_0[6] = dnj_6;
            float _exp_101 = __nv_expf(zs_0[7] * -1.0f);
            float den_7 = 1.0f + _exp_101;
            float sig_7 = __fdiv_rn(1.0f, den_7);
            float gate_7 = __fdiv_rn(zs_0[7], den_7);
            float nj_7 = xs_0[7] * rstd_0;
            float aj_7 = gs_0[7] * gate_7;
            float dzj_7 = gs_0[7] * (nj_7 * wv[7]) * sig_7 * __fmaf_rn(zs_0[7], 1.0f - sig_7, 1.0f);
            float dnj_7 = aj_7 * wv[7];
            dot_0 = __fmaf_rn(dnj_7, xs_0[7], dot_0);
            acc[7] = __fmaf_rn(__uint_as_float(__as_u32(aj_7) & keep_0), __uint_as_float(__as_u32(nj_7) & keep_0), acc[7]);
            zs_0[7] = dzj_7;
            gs_0[7] = dnj_7;
        }
        float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
        dot_0 = dot_0 + _shfl_xor_sync_102;
        float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
        dot_0 = dot_0 + _shfl_xor_sync_103;
        float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 4);
        dot_0 = dot_0 + _shfl_xor_sync_104;
        float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 8);
        dot_0 = dot_0 + _shfl_xor_sync_105;
        float coef_0 = rstd_0 * rstd_0 * rstd_0 * (dot_0 * -0.0078125f);
        float2 coef2_0 = make_float2(coef_0, coef_0);
        float2 rstd2b_0 = make_float2(rstd_0, rstd_0);
        float2 dx2_0 = fma_f32x2_rn_noftz(make_float2(xs_0[0], xs_0[1]), coef2_0, mul_f32x2_noftz(make_float2(gs_0[0], gs_0[1]), rstd2b_0));
        xs_0[0] = dx2_0.x;
        xs_0[1] = dx2_0.y;
        float2 dx2_1 = fma_f32x2_rn_noftz(make_float2(xs_0[2], xs_0[3]), coef2_0, mul_f32x2_noftz(make_float2(gs_0[2], gs_0[3]), rstd2b_0));
        xs_0[2] = dx2_1.x;
        xs_0[3] = dx2_1.y;
        float2 dx2_2 = fma_f32x2_rn_noftz(make_float2(xs_0[4], xs_0[5]), coef2_0, mul_f32x2_noftz(make_float2(gs_0[4], gs_0[5]), rstd2b_0));
        xs_0[4] = dx2_2.x;
        xs_0[5] = dx2_2.y;
        float2 dx2_3 = fma_f32x2_rn_noftz(make_float2(xs_0[6], xs_0[7]), coef2_0, mul_f32x2_noftz(make_float2(gs_0[6], gs_0[7]), rstd2b_0));
        xs_0[6] = dx2_3.x;
        xs_0[7] = dx2_3.y;
        unsigned int packed_dx_0[4];
        unsigned int packed_dz_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 0], xs_0[_lp*2+1 + 0]));
            packed_dx_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 0], zs_0[_lp*2+1 + 0]));
            packed_dz_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 2], xs_0[_lp*2+1 + 2]));
            packed_dx_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 2], zs_0[_lp*2+1 + 2]));
            packed_dz_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 4], xs_0[_lp*2+1 + 4]));
            packed_dx_0[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 4], zs_0[_lp*2+1 + 4]));
            packed_dz_0[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 6], xs_0[_lp*2+1 + 6]));
            packed_dx_0[_lp + 3] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 6], zs_0[_lp*2+1 + 6]));
            packed_dz_0[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (row_0 < rows) {
            long long orow_0 = (long long)row_0 * 128ll + (long long)col;
            reinterpret_cast<int4*>(dx + orow_0)[0] = reinterpret_cast<int4*>(packed_dx_0)[0];
            reinterpret_cast<int4*>(dz + orow_0)[0] = reinterpret_cast<int4*>(packed_dz_0)[0];
        }
    }
    float part32[8];
    float p32_0 = acc[0];
    float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, p32_0, 16);
    p32_0 = p32_0 + _shfl_xor_sync_106;
    part32[0] = p32_0;
    float p32_1 = acc[1];
    float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, p32_1, 16);
    p32_1 = p32_1 + _shfl_xor_sync_107;
    part32[1] = p32_1;
    float p32_2 = acc[2];
    float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, p32_2, 16);
    p32_2 = p32_2 + _shfl_xor_sync_108;
    part32[2] = p32_2;
    float p32_3 = acc[3];
    float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, p32_3, 16);
    p32_3 = p32_3 + _shfl_xor_sync_109;
    part32[3] = p32_3;
    float p32_4 = acc[4];
    float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, p32_4, 16);
    p32_4 = p32_4 + _shfl_xor_sync_110;
    part32[4] = p32_4;
    float p32_5 = acc[5];
    float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, p32_5, 16);
    p32_5 = p32_5 + _shfl_xor_sync_111;
    part32[5] = p32_5;
    float p32_6 = acc[6];
    float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, p32_6, 16);
    p32_6 = p32_6 + _shfl_xor_sync_112;
    part32[6] = p32_6;
    float p32_7 = acc[7];
    float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, p32_7, 16);
    p32_7 = p32_7 + _shfl_xor_sync_113;
    part32[7] = p32_7;
    if ((tid & 31) < 16) {
        fold32[(tid >> 5) * 128 + col] = part32[0];
        fold32[(tid >> 5) * 128 + col + 1] = part32[1];
        fold32[(tid >> 5) * 128 + col + 2] = part32[2];
        fold32[(tid >> 5) * 128 + col + 3] = part32[3];
        fold32[(tid >> 5) * 128 + col + 4] = part32[4];
        fold32[(tid >> 5) * 128 + col + 5] = part32[5];
        fold32[(tid >> 5) * 128 + col + 6] = part32[6];
        fold32[(tid >> 5) * 128 + col + 7] = part32[7];
    }
    __syncthreads();
    if (tid < 128) {
        float w32[8];
        w32[0] = fold32[tid];
        w32[1] = fold32[128 + tid];
        w32[2] = fold32[256 + tid];
        w32[3] = fold32[384 + tid];
        w32[4] = fold32[512 + tid];
        w32[5] = fold32[640 + tid];
        w32[6] = fold32[768 + tid];
        w32[7] = fold32[896 + tid];
        w32[0] = w32[0] + w32[1];
        w32[2] = w32[2] + w32[3];
        w32[4] = w32[4] + w32[5];
        w32[6] = w32[6] + w32[7];
        w32[0] = w32[0] + w32[2];
        w32[4] = w32[4] + w32[6];
        w32[0] = w32[0] + w32[4];
        part[tid] = w32[0];
    }
    __syncthreads();
    if (weight_grad != 0) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                asm volatile(
                    "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 _, [%0], %1;"
                    :: "r"(recv_full_addr), "r"((uint32_t)(512)) : "memory");
            }
        }
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (weight_grad != 0) {
        if (tid < 32) {
            float four[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&four[0])), "=r"(*reinterpret_cast<uint32_t*>(&four[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&four[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&four[(0) + 3]))
                : "r"(part_addr + (unsigned int)(16 * tid)));
            int owner = 4 * tid / 8;
            int slot_at = rank * 8 + 4 * tid - owner * 8;
            uint32_t _cluster_mapa_114;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_114) : "r"(recv_addr + (unsigned int)(4 * slot_at)), "r"(owner));
            uint32_t _mbar_peer_115;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_115) : "r"(recv_full_addr), "r"(owner));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_cluster_mapa_114), "r"(__float_as_uint(four[0])), "r"(__float_as_uint(four[1])), "r"(__float_as_uint(four[2])), "r"(__float_as_uint(four[3])), "r"(_mbar_peer_115) : "memory");
        }
        mbarrier_wait_cluster_hint(recv_full_addr, 0, 10000000);
        if (tid < 8) {
            double sum64 = 0.0;
            sum64 = sum64 + (double)recv[tid];
            sum64 = sum64 + (double)recv[8 + tid];
            sum64 = sum64 + (double)recv[16 + tid];
            sum64 = sum64 + (double)recv[24 + tid];
            sum64 = sum64 + (double)recv[32 + tid];
            sum64 = sum64 + (double)recv[40 + tid];
            sum64 = sum64 + (double)recv[48 + tid];
            sum64 = sum64 + (double)recv[56 + tid];
            sum64 = sum64 + (double)recv[64 + tid];
            sum64 = sum64 + (double)recv[72 + tid];
            sum64 = sum64 + (double)recv[80 + tid];
            sum64 = sum64 + (double)recv[88 + tid];
            sum64 = sum64 + (double)recv[96 + tid];
            sum64 = sum64 + (double)recv[104 + tid];
            sum64 = sum64 + (double)recv[112 + tid];
            sum64 = sum64 + (double)recv[120 + tid];
            int cpush = rank * 8 + tid;
            if (accumulate != 0) {
                __nv_bfloat16 prev_c = dw[cpush];
                sum64 = sum64 + (double)(float)prev_c;
            }
            dw[cpush] = (__nv_bfloat16)(float)sum64;
        }
    }
}

} // extern "C"
