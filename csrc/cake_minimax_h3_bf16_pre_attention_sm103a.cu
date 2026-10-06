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
// MiniMax-H3 fused BF16 pre-attention for Blackwell (compute capability 10.0 / 10.3; one
// fatbin source).  Generated device code; two kernels per operator call:
//
//   a   = BF16(BF16(RMSNorm_fp32(x, x_norm_weight, eps)) * BF16(1 + adaln_scale[idx])
//              + adaln_shift[idx])                        -> workspace: bf16 [M, 5376]
//         adaln_scale / adaln_shift are bf16 [rows, 5376] tables read as base + idx * adaln_row_stride
//         (a contiguous table or a column chunk of a wider [rows, 6 * 5376] projection); rows whose
//         int64 idx lies outside [0, adaln_rows) are zero (zero Q/K/V row)
//   qkv = BF16(a @ qkv_weight^T)                          qkv_weight: bf16 [3 * 56 * 128, 5376],
//                                                         row order [qkv_kind, head, head_dim]
//   q,k = RMSNorm_fp32(BF16 head, q_norm_weight / k_norm_weight, qk_eps)
//   q,k = partial NeoX RoPE over the first 96 of 128 lanes with cache row
//         rope_cos_sin[clamp(rope_positions[m], 0, S - 1)]  ([S, 96] bf16: 48 cos then 48 sin)
//   out[p, m, h, kind, :] for the destination-major Ulysses pack [P, M, 56 / P, 3, 128]
//
// Kernel 1 (norm + AdaLN) is a plain 512-thread launch, four warps per row; its FP32 reduction
// reproduces PyTorch's vectorized RMSNorm association.  Kernel 2 is a persistent 2-CTA
// (cta_group::2) tcgen05 GEMM with TMEM accumulators and TMA operand loads whose epilogue applies
// the per-head Q/K RMSNorm, the RoPE and the pack; it requires a cluster launch of (2, 1, 1) and
// does not compile for SM90 or SM120.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

static_assert(sizeof(CUtensorMap) == 128, "CUDA tensor-map ABI size mismatch");
static_assert(alignof(CUtensorMap) >= 64, "CUDA tensor-map ABI requires at least 64-byte alignment");


__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

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

union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void elect_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
}

__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}

#define MINIMAX_H3_PRE_ATTENTION_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_PARTIALS_OFF 0
#define SMEM_SMEM_PARTIALS_STAGE_BYTES 32
#define SMEM_SMEM_PARTIALS_STRIDE 32
#define SMEM_SMEM_RSTD_OFF 64
#define SMEM_SMEM_RSTD_STAGE_BYTES 16
#define SMEM_SMEM_RSTD_STRIDE 16
#define SMEM_TOTAL 128
#define HIDDEN 5376
#define ROWS_PER_CTA 4

extern "C" {

__global__ __launch_bounds__(512) void
kernel_minimax_h3_pre_attention_norm_adaln_bf16(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ x_norm_weight, __nv_bfloat16* __restrict__ adaln_scale, __nv_bfloat16* __restrict__ adaln_shift, long long* __restrict__ adaln_index, __nv_bfloat16* __restrict__ a_out, int M, int adaln_rows, long long adaln_row_stride, float eps)
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
    float* smem_partials = reinterpret_cast<float*>(smem_raw + 0);
    const int smem_partials_addr = smem + 0;
    float* smem_rstd = reinterpret_cast<float*>(smem_raw + 64);
    const int smem_rstd_addr = smem + 64;

    // === Task calls (dependency order) ===
    int row_slot = warp / 4;
    int warp_in_row = warp % 4;
    int thread_in_row = warp_in_row * 32 + lane;
    int global_row = bid * ROWS_PER_CTA + row_slot;
    float sum_sq = 0.0f;
    if (global_row < M) {
        unsigned long long load_base = (unsigned long long)global_row * (unsigned long long)HIDDEN;
        #pragma unroll
        for (int vec_iter = 0; vec_iter < 11; vec_iter++) {
            int vec_index = thread_in_row + vec_iter * 128;
            if (vec_index < HIDDEN / 4) {
                float _vec_load_0[4];
                {
                    uint2 _vld_0;
                    _vld_0 = *reinterpret_cast<const uint2*>(x + (load_base + (unsigned long long)(vec_index * 4)) + 0);
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_0[0 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _pair * 2])[1])
                            : "r"(_vpairs_0[_pair]));
                    }
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    sum_sq += _vec_load_0[j] * _vec_load_0[j];
                }
            }
        }
    }
    float _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, sum_sq, 16, 32);
    float sh16 = _shfl_down_0;
    sum_sq += sh16;
    float _shfl_down_1 = __shfl_down_sync(0xFFFFFFFF, sum_sq, 8, 32);
    float sh8 = _shfl_down_1;
    sum_sq += sh8;
    float _shfl_down_2 = __shfl_down_sync(0xFFFFFFFF, sum_sq, 4, 32);
    float sh4 = _shfl_down_2;
    sum_sq += sh4;
    float _shfl_down_3 = __shfl_down_sync(0xFFFFFFFF, sum_sq, 2, 32);
    float sh2 = _shfl_down_3;
    sum_sq += sh2;
    float _shfl_down_4 = __shfl_down_sync(0xFFFFFFFF, sum_sq, 1, 32);
    float sh1 = _shfl_down_4;
    sum_sq += sh1;
    if (lane == 0) {
        if (warp_in_row >= 2) {
            smem_partials[row_slot * 2 + warp_in_row - 2] = sum_sq;
        }
    }
    asm volatile("barrier.sync 1, 512;" ::: "memory");
    if (lane == 0) {
        if (warp_in_row < 2) {
            sum_sq = sum_sq + smem_partials[row_slot * 2 + warp_in_row];
        }
    }
    asm volatile("barrier.sync 1, 512;" ::: "memory");
    if (lane == 0) {
        if (warp_in_row == 1) {
            smem_partials[row_slot] = sum_sq;
        }
    }
    asm volatile("barrier.sync 1, 512;" ::: "memory");
    if (lane == 0) {
        if (warp_in_row == 0) {
            sum_sq = sum_sq + smem_partials[row_slot];
            float _rsqrt_0 = rsqrtf(sum_sq / (float)HIDDEN + eps);
            smem_rstd[row_slot] = _rsqrt_0;
        }
    }
    asm volatile("barrier.sync 1, 512;" ::: "memory");
    if (global_row < M) {
        float rstd = smem_rstd[row_slot];
        unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)HIDDEN;
        long long table_index = adaln_index[global_row];
        if (table_index >= 0 && table_index < (long long)adaln_rows) {
            unsigned long long table_base = (unsigned long long)table_index * (unsigned long long)adaln_row_stride;
            #pragma unroll
            for (int vec_iter_1 = 0; vec_iter_1 < 6; vec_iter_1++) {
                int vec8 = thread_in_row + vec_iter_1 * 128;
                if (vec8 < HIDDEN / 8) {
                    int k = vec8 * 8;
                    float _vec_load_1[8];
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(x + (row_base + (unsigned long long)k) + 0);
                        uint4 _vld_1[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_1[_pair]));
                            }
                        }
                    }
                    float _vec_load_2[8];
                    {
                        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(x_norm_weight + k + 0);
                        uint4 _vld_2[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_2[_blk] = _vptr_2[_blk];
                            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_2[_pair]));
                            }
                        }
                    }
                    float _vec_load_3[8];
                    {
                        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(adaln_scale + (table_base + (unsigned long long)k) + 0);
                        uint4 _vld_3[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_3[_blk] = _vptr_3[_blk];
                            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_3[_pair]));
                            }
                        }
                    }
                    float _vec_load_4[8];
                    {
                        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(adaln_shift + (table_base + (unsigned long long)k) + 0);
                        uint4 _vld_4[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_4[_blk] = _vptr_4[_blk];
                            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_4[_pair]));
                            }
                        }
                    }
                    float vals[8];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 8; j_1++) {
                        float scaled = rstd * _vec_load_1[j_1];
                        __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_vec_load_2[j_1] * scaled);
                        float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                        float norm_value = _cvt_f32_0;
                        __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_vec_load_3[j_1] + 1.0f);
                        float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                        float scale_plus_one = _cvt_f32_1;
                        float _fma_0 = __fmaf_rn(norm_value, scale_plus_one, _vec_load_4[j_1]);
                        __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(_fma_0);
                        float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                        vals[j_1] = _cvt_f32_2;
                    }
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(a_out + (row_base + (unsigned long long)k)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int vec_iter_2 = 0; vec_iter_2 < 6; vec_iter_2++) {
                int vec8_1 = thread_in_row + vec_iter_2 * 128;
                if (vec8_1 < HIDDEN / 8) {
                    int k_1 = vec8_1 * 8;
                    float zeros[8];
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 8; j_2++) {
                        zeros[j_2] = 0.0f;
                    }
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(a_out + (row_base + (unsigned long long)k_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef HIDDEN
#undef MINIMAX_H3_PRE_ATTENTION_INF
#undef NUM_MAIN_STAGES
#undef ROWS_PER_CTA
#undef SMEM_SMEM_PARTIALS_OFF
#undef SMEM_SMEM_PARTIALS_STAGE_BYTES
#undef SMEM_SMEM_PARTIALS_STRIDE
#undef SMEM_SMEM_RSTD_OFF
#undef SMEM_SMEM_RSTD_STAGE_BYTES
#undef SMEM_SMEM_RSTD_STRIDE
#undef SMEM_TOTAL
#undef smem_partials_addr
#undef smem_rstd_addr

#define MINIMAX_H3_PRE_ATTENTION_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 7
#define NUM_MAINLOOP_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_WORK_RESPONSE_OFF 230400
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 230528
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 64
#define MMA_K 16
#define CTA_GROUP 2
#define NUM_STAGES 7
#define NUM_EPILOGUE_WARPS 4
#define GROUP_M 32
#define WORK_STAGES 4
#define WORK_CONSUMERS 290
#define NUM_K_ITERS 84
#define N_TILES 84
#define NUM_HEADS 56
#define HEAD_DIM 128
#define QKV_KINDS 3
#define ROPE_DIM 96
#define ROPE_HALF 48
#define num_cluster_tiles ((m_tiles / CTA_GROUP) * N_TILES)
#define tiles_per_group (GROUP_M * N_TILES)

extern "C" {

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_minimax_h3_bf16_pre_attention_gemm(const __grid_constant__ CUtensorMap activation, const __grid_constant__ CUtensorMap qkv_weight, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, long long* __restrict__ rope_positions, __nv_bfloat16* __restrict__ out, int M, int m_tiles, int P, int rope_rows, float qk_eps)
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
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 56)
    #define mainloop_done_addr (mbar_base + 112)
    #define epilogue_done_addr (mbar_base + 128)
    #define work_full_addr (mbar_base + 144)
    #define work_empty_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 230400);
    const int work_response_addr = smem + 230400;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 26 barriers)
    // Mbarriers at smem_raw[0..208)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 7 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            // mma_done: 7 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // epilogue_done: 2 barriers, init_count=8
            mbarrier_init(smem + 128, 8);
            mbarrier_init(smem + 136, 8);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // work_empty: 4 barriers, init_count=290
            mbarrier_init(smem + 176, 290);
            mbarrier_init(smem + 184, 290);
            mbarrier_init(smem + 192, 290);
            mbarrier_init(smem + 200, 290);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    if (warp == 0) {
        int _tmem_hold = smem + 208;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < num_cluster_tiles; _tile_iter++) {
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                            : "memory");
                    }
                    int group = this_bid / (unsigned int)tiles_per_group;
                    int first_m = group * GROUP_M;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                    int local = this_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int weight_row = (bid_n * 2 + cta_rank) * B_HALF_N;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int off_k = iter_k * BLOCK_K;
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&activation), 0, off_m, off_k / 64, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 32768, (&qkv_weight), 0, weight_row, off_k / 64, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        load_stage += 1;
                        if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                    uint32_t _clc_valid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                    this_bid = _clc_ctaid_0 + (unsigned int)cta_rank;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_epi_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_cluster_tiles; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < NUM_K_ITERS; iter_k_1++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (mma_epi_stage * 256))), "r"(((init_flag) ? 0 : 1)));
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 7) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                    mma_epi_stage += 1;
                    if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    uint32_t _clc_valid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (_clc_valid_1 == 0) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int local_row = epi_warp * 32 + lane;
            int heads_per_destination = NUM_HEADS / P;
            unsigned int this_bid_1 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_cluster_tiles; _tile_iter_2++) {
                int group_1 = this_bid_1 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * GROUP_M;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                int local_1 = this_bid_1 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * BLOCK_M;
                int global_row = off_m_1 + local_row;
                int rope_row = 0;
                if (global_row < M) {
                    long long position = rope_positions[global_row];
                    if (position >= (long long)rope_rows) {
                        position = (long long)rope_rows - 1;
                    }
                    if (position < 0) {
                        position = 0;
                    }
                    rope_row = (int)position;
                }
                unsigned long long rope_row_base = (unsigned long long)rope_row * (unsigned long long)ROPE_DIM;
                int kind = bid_n_1 * 2 / NUM_HEADS;
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * (unsigned int)BLOCK_N;
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    int subgroup = bid_n_1 * 2 + half;
                    int head = subgroup % NUM_HEADS;
                    int destination = head / heads_per_destination;
                    int local_head = head % heads_per_destination;
                    unsigned long long out_base = ((((unsigned long long)destination * (unsigned long long)M + (unsigned long long)global_row) * (unsigned long long)heads_per_destination + (unsigned long long)local_head) * (unsigned long long)QKV_KINDS + (unsigned long long)kind) * (unsigned long long)HEAD_DIM;
                    int tmem_base = lane_addr + half * B_HALF_N;
                    if (kind < 2) {
                        float sum_partials[8];
                        #pragma unroll
                        for (int chunk = 0; chunk < HEAD_DIM / 16; chunk++) {
                            float _tmem_load_0[8];
                            tmem_ld_x8(&_tmem_load_0[0], tmem_base + chunk * 8);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            float lower_sum_lo = 0.0f;
                            float lower_sum_hi = 0.0f;
                            #pragma unroll
                            for (int j = 0; j < 4; j++) {
                                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_tmem_load_0[j]);
                                float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                                float rounded = _cvt_f32_0;
                                lower_sum_lo += rounded * rounded;
                                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_tmem_load_0[j + 4]);
                                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                                float rounded_hi = _cvt_f32_1;
                                lower_sum_hi += rounded_hi * rounded_hi;
                            }
                            int upper_chunk = chunk + HEAD_DIM / 16;
                            float _tmem_load_1[8];
                            tmem_ld_x8(&_tmem_load_1[0], tmem_base + upper_chunk * 8);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            float upper_sum_lo = 0.0f;
                            float upper_sum_hi = 0.0f;
                            #pragma unroll
                            for (int j_1 = 0; j_1 < 4; j_1++) {
                                __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(_tmem_load_1[j_1]);
                                float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                                float rounded_1 = _cvt_f32_2;
                                upper_sum_lo += rounded_1 * rounded_1;
                                __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(_tmem_load_1[j_1 + 4]);
                                float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                                float rounded_hi_1 = _cvt_f32_3;
                                upper_sum_hi += rounded_hi_1 * rounded_hi_1;
                            }
                            if (chunk < 4) {
                                sum_partials[chunk * 2] = lower_sum_lo + upper_sum_lo;
                                sum_partials[chunk * 2 + 1] = lower_sum_hi + upper_sum_hi;
                            } else {
                                sum_partials[(chunk - 4) * 2] = sum_partials[(chunk - 4) * 2] + (lower_sum_lo + upper_sum_lo);
                                sum_partials[(chunk - 4) * 2 + 1] = sum_partials[(chunk - 4) * 2 + 1] + (lower_sum_hi + upper_sum_hi);
                            }
                        }
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            sum_partials[i] = sum_partials[i] + sum_partials[i + 4];
                        }
                        #pragma unroll
                        for (int i_1 = 0; i_1 < 2; i_1++) {
                            sum_partials[i_1] = sum_partials[i_1] + sum_partials[i_1 + 2];
                        }
                        float sum_sq = sum_partials[0] + sum_partials[1];
                        float _rsqrt_0 = rsqrtf(sum_sq / (float)HEAD_DIM + qk_eps);
                        float rstd = _rsqrt_0;
                        #pragma unroll
                        for (int chunk_1 = 0; chunk_1 < ROPE_HALF / 8; chunk_1++) {
                            int col_lo = chunk_1 * 8;
                            int col_hi = col_lo + ROPE_HALF;
                            float _tmem_load_2[8];
                            tmem_ld_x8(&_tmem_load_2[0], tmem_base + col_lo);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            float _tmem_load_3[8];
                            tmem_ld_x8(&_tmem_load_3[0], tmem_base + col_hi);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            float _vec_load_0[8];
                            {
                                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(((kind == 0) ? q_norm_weight + col_lo : k_norm_weight + col_lo) + 0);
                                uint4 _vld_0[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_0[_blk] = _vptr_0[_blk];
                                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        asm volatile(
                                            "{\n\t"
                                            "shl.b32 %0, %2, 16;\n\t"
                                            "and.b32 %1, %2, 0xffff0000;\n\t"
                                            "}\n"
                                            : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                            : "r"(_vpairs_0[_pair]));
                                    }
                                }
                            }
                            float _vec_load_1[8];
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(((kind == 0) ? q_norm_weight + col_hi : k_norm_weight + col_hi) + 0);
                                uint4 _vld_1[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_1[_blk] = _vptr_1[_blk];
                                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        asm volatile(
                                            "{\n\t"
                                            "shl.b32 %0, %2, 16;\n\t"
                                            "and.b32 %1, %2, 0xffff0000;\n\t"
                                            "}\n"
                                            : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                            : "r"(_vpairs_1[_pair]));
                                    }
                                }
                            }
                            float _vec_load_2[8];
                            {
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(rope_cos_sin + (rope_row_base + (unsigned long long)col_lo) + 0);
                                uint4 _vld_2[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_2[_blk] = _vptr_2[_blk];
                                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        asm volatile(
                                            "{\n\t"
                                            "shl.b32 %0, %2, 16;\n\t"
                                            "and.b32 %1, %2, 0xffff0000;\n\t"
                                            "}\n"
                                            : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                            : "r"(_vpairs_2[_pair]));
                                    }
                                }
                            }
                            float _vec_load_3[8];
                            {
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(rope_cos_sin + (rope_row_base + (unsigned long long)ROPE_HALF + (unsigned long long)col_lo) + 0);
                                uint4 _vld_3[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_3[_blk] = _vptr_3[_blk];
                                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        asm volatile(
                                            "{\n\t"
                                            "shl.b32 %0, %2, 16;\n\t"
                                            "and.b32 %1, %2, 0xffff0000;\n\t"
                                            "}\n"
                                            : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                            : "r"(_vpairs_3[_pair]));
                                    }
                                }
                            }
                            float rotated_lo[8];
                            float rotated_hi[8];
                            #pragma unroll
                            for (int j_2 = 0; j_2 < 8; j_2++) {
                                __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(_tmem_load_2[j_2]);
                                float _cvt_f32_4 = __bfloat162float(_cvt_bf16_4);
                                float rounded_lo = _cvt_f32_4;
                                __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(_tmem_load_3[j_2]);
                                float _cvt_f32_5 = __bfloat162float(_cvt_bf16_5);
                                float rounded_hi_2 = _cvt_f32_5;
                                float scaled_lo = rstd * rounded_lo;
                                float scaled_hi = rstd * rounded_hi_2;
                                __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(_vec_load_0[j_2] * scaled_lo);
                                float _cvt_f32_6 = __bfloat162float(_cvt_bf16_6);
                                float norm_lo = _cvt_f32_6;
                                __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(_vec_load_1[j_2] * scaled_hi);
                                float _cvt_f32_7 = __bfloat162float(_cvt_bf16_7);
                                float norm_hi = _cvt_f32_7;
                                float cos_lo = norm_lo * _vec_load_2[j_2];
                                float cos_hi = norm_hi * _vec_load_2[j_2];
                                float sin_lo = norm_lo * _vec_load_3[j_2];
                                float sin_hi = norm_hi * _vec_load_3[j_2];
                                rotated_lo[j_2] = cos_lo - sin_hi;
                                rotated_hi[j_2] = cos_hi + sin_lo;
                            }
                            if (global_row < M) {
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(rotated_lo[0 + 0], rotated_lo[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(rotated_lo[0 + 2], rotated_lo[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(rotated_lo[0 + 4], rotated_lo[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(rotated_lo[0 + 6], rotated_lo[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (out_base + (unsigned long long)col_lo)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(rotated_hi[0 + 0], rotated_hi[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(rotated_hi[0 + 2], rotated_hi[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(rotated_hi[0 + 4], rotated_hi[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(rotated_hi[0 + 6], rotated_hi[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (out_base + (unsigned long long)col_hi)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                        #pragma unroll
                        for (int chunk_2 = ROPE_DIM / 8; chunk_2 < HEAD_DIM / 8; chunk_2++) {
                            int col = chunk_2 * 8;
                            float _tmem_load_4[8];
                            tmem_ld_x8(&_tmem_load_4[0], tmem_base + col);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            float _vec_load_4[8];
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(((kind == 0) ? q_norm_weight + col : k_norm_weight + col) + 0);
                                uint4 _vld_4[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_4[_blk] = _vptr_4[_blk];
                                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        asm volatile(
                                            "{\n\t"
                                            "shl.b32 %0, %2, 16;\n\t"
                                            "and.b32 %1, %2, 0xffff0000;\n\t"
                                            "}\n"
                                            : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                                            : "r"(_vpairs_4[_pair]));
                                    }
                                }
                            }
                            float normalized[8];
                            #pragma unroll
                            for (int j_3 = 0; j_3 < 8; j_3++) {
                                __nv_bfloat16 _cvt_bf16_8 = __float2bfloat16(_tmem_load_4[j_3]);
                                float _cvt_f32_8 = __bfloat162float(_cvt_bf16_8);
                                float rounded_2 = _cvt_f32_8;
                                float scaled = rstd * rounded_2;
                                normalized[j_3] = _vec_load_4[j_3] * scaled;
                            }
                            if (global_row < M) {
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(normalized[0 + 0], normalized[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(normalized[0 + 2], normalized[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(normalized[0 + 4], normalized[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(normalized[0 + 6], normalized[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (out_base + (unsigned long long)col)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int chunk_3 = 0; chunk_3 < HEAD_DIM / 8; chunk_3++) {
                            int col_1 = chunk_3 * 8;
                            float _tmem_load_5[8];
                            tmem_ld_x8(&_tmem_load_5[0], tmem_base + col_1);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            if (global_row < M) {
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_tmem_load_5[0 + 0], _tmem_load_5[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_tmem_load_5[0 + 2], _tmem_load_5[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_tmem_load_5[0 + 4], _tmem_load_5[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_tmem_load_5[0 + 6], _tmem_load_5[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (out_base + (unsigned long long)col_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                    }
                }
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                uint32_t _clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                this_bid_1 = _clc_ctaid_2 + (unsigned int)cta_rank;
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"

#undef BLOCK_K
#undef BLOCK_M
#undef BLOCK_N
#undef B_HALF_N
#undef CTA_GROUP
#undef GROUP_M
#undef HEAD_DIM
#undef MINIMAX_H3_PRE_ATTENTION_INF
#undef MMA_K
#undef NUM_EPILOGUE_WARPS
#undef NUM_HEADS
#undef NUM_K_ITERS
#undef NUM_MAINLOOP_PIPE_STAGES
#undef NUM_STAGES
#undef NUM_TMA_PIPE_STAGES
#undef NUM_WORK_PIPE_STAGES
#undef N_TILES
#undef QKV_KINDS
#undef ROPE_DIM
#undef ROPE_HALF
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef SMEM_WORK_RESPONSE_OFF
#undef SMEM_WORK_RESPONSE_STAGE_BYTES
#undef SMEM_WORK_RESPONSE_STRIDE
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef WORK_CONSUMERS
#undef WORK_STAGES
#undef epilogue_done_addr
#undef mainloop_done_addr
#undef mma_done_addr
#undef num_cluster_tiles
#undef smem_a_addr
#undef smem_b_addr
#undef tiles_per_group
#undef tma_full_addr
#undef work_empty_addr
#undef work_full_addr
#undef work_response_addr

#include <cuda_runtime.h>

#include <limits>
#include <mutex>
#include <vector>

#include "tvm_ffi_utils.h"

namespace {

constexpr int64_t kHidden = 5376;
constexpr int64_t kNumHeads = 56;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kQkvKinds = 3;
constexpr int64_t kQkvWidth = kNumHeads * kQkvKinds * kHeadDim;
constexpr int64_t kRopeDim = 96;
constexpr int64_t kMaxRows = 16777216;
constexpr int64_t kMaxTableRows = std::numeric_limits<int>::max();  // adaln_rows is an int kernel argument
constexpr int64_t kMaxRopeRows = std::numeric_limits<int>::max();   // rope_rows is an int kernel argument
constexpr int64_t kTableRowAlign = 8;  // elements: 16-byte rows for the 128-bit table loads
constexpr int64_t kBlockM = 128;
constexpr int64_t kCtaGroup = 2;
constexpr int64_t kNTiles = 84;  // 256 output columns (two native 128-column subgroups) per tile pair

// Kernel 1: four warps per row, kNormRowsPerCta rows per CTA.
constexpr int kNormThreads = 512;
constexpr int kNormRowsPerCta = 4;
constexpr int kNormSmem = 128;

// Kernel 2: persistent cta_group::2 GEMM, one 128-row x 128-column tile per CTA.
constexpr int kGemmThreads = 192;
constexpr int kGemmSmem = 230528;
constexpr unsigned int kClusterX = 2u;
constexpr unsigned int kClusterY = 1u;
constexpr unsigned int kClusterZ = 1u;

// TMA boxes (resolved from the kernel descriptors at export time): one kBoxK-element K group of
// kBoxRows rows per load, 128-byte swizzle.
constexpr uint32_t kBoxK = 64;
constexpr uint32_t kBoxRowsA = 128;
constexpr uint32_t kBoxRowsB = 128;
constexpr uint32_t kBoxGroups = 1;

int64_t MTiles(int64_t rows) {
  // Row tiles are consumed in CTA pairs: round the tile count up to an even number.
  int64_t tiles = (rows + kBlockM - 1) / kBlockM;
  return tiles + tiles % kCtaGroup;
}

// One cluster (CTA pair) per output tile pair; the hardware launches as many clusters as fit and
// running clusters claim the remaining tiles in order through cluster launch control.
int64_t GemmGrid(int64_t m_tiles) { return (m_tiles / kCtaGroup) * kNTiles * kCtaGroup; }

void CheckDeviceTensor(const TensorView& tensor, const char* name, DLDevice device, int64_t alignment) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError) << name << " must be a CUDA tensor";
  TVM_FFI_CHECK(tensor.device().device_id == device.device_id, ValueError)
      << name << " must be on the same CUDA device as x";
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0, ValueError)
      << name << " must be " << alignment << "-byte aligned";
}

void CheckDtype(const TensorView& tensor, const char* name, DLDataType dtype, const char* dtype_name) {
  TVM_FFI_CHECK(encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dtype), ValueError)
      << name << " must be " << dtype_name;
}

// Contiguous tensor of an exact shape (row-major unit strides, innermost first).
void CheckContiguous(const TensorView& tensor, const char* name, std::initializer_list<int64_t> shape,
                     DLDataType dtype, const char* dtype_name, DLDevice device, int64_t alignment) {
  CheckDeviceTensor(tensor, name, device, alignment);
  CheckDtype(tensor, name, dtype, dtype_name);
  TVM_FFI_CHECK(tensor.ndim() == static_cast<int64_t>(shape.size()), ValueError)
      << name << " must have rank " << shape.size();
  int64_t expected_stride = 1;
  int64_t dim = tensor.ndim();
  for (auto it = shape.end(); it != shape.begin();) {
    --it;
    --dim;
    TVM_FFI_CHECK(tensor.size(dim) == *it, ValueError)
        << name << " dimension " << dim << " must be " << *it << ", got " << tensor.size(dim);
    TVM_FFI_CHECK(tensor.stride(dim) == expected_stride, ValueError) << name << " must be contiguous";
    expected_stride *= *it;
  }
}

// Contiguous int64 [M] row index (AdaLN table rows, RoPE cache positions).  Any value is accepted:
// AdaLN indices outside [0, rows) select the zero row on the device, RoPE positions are clamped.
void CheckIndex(const TensorView& tensor, const char* name, int64_t rows, DLDevice device) {
  CheckContiguous(tensor, name, {rows}, dl_int64, "int64", device, 8);
}

// AdaLN tables: bf16 [rows, kHidden] with unit column stride, 1 <= rows <= kMaxTableRows, one common
// row stride that is a multiple of kTableRowAlign elements and 16-byte-aligned bases.  A contiguous
// [rows, 5376] table and a column chunk of a [rows, 6 * 5376] projection both qualify; the kernel
// reads table row r at base + r * row_stride.
struct AdalnTable {
  int64_t rows;
  int64_t row_stride;  // elements between consecutive rows
};

void CheckAdalnTable(const TensorView& tensor, const char* name, DLDevice device) {
  CheckDeviceTensor(tensor, name, device, 16);
  CheckDtype(tensor, name, dl_bfloat16, "bfloat16");
  TVM_FFI_CHECK(tensor.ndim() == 2 && tensor.size(1) == kHidden, ValueError)
      << name << " must have shape [rows, " << kHidden << "]";
  TVM_FFI_CHECK(tensor.size(0) >= 1 && tensor.size(0) <= kMaxTableRows, ValueError)
      << name << " must have 1 <= rows <= " << kMaxTableRows;
  TVM_FFI_CHECK(tensor.stride(1) == 1, ValueError) << name << " must have unit column stride";
  TVM_FFI_CHECK(tensor.stride(0) % kTableRowAlign == 0, ValueError)
      << name << " row stride must be a multiple of " << kTableRowAlign << " elements (16-byte rows)";
}

AdalnTable CheckAdalnTables(const TensorView& adaln_scale, const TensorView& adaln_shift, DLDevice device) {
  CheckAdalnTable(adaln_scale, "adaln_scale", device);
  CheckAdalnTable(adaln_shift, "adaln_shift", device);
  TVM_FFI_CHECK(adaln_shift.size(0) == adaln_scale.size(0) && adaln_shift.stride(0) == adaln_scale.stride(0), ValueError)
      << "adaln_scale and adaln_shift must have the same rows and the same row stride";
  return AdalnTable{adaln_scale.size(0), adaln_scale.stride(0)};
}

// RoPE cache: contiguous bf16 [S, kRopeDim] (48 cos then 48 sin), 1 <= S <= kMaxRopeRows; returns S.
int64_t CheckRopeCache(const TensorView& rope_cos_sin, DLDevice device) {
  CheckDeviceTensor(rope_cos_sin, "rope_cos_sin", device, 16);
  CheckDtype(rope_cos_sin, "rope_cos_sin", dl_bfloat16, "bfloat16");
  TVM_FFI_CHECK(rope_cos_sin.ndim() == 2 && rope_cos_sin.size(1) == kRopeDim, ValueError)
      << "rope_cos_sin must have shape [S, " << kRopeDim << "]";
  TVM_FFI_CHECK(rope_cos_sin.size(0) >= 1 && rope_cos_sin.size(0) <= kMaxRopeRows, ValueError)
      << "rope_cos_sin must have 1 <= S <= " << kMaxRopeRows;
  TVM_FFI_CHECK(rope_cos_sin.stride(1) == 1 && rope_cos_sin.stride(0) == kRopeDim, ValueError)
      << "rope_cos_sin must be contiguous";
  return rope_cos_sin.size(0);
}

// K-major [rows, cols] bf16 operand viewed as the rank-3 tensor (box_k, rows, cols / box_k):
// coordinate (0, row, k / box_k) addresses one box_k-wide K group of one row.  128-byte swizzle;
// activation rows beyond M are zero-filled by TMA (the box may extend past M) and never stored;
// every qkv_weight coordinate is in bounds (kQkvWidth and kHidden are multiples of the box).
CUtensorMap EncodeKMajorRows(const void* base, int64_t rows, int64_t cols, uint32_t box_rows, const char* name) {
  TVM_FFI_CHECK(cols % kBoxK == 0, RuntimeError) << name << ": K=" << cols << " is not a multiple of the TMA box";
  uint64_t global_dim[3] = {kBoxK, static_cast<uint64_t>(rows), static_cast<uint64_t>(cols / kBoxK)};
  uint64_t global_strides[2] = {static_cast<uint64_t>(cols * 2), static_cast<uint64_t>(kBoxK * 2)};
  uint32_t box_dim[3] = {kBoxK, box_rows, kBoxGroups};
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap descriptor{};
  CUresult result = cuTensorMapEncodeTiled(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(base), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "failed to encode the " << name << " tensor map: CUresult=" << static_cast<int>(result);
  return descriptor;
}

template <typename Kernel>
void OptInSmem(Kernel kernel, int bytes, const char* name) {
  if (bytes <= 0) return;
  const cudaError_t status = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to opt in to " << bytes << " bytes of dynamic shared memory (" << name << "): "
      << cudaGetErrorString(status);
}

// Per-device one-time configuration: capability check and dynamic shared memory opt-in.
void ConfigureKernels() {
  static std::mutex mutex;
  static std::vector<int> configured_devices;
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to get the active CUDA device: " << cudaGetErrorString(status);
  std::lock_guard<std::mutex> lock(mutex);
  for (int configured : configured_devices) {
    if (configured == device) return;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to query CUDA device properties: " << cudaGetErrorString(status);
  TVM_FFI_CHECK(properties.major == 10 && (properties.minor == 0 || properties.minor == 3), RuntimeError)
      << "MiniMax-H3 BF16 pre-attention requires compute capability 10.0 or 10.3 (tcgen05 + TMEM + 2-CTA MMA)";
  TVM_FFI_CHECK(properties.multiProcessorCount >= kCtaGroup, RuntimeError)
      << "MiniMax-H3 BF16 pre-attention requires at least " << kCtaGroup << " SMs";
  OptInSmem(kernel_minimax_h3_pre_attention_norm_adaln_bf16, kNormSmem, "norm");
  OptInSmem(kernel_minimax_h3_bf16_pre_attention_gemm, kGemmSmem, "gemm");
  configured_devices.push_back(device);
}

void CheckLaunch(const char* what) {
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << what << " launch failed: " << cudaGetErrorString(status);
}

template <typename Kernel, typename... Args>
void LaunchCluster(Kernel kernel, int64_t grid, int threads, int smem_bytes, cudaStream_t stream, const char* what, Args... args) {
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = kClusterX;
  attrs[0].val.clusterDim.y = kClusterY;
  attrs[0].val.clusterDim.z = kClusterZ;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned int>(grid), 1, 1);
  config.blockDim = dim3(static_cast<unsigned int>(threads), 1, 1);
  config.dynamicSmemBytes = static_cast<size_t>(smem_bytes);
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = 1;
  const cudaError_t status = cudaLaunchKernelEx(&config, kernel, args...);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << what << " launch failed: " << cudaGetErrorString(status);
}

}  // namespace

// x: bf16 [M, 5376]; x_norm_weight: bf16 [5376]; adaln_scale / adaln_shift: bf16 [rows, 5376] tables
// (see CheckAdalnTable); adaln_index: int64 [M], values outside [0, rows) select the zero row;
// qkv_weight: bf16 [21504, 5376], engine-resident row order [qkv_kind, head, head_dim] ([q_all | k_all | v_all]);
// q_norm_weight / k_norm_weight: bf16 [128]; rope_cos_sin: bf16 [S, 96] cache; rope_positions: int64 [M],
// clamped to [0, S) on the device; workspace: caller-owned bf16 [M, 5376] scratch that receives the
// modulated activation a; out: bf16 [ulysses_degree, M, 56 / ulysses_degree, 3, 128] destination-major
// pack; m == x.size(0); ulysses_degree in {1, 2, 4, 8}; eps: input RMSNorm epsilon; qk_eps: per-head
// Q/K RMSNorm epsilon.
void minimax_h3_bf16_pre_attention(TensorView x, TensorView x_norm_weight, TensorView adaln_scale,
                                   TensorView adaln_shift, TensorView adaln_index, TensorView qkv_weight,
                                   TensorView q_norm_weight, TensorView k_norm_weight, TensorView rope_cos_sin,
                                   TensorView rope_positions, TensorView workspace, TensorView out, int64_t m,
                                   int64_t ulysses_degree, double eps, double qk_eps) {
  TVM_FFI_CHECK(m >= 1 && m <= kMaxRows, ValueError) << "M must satisfy 1 <= M <= " << kMaxRows;
  TVM_FFI_CHECK(x.ndim() == 2 && x.size(0) == m, ValueError) << "M must equal x.size(0)";
  TVM_FFI_CHECK(ulysses_degree == 1 || ulysses_degree == 2 || ulysses_degree == 4 || ulysses_degree == 8, ValueError)
      << "ulysses_degree must be one of 1, 2, 4, or 8";

  const DLDevice device = x.device();
  CheckContiguous(x, "x", {m, kHidden}, dl_bfloat16, "bfloat16", device, 16);
  CheckContiguous(x_norm_weight, "x_norm_weight", {kHidden}, dl_bfloat16, "bfloat16", device, 16);
  const AdalnTable table = CheckAdalnTables(adaln_scale, adaln_shift, device);
  CheckIndex(adaln_index, "adaln_index", m, device);
  CheckContiguous(qkv_weight, "qkv_weight", {kQkvWidth, kHidden}, dl_bfloat16, "bfloat16", device, 16);
  CheckContiguous(q_norm_weight, "q_norm_weight", {kHeadDim}, dl_bfloat16, "bfloat16", device, 16);
  CheckContiguous(k_norm_weight, "k_norm_weight", {kHeadDim}, dl_bfloat16, "bfloat16", device, 16);
  const int64_t rope_rows = CheckRopeCache(rope_cos_sin, device);
  CheckIndex(rope_positions, "rope_positions", m, device);
  CheckContiguous(workspace, "workspace", {m, kHidden}, dl_bfloat16, "bfloat16", device, 16);
  CheckContiguous(out, "out", {ulysses_degree, m, kNumHeads / ulysses_degree, kQkvKinds, kHeadDim}, dl_bfloat16,
                  "bfloat16", device, 16);

  ffi::CUDADeviceGuard device_guard(device.device_id);
  const cudaStream_t stream = get_stream(device);
  ConfigureKernels();

  const int64_t norm_grid = (m + kNormRowsPerCta - 1) / kNormRowsPerCta;
  kernel_minimax_h3_pre_attention_norm_adaln_bf16<<<static_cast<unsigned int>(norm_grid), kNormThreads, kNormSmem, stream>>>(
      static_cast<__nv_bfloat16*>(x.data_ptr()), static_cast<__nv_bfloat16*>(x_norm_weight.data_ptr()),
      static_cast<__nv_bfloat16*>(adaln_scale.data_ptr()), static_cast<__nv_bfloat16*>(adaln_shift.data_ptr()),
      static_cast<long long*>(adaln_index.data_ptr()), static_cast<__nv_bfloat16*>(workspace.data_ptr()),
      static_cast<int>(m), static_cast<int>(table.rows), static_cast<long long>(table.row_stride),
      static_cast<float>(eps));
  CheckLaunch("MiniMax-H3 pre-attention norm+AdaLN");

  const int64_t m_tiles = MTiles(m);
  const CUtensorMap activation_map = EncodeKMajorRows(workspace.data_ptr(), m, kHidden, kBoxRowsA, "workspace");
  const CUtensorMap qkv_weight_map = EncodeKMajorRows(qkv_weight.data_ptr(), kQkvWidth, kHidden, kBoxRowsB, "qkv_weight");
  LaunchCluster(kernel_minimax_h3_bf16_pre_attention_gemm, GemmGrid(m_tiles), kGemmThreads, kGemmSmem, stream,
                "MiniMax-H3 pre-attention QKV GEMM", activation_map, qkv_weight_map,
                static_cast<__nv_bfloat16*>(q_norm_weight.data_ptr()),
                static_cast<__nv_bfloat16*>(k_norm_weight.data_ptr()),
                static_cast<__nv_bfloat16*>(rope_cos_sin.data_ptr()),
                static_cast<long long*>(rope_positions.data_ptr()), static_cast<__nv_bfloat16*>(out.data_ptr()),
                static_cast<int>(m), static_cast<int>(m_tiles), static_cast<int>(ulysses_degree),
                static_cast<int>(rope_rows), static_cast<float>(qk_eps));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_bf16_pre_attention, minimax_h3_bf16_pre_attention);
// clang-format on
