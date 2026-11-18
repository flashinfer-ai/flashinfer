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

#define LAUNCH_MIN_BLOCKS 1

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


__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)





__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    const unsigned int w = *reinterpret_cast<const unsigned int*>(&v);
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

extern "C" {

__global__ __launch_bounds__(800, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_sage_block_sparse_attention_9c29009b4dc0cd7fd54b(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, const __grid_constant__ CUtensorMap v, uint8_t* q_out, uint8_t* k_out, uint8_t* v_out, float* q_scale, float* k_scale, float* v_scale, int batch, int seqlen_q, int seqlen_k, int num_heads, int num_kv_heads, int k_scale_groups, int q_chunks, int k_chunks, int v_boxes)
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
    #define v_full_addr (mbar_base + 0)
    #define full_addr (mbar_base + 8)
    #define empty_addr (mbar_base + 72)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    __nv_bfloat16* tiles = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int tiles_addr = smem + 1024;
    __nv_bfloat16* vbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int vbuf_addr = smem + 132096;
    float* vmax_smem = reinterpret_cast<float*>(smem_raw + 164864);
    const int vmax_smem_addr = smem + 164864;
    float* part = reinterpret_cast<float*>(smem_raw + 165376);
    const int part_addr = smem + 165376;
    float* fin = reinterpret_cast<float*>(smem_raw + 165440);
    const int fin_addr = smem + 165440;

    // Mbarrier init (17 barriers at smem_raw[0..136))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // v_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'ring' ---
            // full: 8 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // empty: 8 barriers, init_count=128
            mbarrier_init(smem + 72, 128);
            mbarrier_init(smem + 80, 128);
            mbarrier_init(smem + 88, 128);
            mbarrier_init(smem + 96, 128);
            mbarrier_init(smem + 104, 128);
            mbarrier_init(smem + 112, 128);
            mbarrier_init(smem + 120, 128);
            mbarrier_init(smem + 128, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: vgrp ----
    if (warp <= 7) {
        { // vgrp_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int vs_cid = bid / 4;
            int vs_bh = vs_cid / 8;
            int hv_ctas = batch * num_kv_heads * 32;
            int kv_bh_v = vs_bh / num_kv_heads * num_kv_heads + vs_bh % num_kv_heads;
            int vcol = tid % 2;
            int vrow0 = tid / 2;
            asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
            if (((bid < hv_ctas) ? 1 : 0) != 0) {
                mbarrier_wait(v_full_addr, 0);
                unsigned int vmax[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    vmax[j] = 0;
                }
                #pragma unroll 1
                for (int i = 0; i < v_boxes * 2; i += 2) {
                    unsigned int vwa[8];
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&vwa[r * 4])), "=r"(*reinterpret_cast<uint32_t*>(&vwa[(r * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vwa[(r * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vwa[(r * 4) + 3]))
                            : "r"(vbuf_addr + (unsigned int)((vrow0 + (i + r) * 128) * 32 + vcol * 16)));
                    }
                    #pragma unroll
                    for (int r_1 = 0; r_1 < 2; r_1++) {
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 4; j_1++) {
                            uint32_t _bf16x2_abs_0;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(vwa[r_1 * 4 + j_1]));
                            uint32_t _bf16x2_max_0;
                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(vmax[j_1]), "r"(_bf16x2_abs_0));
                            vmax[j_1] = _bf16x2_max_0;
                        }
                    }
                }
                #pragma unroll
                for (int j_0 = 0; j_0 < 4; j_0++) {
                    unsigned int _shfl_xor_sync_0 = __shfl_xor_sync(0xFFFFFFFF, vmax[j_0], 2);
                    uint32_t _bf16x2_max_1_0;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1_0) : "r"(vmax[j_0]), "r"(_shfl_xor_sync_0));
                    vmax[j_0] = _bf16x2_max_1_0;
                }
                #pragma unroll
                for (int j_1_1 = 0; j_1_1 < 4; j_1_1++) {
                    unsigned int _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, vmax[j_1_1], 4);
                    uint32_t _bf16x2_max_1_1;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1_1) : "r"(vmax[j_1_1]), "r"(_shfl_xor_sync_1));
                    vmax[j_1_1] = _bf16x2_max_1_1;
                }
                #pragma unroll
                for (int j_2 = 0; j_2 < 4; j_2++) {
                    unsigned int _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, vmax[j_2], 8);
                    uint32_t _bf16x2_max_1_2;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1_2) : "r"(vmax[j_2]), "r"(_shfl_xor_sync_2));
                    vmax[j_2] = _bf16x2_max_1_2;
                }
                #pragma unroll
                for (int j_3 = 0; j_3 < 4; j_3++) {
                    unsigned int _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, vmax[j_3], 16);
                    uint32_t _bf16x2_max_1_3;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1_3) : "r"(vmax[j_3]), "r"(_shfl_xor_sync_3));
                    vmax[j_3] = _bf16x2_max_1_3;
                }
                if (lane < 2) {
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 4; j_4++) {
                        float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(vmax[j_4]));
                        vmax_smem[warp * 16 + vcol * 8 + 2 * j_4] = _cvt_f32_0.x;
                        vmax_smem[warp * 16 + vcol * 8 + 2 * j_4 + 1] = _cvt_f32_0.y;
                    }
                }
                asm volatile("barrier.sync 2, 256;" ::: "memory");
                if (tid < 16) {
                    float cta_max = vmax_smem[tid];
                    #pragma unroll
                    for (int other = 1; other < 8; other++) {
                        cta_max = fmaxf(cta_max, vmax_smem[other * 16 + tid]);
                    }
                    part[tid] = cta_max;
                }
                asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
            }
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            if (((bid < hv_ctas) ? 1 : 0) != 0) {
                if (tid < 16) {
                    float peer[4];
                    uint32_t _cluster_mapa_4;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_4) : "r"(part_addr + (unsigned int)(tid * 4)), "r"(0));
                    asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[0])) : "r"(_cluster_mapa_4));
                    uint32_t _cluster_mapa_5;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_5) : "r"(part_addr + (unsigned int)(tid * 4)), "r"(1));
                    asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[1])) : "r"(_cluster_mapa_5));
                    uint32_t _cluster_mapa_6;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_6) : "r"(part_addr + (unsigned int)(tid * 4)), "r"(2));
                    asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[2])) : "r"(_cluster_mapa_6));
                    uint32_t _cluster_mapa_7;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_7) : "r"(part_addr + (unsigned int)(tid * 4)), "r"(3));
                    asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[3])) : "r"(_cluster_mapa_7));
                    float cl_max = peer[0];
                    cl_max = fmaxf(cl_max, peer[1]);
                    cl_max = fmaxf(cl_max, peer[2]);
                    cl_max = fmaxf(cl_max, peer[3]);
                    fin[tid] = cl_max;
                }
                asm volatile("barrier.sync 2, 256;" ::: "memory");
            }
            asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
            if (((bid < hv_ctas) ? 1 : 0) != 0) {
                float amax_v[8];
                #pragma unroll
                for (int j_5 = 0; j_5 < 8; j_5++) {
                    amax_v[j_5] = fin[vcol * 8 + j_5];
                }
                float inv[8];
                #pragma unroll
                for (int j_6 = 0; j_6 < 8; j_6++) {
                    inv[j_6] = 448.0f / fmaxf(amax_v[j_6], 1e-06f);
                }
                if (cta_rank == 0 && tid < 16) {
                    v_scale[kv_bh_v * 128 + vs_cid % 8 * 16 + tid] = fmaxf(fin[tid], 1e-06f) / 448.0f;
                }
                #pragma unroll 1
                for (int i_1 = 0; i_1 < v_boxes * 2; i_1 += 2) {
                    unsigned int vwq[8];
                    #pragma unroll
                    for (int r_2 = 0; r_2 < 2; r_2++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&vwq[r_2 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&vwq[(r_2 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vwq[(r_2 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vwq[(r_2 * 4) + 3]))
                            : "r"(vbuf_addr + (unsigned int)((vrow0 + (i_1 + r_2) * 128) * 32 + vcol * 16)));
                    }
                    #pragma unroll
                    for (int r_3 = 0; r_3 < 2; r_3++) {
                        int vtoken = cta_rank * (v_boxes * 256) + vrow0 + (i_1 + r_3) * 128;
                        float values[8];
                        #pragma unroll
                        for (int j_7 = 0; j_7 < 4; j_7++) {
                            float2 _cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vwq[r_3 * 4 + j_7]));
                            values[2 * j_7] = _cvt_f32_5.x;
                            values[2 * j_7 + 1] = _cvt_f32_5.y;
                        }
                        #pragma unroll
                        for (int j_8 = 0; j_8 < 8; j_8++) {
                            values[j_8] = values[j_8] * inv[j_8];
                        }
                        if (vtoken < seqlen_k) {
                            int v_off = ((vs_bh / num_kv_heads * seqlen_k + vtoken) * num_kv_heads + vs_bh % num_kv_heads) * 128 + vs_cid % 8 * 16 + vcol * 8;
                            unsigned int _packed_2[2];
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(values[0]), "f"(values[1]),
                                                       "f"(values[2]), "f"(values[3]));
                                _packed_2[0] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(values[4]), "f"(values[5]),
                                                       "f"(values[6]), "f"(values[7]));
                                _packed_2[1] = _packed;
                            }
                            reinterpret_cast<int2*>(v_out + v_off)[0] = reinterpret_cast<int2*>(_packed_2)[0];
                        }
                    }
                }
            }
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }
    // ---- Role: ugrp ----
    if (warp >= 8 && warp <= 23) {
        { // ugrp_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
            int cu_q = batch * num_heads * q_chunks;
            int cu_all = cu_q + batch * num_kv_heads * k_chunks;
            int cu_grid = (int)gridDim.x;
            int cu_vctas = batch * num_kv_heads * 32;
            int cu_extra = cu_grid - cu_vctas;
            int cu_rest = cu_all - cu_extra * (v_boxes * 16) / 32;
            int cu_vunits = ((cu_rest > 0) ? cu_vctas * cu_rest / cu_grid : 0);
            int cu_extra1 = ((cu_extra) > (1) ? (cu_extra) : (1));
            int cu_e = bid - cu_vctas;
            int cu_lo = ((bid < cu_vctas) ? bid * cu_vunits / cu_vctas : cu_vunits + cu_e * (cu_all - cu_vunits) / cu_extra1);
            int cu_hi = ((bid < cu_vctas) ? (bid + 1) * cu_vunits / cu_vctas : cu_vunits + (cu_e + 1) * (cu_all - cu_vunits) / cu_extra1);
            int group = (warp - 8) / 4;
            int wq = warp % 4;
            int hrow = lane / 16;
            int c16 = lane % 16;
            int channel0 = c16 * 8;
            #pragma unroll 1
            for (int kk = group; kk < cu_hi - cu_lo; kk += 4) {
                int uc_q = ((cu_lo + kk < cu_q) ? 1 : 0);
                int uc_w = ((uc_q != 0) ? cu_lo + kk : cu_lo + kk - cu_q);
                int uc_heads = ((uc_q != 0) ? num_heads : num_kv_heads);
                int uc_chunks = ((uc_q != 0) ? q_chunks : k_chunks);
                int uc_bc = uc_w / uc_heads;
                int token0 = uc_bc % uc_chunks * 64 + wq * 16 + hrow;
                int _stage_full = (int)((unsigned int)kk % 8);
                int _parity_full = (int)((unsigned int)kk / 8 % 2);
                mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
                unsigned int slot_addr = tiles_addr + (unsigned int)kk % 8 * 16384;
                unsigned int xw[32];
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xw[p * 4])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(p * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(p * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(p * 4) + 3]))
                        : "r"(slot_addr + (unsigned int)((wq * 16 + hrow + p * 2) * 256 + c16 * 16)));
                }
                mbarrier_arrive(empty_addr + ((int)((unsigned int)kk % 8)) * 8);
                if (uc_q != 0) {
                    int bh = uc_bc / uc_chunks * num_heads + uc_w % uc_heads;
                    #pragma unroll
                    for (int p_1 = 0; p_1 < 8; p_1++) {
                        int token = token0 + p_1 * 2;
                        uint32_t _bf16x2_abs_1;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(xw[p_1 * 4]));
                        unsigned int packed = _bf16x2_abs_1;
                        #pragma unroll
                        for (int j_9 = 1; j_9 < 4; j_9++) {
                            uint32_t _bf16x2_abs_2;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(xw[p_1 * 4 + j_9]));
                            uint32_t _bf16x2_max_2;
                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(packed), "r"(_bf16x2_abs_2));
                            packed = _bf16x2_max_2;
                        }
                        float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(packed));
                        float amax = fmaxf(_cvt_f32_1.x, _cvt_f32_1.y);
                        float _shfl_xor_sync_0_1 = __shfl_xor_sync(0xFFFFFFFF, amax, 8);
                        amax = fmaxf(amax, _shfl_xor_sync_0_1);
                        float _shfl_xor_sync_1_1 = __shfl_xor_sync(0xFFFFFFFF, amax, 4);
                        amax = fmaxf(amax, _shfl_xor_sync_1_1);
                        float _shfl_xor_sync_2_1 = __shfl_xor_sync(0xFFFFFFFF, amax, 2);
                        amax = fmaxf(amax, _shfl_xor_sync_2_1);
                        float _shfl_xor_sync_3_1 = __shfl_xor_sync(0xFFFFFFFF, amax, 1);
                        amax = fmaxf(amax, _shfl_xor_sync_3_1);
                        amax = fmaxf(amax, 1e-06f);
                        float values_1[8];
                        #pragma unroll
                        for (int j_10 = 0; j_10 < 4; j_10++) {
                            float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[p_1 * 4 + j_10]));
                            values_1[2 * j_10] = _cvt_f32_2.x;
                            values_1[2 * j_10 + 1] = _cvt_f32_2.y;
                        }
                        const float2 _scale2_4 = {448.0f / amax, 448.0f / amax};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(values_1)[_ls], _scale2_4);
                        if (token < seqlen_q) {
                            int q_off = ((uc_bc / uc_chunks * seqlen_q + token) * num_heads + uc_w % uc_heads) * 128 + channel0;
                            unsigned int _packed_0[2];
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(values_1[0]), "f"(values_1[1]),
                                                       "f"(values_1[2]), "f"(values_1[3]));
                                _packed_0[0] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(values_1[4]), "f"(values_1[5]),
                                                       "f"(values_1[6]), "f"(values_1[7]));
                                _packed_0[1] = _packed;
                            }
                            reinterpret_cast<int2*>(q_out + q_off)[0] = reinterpret_cast<int2*>(_packed_0)[0];
                            if (c16 == 0) {
                                q_scale[bh * seqlen_q + token] = amax / 448.0f;
                            }
                        }
                    }
                } else {
                    int kv_bh = uc_bc / uc_chunks * num_kv_heads + uc_w % uc_heads;
                    int group_index = uc_bc % uc_chunks * 4 + wq;
                    unsigned int kmax = 0;
                    #pragma unroll
                    for (int p_2 = 0; p_2 < 8; p_2++) {
                        uint32_t _bf16x2_abs_3;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(xw[p_2 * 4]));
                        unsigned int kabs = _bf16x2_abs_3;
                        #pragma unroll
                        for (int j_11 = 1; j_11 < 4; j_11++) {
                            uint32_t _bf16x2_abs_4;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_4) : "r"(xw[p_2 * 4 + j_11]));
                            uint32_t _bf16x2_max_3;
                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_3) : "r"(kabs), "r"(_bf16x2_abs_4));
                            kabs = _bf16x2_max_3;
                        }
                        uint32_t _bf16x2_max_4;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_4) : "r"(kmax), "r"(kabs));
                        kmax = _bf16x2_max_4;
                    }
                    float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(kmax));
                    float group_max = fmaxf(_cvt_f32_3.x, _cvt_f32_3.y);
                    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, group_max, 16);
                    group_max = fmaxf(group_max, _shfl_xor_sync_5);
                    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, group_max, 8);
                    group_max = fmaxf(group_max, _shfl_xor_sync_6);
                    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, group_max, 4);
                    group_max = fmaxf(group_max, _shfl_xor_sync_7);
                    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, group_max, 2);
                    group_max = fmaxf(group_max, _shfl_xor_sync_8);
                    float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, group_max, 1);
                    group_max = fmaxf(group_max, _shfl_xor_sync_9);
                    group_max = fmaxf(group_max, 1e-06f);
                    #pragma unroll
                    for (int p_3 = 0; p_3 < 8; p_3++) {
                        int token_1 = token0 + p_3 * 2;
                        float values_2[8];
                        #pragma unroll
                        for (int j_12 = 0; j_12 < 4; j_12++) {
                            float2 _cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[p_3 * 4 + j_12]));
                            values_2[2 * j_12] = _cvt_f32_4.x;
                            values_2[2 * j_12 + 1] = _cvt_f32_4.y;
                        }
                        const float2 _scale2_10 = {448.0f / group_max, 448.0f / group_max};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(values_2)[_ls], _scale2_10);
                        if (token_1 < seqlen_k) {
                            int k_off = ((uc_bc / uc_chunks * seqlen_k + token_1) * num_kv_heads + uc_w % uc_heads) * 128 + channel0;
                            unsigned int _packed_1[2];
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(values_2[0]), "f"(values_2[1]),
                                                       "f"(values_2[2]), "f"(values_2[3]));
                                _packed_1[0] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(values_2[4]), "f"(values_2[5]),
                                                       "f"(values_2[6]), "f"(values_2[7]));
                                _packed_1[1] = _packed;
                            }
                            reinterpret_cast<int2*>(k_out + k_off)[0] = reinterpret_cast<int2*>(_packed_1)[0];
                        }
                    }
                    if (lane == 0 && group_index < k_scale_groups) {
                        k_scale[kv_bh * k_scale_groups + group_index] = group_max / 448.0f;
                    }
                }
            }
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }
    // ---- Role: prod ----
    if (warp == 24) {
        { // prod_main
            int cu_q_1 = batch * num_heads * q_chunks;
            int cu_all_1 = cu_q_1 + batch * num_kv_heads * k_chunks;
            int cu_grid_1 = (int)gridDim.x;
            int cu_vctas_1 = batch * num_kv_heads * 32;
            int cu_extra_1 = cu_grid_1 - cu_vctas_1;
            int cu_rest_1 = cu_all_1 - cu_extra_1 * (v_boxes * 16) / 32;
            int cu_vunits_1 = ((cu_rest_1 > 0) ? cu_vctas_1 * cu_rest_1 / cu_grid_1 : 0);
            int cu_extra1_1 = ((cu_extra_1) > (1) ? (cu_extra_1) : (1));
            int cu_e_1 = bid - cu_vctas_1;
            int cu_lo_1 = ((bid < cu_vctas_1) ? bid * cu_vunits_1 / cu_vctas_1 : cu_vunits_1 + cu_e_1 * (cu_all_1 - cu_vunits_1) / cu_extra1_1);
            int cu_hi_1 = ((bid < cu_vctas_1) ? (bid + 1) * cu_vunits_1 / cu_vctas_1 : cu_vunits_1 + (cu_e_1 + 1) * (cu_all_1 - cu_vunits_1) / cu_extra1_1);
            int vs_cid_1 = bid / 4;
            int vs_bh_1 = vs_cid_1 / 8;
            int hv_ctas_1 = batch * num_kv_heads * 32;
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&v))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&k))) : "memory");
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
            unsigned int fill = 0;
            int _min_0 = ((cu_hi_1 - cu_lo_1) < (4) ? (cu_hi_1 - cu_lo_1) : (4));
            int _min_1 = ((cu_hi_1 - cu_lo_1) < (8) ? (cu_hi_1 - cu_lo_1) : (8));
            int lead = ((((bid < hv_ctas_1) ? 1 : 0) != 0) ? _min_0 : _min_1);
            unsigned int _phase_empty = 1;
            if (lane == 0) {
                if (((bid < hv_ctas_1) ? 1 : 0) != 0) {
                    mbarrier_arrive_expect_tx(v_full_addr, v_boxes * 8192);
                    #pragma unroll 1
                    for (int j_13 = 0; j_13 < v_boxes; j_13++) {
                        tma_4d_gmem2smem(vbuf_addr + (unsigned int)(j_13 * 8192), (&v), vs_cid_1 % 8 * 16, vs_bh_1 % num_kv_heads, cta_rank * (v_boxes * 256) + j_13 * 256, vs_bh_1 / num_kv_heads, v_full_addr);
                    }
                }
                #pragma unroll 1
                for (int kk_1 = 0; kk_1 < lead; kk_1++) {
                    int uc_q_1 = ((cu_lo_1 + kk_1 < cu_q_1) ? 1 : 0);
                    int uc_w_1 = ((uc_q_1 != 0) ? cu_lo_1 + kk_1 : cu_lo_1 + kk_1 - cu_q_1);
                    int uc_heads_1 = ((uc_q_1 != 0) ? num_heads : num_kv_heads);
                    int uc_chunks_1 = ((uc_q_1 != 0) ? q_chunks : k_chunks);
                    int uc_bc_1 = uc_w_1 / uc_heads_1;
                    mbarrier_wait(empty_addr + (fill) * 8, _phase_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(full_addr + (fill) * 8, 16384);
                    if (uc_q_1 != 0) {
                        tma_4d_gmem2smem(tiles_addr + fill * 16384, (&q), 0, uc_w_1 % uc_heads_1, uc_bc_1 % uc_chunks_1 * 64, uc_bc_1 / uc_chunks_1, full_addr + (fill) * 8);
                    } else {
                        tma_4d_gmem2smem(tiles_addr + fill * 16384, (&k), 0, uc_w_1 % uc_heads_1, uc_bc_1 % uc_chunks_1 * 64, uc_bc_1 / uc_chunks_1, full_addr + (fill) * 8);
                    }
                    fill += 1;
                    if (fill == 8) { fill = 0; _phase_empty ^= 1; }
                }
            }
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
            if (lane == 0) {
                if (((bid < hv_ctas_1) ? 1 : 0) != 0) {
                    mbarrier_wait(v_full_addr, 0);
                }
                #pragma unroll 1
                for (int kk_2 = lead; kk_2 < cu_hi_1 - cu_lo_1; kk_2++) {
                    int uc_q_2 = ((cu_lo_1 + kk_2 < cu_q_1) ? 1 : 0);
                    int uc_w_2 = ((uc_q_2 != 0) ? cu_lo_1 + kk_2 : cu_lo_1 + kk_2 - cu_q_1);
                    int uc_heads_2 = ((uc_q_2 != 0) ? num_heads : num_kv_heads);
                    int uc_chunks_2 = ((uc_q_2 != 0) ? q_chunks : k_chunks);
                    int uc_bc_2 = uc_w_2 / uc_heads_2;
                    mbarrier_wait(empty_addr + (fill) * 8, _phase_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(full_addr + (fill) * 8, 16384);
                    if (uc_q_2 != 0) {
                        tma_4d_gmem2smem(tiles_addr + fill * 16384, (&q), 0, uc_w_2 % uc_heads_2, uc_bc_2 % uc_chunks_2 * 64, uc_bc_2 / uc_chunks_2, full_addr + (fill) * 8);
                    } else {
                        tma_4d_gmem2smem(tiles_addr + fill * 16384, (&k), 0, uc_w_2 % uc_heads_2, uc_bc_2 % uc_chunks_2 * 64, uc_bc_2 / uc_chunks_2, full_addr + (fill) * 8);
                    }
                    fill += 1;
                    if (fill == 8) { fill = 0; _phase_empty ^= 1; }
                }
            }
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
