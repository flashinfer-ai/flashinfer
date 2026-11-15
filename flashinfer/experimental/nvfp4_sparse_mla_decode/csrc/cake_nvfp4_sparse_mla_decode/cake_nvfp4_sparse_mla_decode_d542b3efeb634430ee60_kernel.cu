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
#define TMEM_NCOLS 488
#define TMEM_TMEM_KD_OFFSET 0
#define TMEM_TMEM_KR_OFFSET 128
#define TMEM_TMEM_S_OFFSET 160
#define TMEM_TMEM_O_OFFSET 192
#define TMEM_TMEM_ACC_OFFSET 0
#define TMEM_TMEM_VT_OFFSET 256
#define TMEM_TMEM_ID_OFFSET 384
#define TMEM_TMEM_SFAV_OFFSET 400
#define TMEM_TMEM_SFBK_OFFSET 404
#define TMEM_TMEM_SFBV_OFFSET 408
#define TMEM_TMEM_SFAK_OFFSET 424
#define NUM_NOPE_RING_STAGES 1
#define NUM_SFK_RING_STAGES 1
#define NUM_SFV_RING_STAGES 1
#define NUM_KR_RING_STAGES 1
#define NUM_KD_RING_STAGES 2
#define NUM_S_RING_STAGES 1
#define NUM_P_RING_STAGES 1
#define NUM_VT_RING_STAGES 2
#define SMEM_SMEM_NOPE_OFF 1024
#define SMEM_SMEM_NOPE_STAGE_BYTES 32768
#define SMEM_SMEM_NOPE_STRIDE 32768
#define SMEM_SMEM_NOPE_P_OFF 1024
#define SMEM_SMEM_NOPE_P_STAGE_BYTES 16384
#define SMEM_SMEM_NOPE_P_STRIDE 16384
#define SMEM_SMEM_QN_OFF 33792
#define SMEM_SMEM_QN_STAGE_BYTES 32768
#define SMEM_SMEM_QN_STRIDE 32768
#define SMEM_SMEM_QNK_OFF 33792
#define SMEM_SMEM_QNK_STAGE_BYTES 4096
#define SMEM_SMEM_QNK_STRIDE 4096
#define SMEM_SMEM_QR_OFF 66560
#define SMEM_SMEM_QR_STAGE_BYTES 2048
#define SMEM_SMEM_QR_STRIDE 2048
#define SMEM_SMEM_P_OFF 68608
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_SMEM_PH_OFF 68608
#define SMEM_SMEM_PH_STAGE_BYTES 4096
#define SMEM_SMEM_PH_STRIDE 4096
#define SMEM_SMEM_IDB_OFF 84992
#define SMEM_SMEM_IDB_STAGE_BYTES 2048
#define SMEM_SMEM_IDB_STRIDE 2048
#define SMEM_SMEM_SF_OFF 87040
#define SMEM_SMEM_SF_STAGE_BYTES 8192
#define SMEM_SMEM_SF_STRIDE 8192
#define SMEM_SMEM_SFH_OFF 87040
#define SMEM_SMEM_SFH_STAGE_BYTES 4096
#define SMEM_SMEM_SFH_STRIDE 4096
#define SMEM_XMAX_OFF 95232
#define SMEM_XMAX_STAGE_BYTES 512
#define SMEM_XMAX_STRIDE 512
#define SMEM_XSUM_OFF 95744
#define SMEM_XSUM_STAGE_BYTES 256
#define SMEM_XSUM_STRIDE 256
#define SMEM_ZONE_OFF 96128
#define SMEM_ZONE_STAGE_BYTES 33280
#define SMEM_ZONE_STRIDE 33280
#define SMEM_WTAB_OFF 95232
#define SMEM_WTAB_STAGE_BYTES 256
#define SMEM_WTAB_STRIDE 256
#define SMEM_INVT_OFF 96000
#define SMEM_INVT_STAGE_BYTES 64
#define SMEM_INVT_STRIDE 64
#define SMEM_TOTAL 129408
#define THREADS 512
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

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}




__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}



union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



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


__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}



__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo128(int addr) {
    const int SBO = 128;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}



__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_st_x8_u32(int addr, const uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(addr),
           "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
           "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]));
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(1,4,1) void
kernel_cake_nvfp4_sparse_mla_decode_d542b3efeb634430ee60(int* __restrict__ q, int* __restrict__ kv, const __grid_constant__ CUtensorMap kv_nope, int* __restrict__ indices, __nv_bfloat16* __restrict__ out, int topk, int keys_per_cta, float qk_scale, float out_scale)
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
    #define nope_full_addr (mbar_base + 0)
    #define sfk_full_addr (mbar_base + 8)
    #define sfv_full_addr (mbar_base + 16)
    #define kr_full_addr (mbar_base + 24)
    #define const_ready_addr (mbar_base + 32)
    #define q_full_addr (mbar_base + 40)
    #define kd_full_addr (mbar_base + 48)
    #define s_full_addr (mbar_base + 64)
    #define p_full_addr (mbar_base + 72)
    #define vt_full_addr (mbar_base + 80)
    #define tmem_free_addr (mbar_base + 96)
    #define zone_bar_addr (mbar_base + 104)
    #define ml_bar_addr (mbar_base + 112)
    #define oblk_addr (mbar_base + 120)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 1;
    const unsigned int cluster_id = ((blockIdx.z * (gridDim.y / 4) + (blockIdx.y / 4)) * clusters_x) + blockIdx.x / 1;
    const unsigned int num_clusters = clusters_x * (gridDim.y / 4) * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_nope = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_nope_addr = smem + 1024;
    uint8_t* smem_nope_p = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_nope_p_addr = smem + 1024;
    float* smem_qn = reinterpret_cast<float*>(smem_raw + 33792);
    const int smem_qn_addr = smem + 33792;
    float* smem_qnk = reinterpret_cast<float*>(smem_raw + 33792);
    const int smem_qnk_addr = smem + 33792;
    __half* smem_qr = reinterpret_cast<__half*>(smem_raw + 66560);
    const int smem_qr_addr = smem + 66560;
    float* smem_p = reinterpret_cast<float*>(smem_raw + 68608);
    const int smem_p_addr = smem + 68608;
    float* smem_ph = reinterpret_cast<float*>(smem_raw + 68608);
    const int smem_ph_addr = smem + 68608;
    uint8_t* smem_idb = reinterpret_cast<uint8_t*>(smem_raw + 84992);
    const int smem_idb_addr = smem + 84992;
    uint8_t* smem_sf = reinterpret_cast<uint8_t*>(smem_raw + 87040);
    const int smem_sf_addr = smem + 87040;
    uint8_t* smem_sfh = reinterpret_cast<uint8_t*>(smem_raw + 87040);
    const int smem_sfh_addr = smem + 87040;
    float* xmax = reinterpret_cast<float*>(smem_raw + 95232);
    const int xmax_addr = smem + 95232;
    float* xsum = reinterpret_cast<float*>(smem_raw + 95744);
    const int xsum_addr = smem + 95744;
    uint8_t* zone = reinterpret_cast<uint8_t*>(smem_raw + 96128);
    const int zone_addr = smem + 96128;
    float* wtab = reinterpret_cast<float*>(smem_raw + 95232);
    const int wtab_addr = smem + 95232;
    float* invt = reinterpret_cast<float*>(smem_raw + 96000);
    const int invt_addr = smem + 96000;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&kv_nope))) : "memory");
    int token = blockIdx.x;
    int cta = blockIdx.y;
    int key_lo = cta * keys_per_cta;
    int key_hi = ((key_lo + keys_per_cta) < (topk) ? (key_lo + keys_per_cta) : (topk));
    int n_keys = ((key_hi - key_lo) > (0) ? (key_hi - key_lo) : (0));

    // Mbarrier init (14 pipeline groups, 0 ordered-sequence groups, 19 barriers)
    // Mbarriers at smem_raw[0..152)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'nope_ring' ---
            // nope_full: 1 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            // --- pipeline 'sfk_ring' ---
            // sfk_full: 1 barriers, init_count=4
            mbarrier_init(smem + 8, 4);
            // --- pipeline 'sfv_ring' ---
            // sfv_full: 1 barriers, init_count=4
            mbarrier_init(smem + 16, 4);
            // --- pipeline 'kr_ring' ---
            // kr_full: 1 barriers, init_count=4
            mbarrier_init(smem + 24, 4);
            // const_ready: 1 barriers, init_count=4
            mbarrier_init(smem + 32, 4);
            // q_full: 1 barriers, init_count=4
            mbarrier_init(smem + 40, 4);
            // --- pipeline 'kd_ring' ---
            // kd_full: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 's_ring' ---
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // --- pipeline 'p_ring' ---
            // p_full: 1 barriers, init_count=4
            mbarrier_init(smem + 72, 4);
            // --- pipeline 'vt_ring' ---
            // vt_full: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // tmem_free: 1 barriers, init_count=4
            mbarrier_init(smem + 96, 4);
            // zone_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            // ml_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // oblk: 4 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // Kernel pre-TMEM ops
    int pw = warp;
    int pl = lane;
    unsigned long long kv_p = (unsigned long long)kv;
    if ((pw & 10) == 10) {
        int tcp = ((pw & 1) + (pw >> 2 & 1) * 2) * 32 + pl;
        int kc0 = tcp & 15;
        int rid[16];
        rid[0] = -1;
        if (n_keys > tcp >> 4) {
            int _vec_load_0[1];
            {
                uint32_t _scalar_bits_0;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_0) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4)))) : "memory");
                _vec_load_0[0] = (int32_t)_scalar_bits_0;
            }
            rid[0] = _vec_load_0[0];
        }
        rid[1] = -1;
        if (n_keys > (tcp >> 4) + 8) {
            int _vec_load_1[1];
            {
                uint32_t _scalar_bits_1;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_1) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4) + 8))) : "memory");
                _vec_load_1[0] = (int32_t)_scalar_bits_1;
            }
            rid[1] = _vec_load_1[0];
        }
        rid[2] = -1;
        if (n_keys > (tcp >> 4) + 16) {
            int _vec_load_2[1];
            {
                uint32_t _scalar_bits_2;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_2) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4) + 16))) : "memory");
                _vec_load_2[0] = (int32_t)_scalar_bits_2;
            }
            rid[2] = _vec_load_2[0];
        }
        rid[3] = -1;
        if (n_keys > (tcp >> 4) + 24) {
            int _vec_load_3[1];
            {
                uint32_t _scalar_bits_3;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_3) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4) + 24))) : "memory");
                _vec_load_3[0] = (int32_t)_scalar_bits_3;
            }
            rid[3] = _vec_load_3[0];
        }
        rid[4] = -1;
        if (n_keys > (tcp >> 4) + 32) {
            int _vec_load_4[1];
            {
                uint32_t _scalar_bits_4;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_4) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4) + 32))) : "memory");
                _vec_load_4[0] = (int32_t)_scalar_bits_4;
            }
            rid[4] = _vec_load_4[0];
        }
        rid[5] = -1;
        if (n_keys > (tcp >> 4) + 40) {
            int _vec_load_5[1];
            {
                uint32_t _scalar_bits_5;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_5) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4) + 40))) : "memory");
                _vec_load_5[0] = (int32_t)_scalar_bits_5;
            }
            rid[5] = _vec_load_5[0];
        }
        rid[6] = -1;
        if (n_keys > (tcp >> 4) + 48) {
            int _vec_load_6[1];
            {
                uint32_t _scalar_bits_6;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_6) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4) + 48))) : "memory");
                _vec_load_6[0] = (int32_t)_scalar_bits_6;
            }
            rid[6] = _vec_load_6[0];
        }
        rid[7] = -1;
        if (n_keys > (tcp >> 4) + 56) {
            int _vec_load_7[1];
            {
                uint32_t _scalar_bits_7;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_7) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4) + 56))) : "memory");
                _vec_load_7[0] = (int32_t)_scalar_bits_7;
            }
            rid[7] = _vec_load_7[0];
        }
        rid[8] = -1;
        if (n_keys > (tcp >> 4) + 64) {
            int _vec_load_8[1];
            {
                uint32_t _scalar_bits_8;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_8) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4) + 64))) : "memory");
                _vec_load_8[0] = (int32_t)_scalar_bits_8;
            }
            rid[8] = _vec_load_8[0];
        }
        rid[9] = -1;
        if (n_keys > (tcp >> 4) + 72) {
            int _vec_load_9[1];
            {
                uint32_t _scalar_bits_9;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_9) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4) + 72))) : "memory");
                _vec_load_9[0] = (int32_t)_scalar_bits_9;
            }
            rid[9] = _vec_load_9[0];
        }
        rid[10] = -1;
        if (n_keys > (tcp >> 4) + 80) {
            int _vec_load_10[1];
            {
                uint32_t _scalar_bits_10;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_10) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4) + 80))) : "memory");
                _vec_load_10[0] = (int32_t)_scalar_bits_10;
            }
            rid[10] = _vec_load_10[0];
        }
        rid[11] = -1;
        if (n_keys > (tcp >> 4) + 88) {
            int _vec_load_11[1];
            {
                uint32_t _scalar_bits_11;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_11) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4) + 88))) : "memory");
                _vec_load_11[0] = (int32_t)_scalar_bits_11;
            }
            rid[11] = _vec_load_11[0];
        }
        rid[12] = -1;
        if (n_keys > (tcp >> 4) + 96) {
            int _vec_load_12[1];
            {
                uint32_t _scalar_bits_12;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_12) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4) + 96))) : "memory");
                _vec_load_12[0] = (int32_t)_scalar_bits_12;
            }
            rid[12] = _vec_load_12[0];
        }
        rid[13] = -1;
        if (n_keys > (tcp >> 4) + 104) {
            int _vec_load_13[1];
            {
                uint32_t _scalar_bits_13;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_13) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4) + 104))) : "memory");
                _vec_load_13[0] = (int32_t)_scalar_bits_13;
            }
            rid[13] = _vec_load_13[0];
        }
        rid[14] = -1;
        if (n_keys > (tcp >> 4) + 112) {
            int _vec_load_14[1];
            {
                uint32_t _scalar_bits_14;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_14) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4) + 112))) : "memory");
                _vec_load_14[0] = (int32_t)_scalar_bits_14;
            }
            rid[14] = _vec_load_14[0];
        }
        rid[15] = -1;
        if (n_keys > (tcp >> 4) + 120) {
            int _vec_load_15[1];
            {
                uint32_t _scalar_bits_15;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_15) : "l"((const void*)(indices + (token * topk + key_lo + (tcp >> 4) + 120))) : "memory");
                _vec_load_15[0] = (int32_t)_scalar_bits_15;
            }
            rid[15] = _vec_load_15[0];
        }
        unsigned int nope0 = smem_nope_addr;
        int ok0 = ((rid[0] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + (tcp >> 4) * 128 + ((kc0 & 7 ^ tcp >> 4 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0 != 0) ? rid[0] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0 * 16)));
        int ok0_0 = ((rid[1] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + ((tcp >> 4) + 8) * 128 + ((kc0 & 7 ^ (tcp >> 4) + 8 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0_0 != 0) ? rid[1] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0_0 * 16)));
        int ok0_1 = ((rid[2] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + ((tcp >> 4) + 16) * 128 + ((kc0 & 7 ^ (tcp >> 4) + 16 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0_1 != 0) ? rid[2] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0_1 * 16)));
        int ok0_2 = ((rid[3] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + ((tcp >> 4) + 24) * 128 + ((kc0 & 7 ^ (tcp >> 4) + 24 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0_2 != 0) ? rid[3] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0_2 * 16)));
        int ok0_3 = ((rid[4] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + ((tcp >> 4) + 32) * 128 + ((kc0 & 7 ^ (tcp >> 4) + 32 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0_3 != 0) ? rid[4] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0_3 * 16)));
        int ok0_4 = ((rid[5] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + ((tcp >> 4) + 40) * 128 + ((kc0 & 7 ^ (tcp >> 4) + 40 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0_4 != 0) ? rid[5] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0_4 * 16)));
        int ok0_5 = ((rid[6] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + ((tcp >> 4) + 48) * 128 + ((kc0 & 7 ^ (tcp >> 4) + 48 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0_5 != 0) ? rid[6] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0_5 * 16)));
        int ok0_6 = ((rid[7] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + ((tcp >> 4) + 56) * 128 + ((kc0 & 7 ^ (tcp >> 4) + 56 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0_6 != 0) ? rid[7] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0_6 * 16)));
        int ok0_7 = ((rid[8] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + ((tcp >> 4) + 64) * 128 + ((kc0 & 7 ^ (tcp >> 4) + 64 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0_7 != 0) ? rid[8] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0_7 * 16)));
        int ok0_8 = ((rid[9] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + ((tcp >> 4) + 72) * 128 + ((kc0 & 7 ^ (tcp >> 4) + 72 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0_8 != 0) ? rid[9] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0_8 * 16)));
        int ok0_9 = ((rid[10] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + ((tcp >> 4) + 80) * 128 + ((kc0 & 7 ^ (tcp >> 4) + 80 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0_9 != 0) ? rid[10] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0_9 * 16)));
        int ok0_10 = ((rid[11] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + ((tcp >> 4) + 88) * 128 + ((kc0 & 7 ^ (tcp >> 4) + 88 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0_10 != 0) ? rid[11] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0_10 * 16)));
        int ok0_11 = ((rid[12] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + ((tcp >> 4) + 96) * 128 + ((kc0 & 7 ^ (tcp >> 4) + 96 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0_11 != 0) ? rid[12] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0_11 * 16)));
        int ok0_12 = ((rid[13] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + ((tcp >> 4) + 104) * 128 + ((kc0 & 7 ^ (tcp >> 4) + 104 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0_12 != 0) ? rid[13] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0_12 * 16)));
        int ok0_13 = ((rid[14] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + ((tcp >> 4) + 112) * 128 + ((kc0 & 7 ^ (tcp >> 4) + 112 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0_13 != 0) ? rid[14] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0_13 * 16)));
        int ok0_14 = ((rid[15] >= 0) ? 1 : 0);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(nope0 + (unsigned int)((kc0 >> 3) * 16384 + ((tcp >> 4) + 120) * 128 + ((kc0 & 7 ^ (tcp >> 4) + 120 & 7) << 4))), "l"(kv_p + (unsigned long long)(((ok0_14 != 0) ? rid[15] : 0)) * 352 + (unsigned long long)(kc0 * 16)), "r"((unsigned int)(ok0_14 * 16)));
        asm volatile(
            "{\n\t"
            "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
            "}"
            :: "r"(nope_full_addr) : "memory");
        asm volatile("barrier.sync 2, 128;" ::: "memory");
        if (tcp < 64) {
            if (elect_sync()) {
                mbarrier_arrive(nope_full_addr);
            }
        }
    }
    if (pw < 4) {
        int rowp = pw * 32 + pl;
        int tix[1];
        tix[0] = -1;
        if (rowp < n_keys) {
            int _vec_load_16[1];
            {
                uint32_t _scalar_bits_16;
                asm volatile("ld.global.nc.b32 %0, [%1];"
                    : "=r"(_scalar_bits_16) : "l"((const void*)(indices + (token * topk + key_lo + rowp))) : "memory");
                _vec_load_16[0] = (int32_t)_scalar_bits_16;
            }
            tix[0] = _vec_load_16[0];
        }
        int tw = tix[0] * 88 + 64;
        if (tix[0] >= 0) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(kv + tw))); }
        if (tix[0] >= 0) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(kv + (tw + 23)))); }
    }
    if (pw >> 2 == 1) {
        int tq = (pw & 3) * 32 + pl;
        unsigned long long qp = (unsigned long long)q + (unsigned long long)(token * 9216);
        unsigned int qs0 = smem_p_addr;
        int qo = ((tq >> 5) * 36 + (tq & 31)) * 16;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(qs0 + (unsigned int)qo), "l"(qp + (unsigned long long)qo));
        int qo_0 = ((tq + 128 >> 5) * 36 + (tq + 128 & 31)) * 16;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(qs0 + (unsigned int)qo_0), "l"(qp + (unsigned long long)qo_0));
        int qo_1 = ((tq + 256 >> 5) * 36 + (tq + 256 & 31)) * 16;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(qs0 + (unsigned int)qo_1), "l"(qp + (unsigned long long)qo_1));
        int qo_2 = ((tq + 384 >> 5) * 36 + (tq + 384 & 31)) * 16;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(qs0 + (unsigned int)qo_2), "l"(qp + (unsigned long long)qo_2));
        if (tq < 64) {
            int qro = ((tq >> 2) * 36 + 32 + (tq & 3)) * 16;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(qs0 + (unsigned int)qro), "l"(qp + (unsigned long long)qro));
        }
        asm volatile("cp.async.commit_group;");
    }

    // TMEM alloc (512 columns, 488 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 152);
    if (warp == 12) {
        int _tmem_hold = smem + 152;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_kd = taddr;
    const int tmem_tmem_kr = taddr + 128;
    const int tmem_tmem_s = taddr + 160;
    const int tmem_tmem_o = taddr + 192;
    const int tmem_tmem_acc = taddr;
    const int tmem_tmem_vt = taddr + 256;
    const int tmem_tmem_id = taddr + 384;
    const int tmem_tmem_sfav = taddr + 400;
    const int tmem_tmem_sfbk = taddr + 404;
    const int tmem_tmem_sfbv = taddr + 408;
    const int tmem_tmem_sfak = taddr + 424;
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
    }

    // ---- Role: ld ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 152;");
        { // ld_main
            int lane_l = lane;
            int wl = warp % 4;
            int row_l = wl * 32 + lane_l;
            unsigned int lane_addr_l = (unsigned int)(wl * 32) << 16;
            unsigned int sfw[8];
            unsigned int words[32];
            unsigned int cur[24];
            unsigned int nxt[24];
            int inx[1];
            unsigned int idw[8];
            idw[0] = 0;
            if (row_l >> 6 == 0) {
                if ((row_l & 63) >> 3 == 0) {
                    idw[0] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            idw[1] = 0;
            if (row_l >> 6 == 0) {
                if ((row_l & 63) >> 3 == 1) {
                    idw[1] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            idw[2] = 0;
            if (row_l >> 6 == 0) {
                if ((row_l & 63) >> 3 == 2) {
                    idw[2] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            idw[3] = 0;
            if (row_l >> 6 == 0) {
                if ((row_l & 63) >> 3 == 3) {
                    idw[3] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            idw[4] = 0;
            if (row_l >> 6 == 0) {
                if ((row_l & 63) >> 3 == 4) {
                    idw[4] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            idw[5] = 0;
            if (row_l >> 6 == 0) {
                if ((row_l & 63) >> 3 == 5) {
                    idw[5] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            idw[6] = 0;
            if (row_l >> 6 == 0) {
                if ((row_l & 63) >> 3 == 6) {
                    idw[6] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            idw[7] = 0;
            if (row_l >> 6 == 0) {
                if ((row_l & 63) >> 3 == 7) {
                    idw[7] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            tmem_st_x8_u32(taddr + 384 + lane_addr_l, (const uint32_t*)idw);
            idw[0] = 0;
            if (row_l >> 6 == 1) {
                if ((row_l & 63) >> 3 == 0) {
                    idw[0] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            idw[1] = 0;
            if (row_l >> 6 == 1) {
                if ((row_l & 63) >> 3 == 1) {
                    idw[1] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            idw[2] = 0;
            if (row_l >> 6 == 1) {
                if ((row_l & 63) >> 3 == 2) {
                    idw[2] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            idw[3] = 0;
            if (row_l >> 6 == 1) {
                if ((row_l & 63) >> 3 == 3) {
                    idw[3] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            idw[4] = 0;
            if (row_l >> 6 == 1) {
                if ((row_l & 63) >> 3 == 4) {
                    idw[4] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            idw[5] = 0;
            if (row_l >> 6 == 1) {
                if ((row_l & 63) >> 3 == 5) {
                    idw[5] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            idw[6] = 0;
            if (row_l >> 6 == 1) {
                if ((row_l & 63) >> 3 == 6) {
                    idw[6] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            idw[7] = 0;
            if (row_l >> 6 == 1) {
                if ((row_l & 63) >> 3 == 7) {
                    idw[7] = 2 << (unsigned int)(4 * (row_l & 7));
                }
            }
            tmem_st_x8_u32(taddr + 392 + lane_addr_l, (const uint32_t*)idw);
            unsigned int ones[4];
            ones[0] = 943208504;
            ones[1] = 943208504;
            ones[2] = 943208504;
            ones[3] = 943208504;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x4.b32"
                " [%0], {%1, %2, %3, %4};"
                :: "r"(taddr + 400 + lane_addr_l), "r"(ones[0]), "r"(ones[1]), "r"(ones[2]), "r"(ones[3]));
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x4.b32"
                " [%0], {%1, %2, %3, %4};"
                :: "r"(taddr + 404 + lane_addr_l), "r"(ones[0]), "r"(ones[1]), "r"(ones[2]), "r"(ones[3]));
            unsigned int idb[4];
            if (row_l < 64) {
                idb[0] = 0;
                if (row_l >> 5 == 0) {
                    if ((row_l & 31) >> 3 == 0) {
                        idb[0] = 2 << (unsigned int)(4 * (row_l & 7));
                    }
                }
                idb[1] = 0;
                if (row_l >> 5 == 0) {
                    if ((row_l & 31) >> 3 == 1) {
                        idb[1] = 2 << (unsigned int)(4 * (row_l & 7));
                    }
                }
                idb[2] = 0;
                if (row_l >> 5 == 0) {
                    if ((row_l & 31) >> 3 == 2) {
                        idb[2] = 2 << (unsigned int)(4 * (row_l & 7));
                    }
                }
                idb[3] = 0;
                if (row_l >> 5 == 0) {
                    if ((row_l & 31) >> 3 == 3) {
                        idb[3] = 2 << (unsigned int)(4 * (row_l & 7));
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_idb_addr + (unsigned int)(row_l * 32 ^ (row_l * 32 >> 7 & 1) << 4))), "r"(idb[0]), "r"(idb[1]), "r"(idb[2]), "r"(idb[3]) : "memory");
                idb[0] = 0;
                if (row_l >> 5 == 1) {
                    if ((row_l & 31) >> 3 == 0) {
                        idb[0] = 2 << (unsigned int)(4 * (row_l & 7));
                    }
                }
                idb[1] = 0;
                if (row_l >> 5 == 1) {
                    if ((row_l & 31) >> 3 == 1) {
                        idb[1] = 2 << (unsigned int)(4 * (row_l & 7));
                    }
                }
                idb[2] = 0;
                if (row_l >> 5 == 1) {
                    if ((row_l & 31) >> 3 == 2) {
                        idb[2] = 2 << (unsigned int)(4 * (row_l & 7));
                    }
                }
                idb[3] = 0;
                if (row_l >> 5 == 1) {
                    if ((row_l & 31) >> 3 == 3) {
                        idb[3] = 2 << (unsigned int)(4 * (row_l & 7));
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_idb_addr + (unsigned int)(row_l * 32 + 16 ^ (row_l * 32 + 16 >> 7 & 1) << 4))), "r"(idb[0]), "r"(idb[1]), "r"(idb[2]), "r"(idb[3]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(const_ready_addr);
            }
            inx[0] = -1;
            if (row_l < n_keys) {
                int _vec_load_17[1];
                {
                    uint32_t _scalar_bits_0;
                    asm volatile("ld.global.nc.b32 %0, [%1];"
                        : "=r"(_scalar_bits_0) : "l"((const void*)(indices + (token * topk + key_lo + row_l))) : "memory");
                    _vec_load_17[0] = (int32_t)_scalar_bits_0;
                }
                inx[0] = _vec_load_17[0];
            }
            cur[0] = 0;
            cur[1] = 0;
            cur[2] = 0;
            cur[3] = 0;
            cur[4] = 0;
            cur[5] = 0;
            cur[6] = 0;
            cur[7] = 0;
            cur[8] = 0;
            cur[9] = 0;
            cur[10] = 0;
            cur[11] = 0;
            cur[12] = 0;
            cur[13] = 0;
            cur[14] = 0;
            cur[15] = 0;
            cur[16] = 0;
            cur[17] = 0;
            cur[18] = 0;
            cur[19] = 0;
            cur[20] = 0;
            cur[21] = 0;
            cur[22] = 0;
            cur[23] = 0;
            if (inx[0] >= 0) {
                int _vec_load_18[4];
                {
                    const int4* _ivptr_1 = reinterpret_cast<const int4*>(kv + inx[0] * 88 + 64);
                    int4 _ivld_1;
                    _ivld_1 = *_ivptr_1;
                    _vec_load_18[0 + 0] = _ivld_1.x;
                    _vec_load_18[0 + 1] = _ivld_1.y;
                    _vec_load_18[0 + 2] = _ivld_1.z;
                    _vec_load_18[0 + 3] = _ivld_1.w;
                }
                cur[0] = (unsigned int)_vec_load_18[0];
                cur[1] = (unsigned int)_vec_load_18[1];
                cur[2] = (unsigned int)_vec_load_18[2];
                cur[3] = (unsigned int)_vec_load_18[3];
                int _vec_load_19[4];
                {
                    const int4* _ivptr_2 = reinterpret_cast<const int4*>(kv + inx[0] * 88 + 64 + 4);
                    int4 _ivld_2;
                    _ivld_2 = *_ivptr_2;
                    _vec_load_19[0 + 0] = _ivld_2.x;
                    _vec_load_19[0 + 1] = _ivld_2.y;
                    _vec_load_19[0 + 2] = _ivld_2.z;
                    _vec_load_19[0 + 3] = _ivld_2.w;
                }
                cur[4] = (unsigned int)_vec_load_19[0];
                cur[5] = (unsigned int)_vec_load_19[1];
                cur[6] = (unsigned int)_vec_load_19[2];
                cur[7] = (unsigned int)_vec_load_19[3];
                int _vec_load_20[4];
                {
                    const int4* _ivptr_3 = reinterpret_cast<const int4*>(kv + inx[0] * 88 + 64 + 8);
                    int4 _ivld_3;
                    _ivld_3 = *_ivptr_3;
                    _vec_load_20[0 + 0] = _ivld_3.x;
                    _vec_load_20[0 + 1] = _ivld_3.y;
                    _vec_load_20[0 + 2] = _ivld_3.z;
                    _vec_load_20[0 + 3] = _ivld_3.w;
                }
                cur[8] = (unsigned int)_vec_load_20[0];
                cur[9] = (unsigned int)_vec_load_20[1];
                cur[10] = (unsigned int)_vec_load_20[2];
                cur[11] = (unsigned int)_vec_load_20[3];
                int _vec_load_21[4];
                {
                    const int4* _ivptr_4 = reinterpret_cast<const int4*>(kv + inx[0] * 88 + 64 + 12);
                    int4 _ivld_4;
                    _ivld_4 = *_ivptr_4;
                    _vec_load_21[0 + 0] = _ivld_4.x;
                    _vec_load_21[0 + 1] = _ivld_4.y;
                    _vec_load_21[0 + 2] = _ivld_4.z;
                    _vec_load_21[0 + 3] = _ivld_4.w;
                }
                cur[12] = (unsigned int)_vec_load_21[0];
                cur[13] = (unsigned int)_vec_load_21[1];
                cur[14] = (unsigned int)_vec_load_21[2];
                cur[15] = (unsigned int)_vec_load_21[3];
                int _vec_load_22[4];
                {
                    const int4* _ivptr_5 = reinterpret_cast<const int4*>(kv + inx[0] * 88 + 64 + 16);
                    int4 _ivld_5;
                    _ivld_5 = *_ivptr_5;
                    _vec_load_22[0 + 0] = _ivld_5.x;
                    _vec_load_22[0 + 1] = _ivld_5.y;
                    _vec_load_22[0 + 2] = _ivld_5.z;
                    _vec_load_22[0 + 3] = _ivld_5.w;
                }
                cur[16] = (unsigned int)_vec_load_22[0];
                cur[17] = (unsigned int)_vec_load_22[1];
                cur[18] = (unsigned int)_vec_load_22[2];
                cur[19] = (unsigned int)_vec_load_22[3];
                int _vec_load_23[4];
                {
                    const int4* _ivptr_6 = reinterpret_cast<const int4*>(kv + inx[0] * 88 + 64 + 20);
                    int4 _ivld_6;
                    _ivld_6 = *_ivptr_6;
                    _vec_load_23[0 + 0] = _ivld_6.x;
                    _vec_load_23[0 + 1] = _ivld_6.y;
                    _vec_load_23[0 + 2] = _ivld_6.z;
                    _vec_load_23[0 + 3] = _ivld_6.w;
                }
                cur[20] = (unsigned int)_vec_load_23[0];
                cur[21] = (unsigned int)_vec_load_23[1];
                cur[22] = (unsigned int)_vec_load_23[2];
                cur[23] = (unsigned int)_vec_load_23[3];
            }
            unsigned int _phase_nope_full = 0;
            #pragma unroll
            for (int t = 0; t < 1; t++) {
                if (t + 1 < 1) {
                    nxt[0] = 0;
                    nxt[1] = 0;
                    nxt[2] = 0;
                    nxt[3] = 0;
                    nxt[4] = 0;
                    nxt[5] = 0;
                    nxt[6] = 0;
                    nxt[7] = 0;
                    nxt[8] = 0;
                    nxt[9] = 0;
                    nxt[10] = 0;
                    nxt[11] = 0;
                    nxt[12] = 0;
                    nxt[13] = 0;
                    nxt[14] = 0;
                    nxt[15] = 0;
                    nxt[16] = 0;
                    nxt[17] = 0;
                    nxt[18] = 0;
                    nxt[19] = 0;
                    nxt[20] = 0;
                    nxt[21] = 0;
                    nxt[22] = 0;
                    nxt[23] = 0;
                    if (inx[0] >= 0) {
                        int _vec_load_24[4];
                        {
                            const int4* _ivptr_7 = reinterpret_cast<const int4*>(kv + inx[0] * 88 + 64);
                            int4 _ivld_7;
                            _ivld_7 = *_ivptr_7;
                            _vec_load_24[0 + 0] = _ivld_7.x;
                            _vec_load_24[0 + 1] = _ivld_7.y;
                            _vec_load_24[0 + 2] = _ivld_7.z;
                            _vec_load_24[0 + 3] = _ivld_7.w;
                        }
                        nxt[0] = (unsigned int)_vec_load_24[0];
                        nxt[1] = (unsigned int)_vec_load_24[1];
                        nxt[2] = (unsigned int)_vec_load_24[2];
                        nxt[3] = (unsigned int)_vec_load_24[3];
                        int _vec_load_25[4];
                        {
                            const int4* _ivptr_8 = reinterpret_cast<const int4*>(kv + inx[0] * 88 + 64 + 4);
                            int4 _ivld_8;
                            _ivld_8 = *_ivptr_8;
                            _vec_load_25[0 + 0] = _ivld_8.x;
                            _vec_load_25[0 + 1] = _ivld_8.y;
                            _vec_load_25[0 + 2] = _ivld_8.z;
                            _vec_load_25[0 + 3] = _ivld_8.w;
                        }
                        nxt[4] = (unsigned int)_vec_load_25[0];
                        nxt[5] = (unsigned int)_vec_load_25[1];
                        nxt[6] = (unsigned int)_vec_load_25[2];
                        nxt[7] = (unsigned int)_vec_load_25[3];
                        int _vec_load_26[4];
                        {
                            const int4* _ivptr_9 = reinterpret_cast<const int4*>(kv + inx[0] * 88 + 64 + 8);
                            int4 _ivld_9;
                            _ivld_9 = *_ivptr_9;
                            _vec_load_26[0 + 0] = _ivld_9.x;
                            _vec_load_26[0 + 1] = _ivld_9.y;
                            _vec_load_26[0 + 2] = _ivld_9.z;
                            _vec_load_26[0 + 3] = _ivld_9.w;
                        }
                        nxt[8] = (unsigned int)_vec_load_26[0];
                        nxt[9] = (unsigned int)_vec_load_26[1];
                        nxt[10] = (unsigned int)_vec_load_26[2];
                        nxt[11] = (unsigned int)_vec_load_26[3];
                        int _vec_load_27[4];
                        {
                            const int4* _ivptr_10 = reinterpret_cast<const int4*>(kv + inx[0] * 88 + 64 + 12);
                            int4 _ivld_10;
                            _ivld_10 = *_ivptr_10;
                            _vec_load_27[0 + 0] = _ivld_10.x;
                            _vec_load_27[0 + 1] = _ivld_10.y;
                            _vec_load_27[0 + 2] = _ivld_10.z;
                            _vec_load_27[0 + 3] = _ivld_10.w;
                        }
                        nxt[12] = (unsigned int)_vec_load_27[0];
                        nxt[13] = (unsigned int)_vec_load_27[1];
                        nxt[14] = (unsigned int)_vec_load_27[2];
                        nxt[15] = (unsigned int)_vec_load_27[3];
                        int _vec_load_28[4];
                        {
                            const int4* _ivptr_11 = reinterpret_cast<const int4*>(kv + inx[0] * 88 + 64 + 16);
                            int4 _ivld_11;
                            _ivld_11 = *_ivptr_11;
                            _vec_load_28[0 + 0] = _ivld_11.x;
                            _vec_load_28[0 + 1] = _ivld_11.y;
                            _vec_load_28[0 + 2] = _ivld_11.z;
                            _vec_load_28[0 + 3] = _ivld_11.w;
                        }
                        nxt[16] = (unsigned int)_vec_load_28[0];
                        nxt[17] = (unsigned int)_vec_load_28[1];
                        nxt[18] = (unsigned int)_vec_load_28[2];
                        nxt[19] = (unsigned int)_vec_load_28[3];
                        int _vec_load_29[4];
                        {
                            const int4* _ivptr_12 = reinterpret_cast<const int4*>(kv + inx[0] * 88 + 64 + 20);
                            int4 _ivld_12;
                            _ivld_12 = *_ivptr_12;
                            _vec_load_29[0 + 0] = _ivld_12.x;
                            _vec_load_29[0 + 1] = _ivld_12.y;
                            _vec_load_29[0 + 2] = _ivld_12.z;
                            _vec_load_29[0 + 3] = _ivld_12.w;
                        }
                        nxt[20] = (unsigned int)_vec_load_29[0];
                        nxt[21] = (unsigned int)_vec_load_29[1];
                        nxt[22] = (unsigned int)_vec_load_29[2];
                        nxt[23] = (unsigned int)_vec_load_29[3];
                    }
                }
                unsigned int neg = 0;
                neg = neg | cur[16];
                neg = neg | cur[17];
                neg = neg | cur[18];
                neg = neg | cur[19];
                neg = neg | cur[20];
                neg = neg | cur[21];
                neg = neg | cur[22];
                neg = neg | cur[23];
                neg = neg & 2155905152u;
                if (neg != 0) {
                    mbarrier_wait(nope_full_addr, _phase_nope_full);
                    unsigned int nope_l = smem_nope_addr + 0 * 32768;
                    #pragma unroll 1
                    for (int b = 0; b < 32; b++) {
                        int kw = 2 * (b & 3) + (b >> 4);
                        unsigned int sword = cur[16];
                        sword = ((kw == 1) ? cur[17] : sword);
                        sword = ((kw == 2) ? cur[18] : sword);
                        sword = ((kw == 3) ? cur[19] : sword);
                        sword = ((kw == 4) ? cur[20] : sword);
                        sword = ((kw == 5) ? cur[21] : sword);
                        sword = ((kw == 6) ? cur[22] : sword);
                        sword = ((kw == 7) ? cur[23] : sword);
                        if ((sword >> (unsigned int)(8 * (b >> 2 & 3) + 7) & 1) != 0) {
                            unsigned int faddr = nope_l + (unsigned int)((b >> 4) * 16384 + row_l * 128 + (((b & 15) >> 1 ^ row_l & 7) << 4) + (b & 1) * 8);
                            unsigned int pair[2];
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pair[0])), "=r"(*reinterpret_cast<uint32_t*>(&pair[(0) + 1]))
                                : "r"(faddr));
                            pair[0] = pair[0] ^ 2290649224u;
                            pair[1] = pair[1] ^ 2290649224u;
                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(faddr), "r"(pair[0]), "r"(pair[1]) : "memory");
                        }
                    }
                    cur[16] = cur[16] & 2139062143;
                    cur[17] = cur[17] & 2139062143;
                    cur[18] = cur[18] & 2139062143;
                    cur[19] = cur[19] & 2139062143;
                    cur[20] = cur[20] & 2139062143;
                    cur[21] = cur[21] & 2139062143;
                    cur[22] = cur[22] & 2139062143;
                    cur[23] = cur[23] & 2139062143;
                }
                _phase_nope_full ^= 1;
                uint32_t lo;
                asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(lo) : "r"(cur[16]), "r"(cur[18]));
                uint32_t hi;
                asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(hi) : "r"(cur[20]), "r"(cur[22]));
                uint32_t _prmt_13;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_13) : "r"(lo), "r"(hi));
                sfw[0] = _prmt_13;
                uint32_t lo_0;
                asm("prmt.b32 %0, %1, %2, 0x0051;" : "=r"(lo_0) : "r"(cur[16]), "r"(cur[18]));
                uint32_t hi_1;
                asm("prmt.b32 %0, %1, %2, 0x0051;" : "=r"(hi_1) : "r"(cur[20]), "r"(cur[22]));
                uint32_t _prmt_14;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_14) : "r"(lo_0), "r"(hi_1));
                sfw[1] = _prmt_14;
                uint32_t lo_2;
                asm("prmt.b32 %0, %1, %2, 0x0062;" : "=r"(lo_2) : "r"(cur[16]), "r"(cur[18]));
                uint32_t hi_3;
                asm("prmt.b32 %0, %1, %2, 0x0062;" : "=r"(hi_3) : "r"(cur[20]), "r"(cur[22]));
                uint32_t _prmt_15;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_15) : "r"(lo_2), "r"(hi_3));
                sfw[2] = _prmt_15;
                uint32_t lo_4;
                asm("prmt.b32 %0, %1, %2, 0x0073;" : "=r"(lo_4) : "r"(cur[16]), "r"(cur[18]));
                uint32_t hi_5;
                asm("prmt.b32 %0, %1, %2, 0x0073;" : "=r"(hi_5) : "r"(cur[20]), "r"(cur[22]));
                uint32_t _prmt_16;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_16) : "r"(lo_4), "r"(hi_5));
                sfw[3] = _prmt_16;
                uint32_t lo_6;
                asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(lo_6) : "r"(cur[17]), "r"(cur[19]));
                uint32_t hi_7;
                asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(hi_7) : "r"(cur[21]), "r"(cur[23]));
                uint32_t _prmt_17;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_17) : "r"(lo_6), "r"(hi_7));
                sfw[4] = _prmt_17;
                uint32_t lo_8;
                asm("prmt.b32 %0, %1, %2, 0x0051;" : "=r"(lo_8) : "r"(cur[17]), "r"(cur[19]));
                uint32_t hi_9;
                asm("prmt.b32 %0, %1, %2, 0x0051;" : "=r"(hi_9) : "r"(cur[21]), "r"(cur[23]));
                uint32_t _prmt_18;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_18) : "r"(lo_8), "r"(hi_9));
                sfw[5] = _prmt_18;
                uint32_t lo_10;
                asm("prmt.b32 %0, %1, %2, 0x0062;" : "=r"(lo_10) : "r"(cur[17]), "r"(cur[19]));
                uint32_t hi_11;
                asm("prmt.b32 %0, %1, %2, 0x0062;" : "=r"(hi_11) : "r"(cur[21]), "r"(cur[23]));
                uint32_t _prmt_19;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_19) : "r"(lo_10), "r"(hi_11));
                sfw[6] = _prmt_19;
                uint32_t lo_12;
                asm("prmt.b32 %0, %1, %2, 0x0073;" : "=r"(lo_12) : "r"(cur[17]), "r"(cur[19]));
                uint32_t hi_13;
                asm("prmt.b32 %0, %1, %2, 0x0073;" : "=r"(hi_13) : "r"(cur[21]), "r"(cur[23]));
                uint32_t _prmt_20;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_20) : "r"(lo_12), "r"(hi_13));
                sfw[7] = _prmt_20;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_addr + 0 * 8192 + (unsigned int)((wl >> 1) * 512 + lane_l * 16 + (wl & 1) * 4)), "r"(sfw[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_addr + 0 * 8192 + (unsigned int)((2 + (wl >> 1)) * 512 + lane_l * 16 + (wl & 1) * 4)), "r"(sfw[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_addr + 0 * 8192 + (unsigned int)((4 + (wl >> 1)) * 512 + lane_l * 16 + (wl & 1) * 4)), "r"(sfw[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_addr + 0 * 8192 + (unsigned int)((6 + (wl >> 1)) * 512 + lane_l * 16 + (wl & 1) * 4)), "r"(sfw[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_addr + 0 * 8192 + (unsigned int)((8 + (wl >> 1)) * 512 + lane_l * 16 + (wl & 1) * 4)), "r"(sfw[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_addr + 0 * 8192 + (unsigned int)((10 + (wl >> 1)) * 512 + lane_l * 16 + (wl & 1) * 4)), "r"(sfw[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_addr + 0 * 8192 + (unsigned int)((12 + (wl >> 1)) * 512 + lane_l * 16 + (wl & 1) * 4)), "r"(sfw[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_addr + 0 * 8192 + (unsigned int)((14 + (wl >> 1)) * 512 + lane_l * 16 + (wl & 1) * 4)), "r"(sfw[7]) : "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                words[0] = sfw[0];
                words[1] = sfw[0];
                words[2] = sfw[0];
                words[3] = sfw[0];
                words[4] = sfw[1];
                words[5] = sfw[1];
                words[6] = sfw[1];
                words[7] = sfw[1];
                words[8] = sfw[2];
                words[9] = sfw[2];
                words[10] = sfw[2];
                words[11] = sfw[2];
                words[12] = sfw[3];
                words[13] = sfw[3];
                words[14] = sfw[3];
                words[15] = sfw[3];
                words[16] = sfw[4];
                words[17] = sfw[4];
                words[18] = sfw[4];
                words[19] = sfw[4];
                words[20] = sfw[5];
                words[21] = sfw[5];
                words[22] = sfw[5];
                words[23] = sfw[5];
                words[24] = sfw[6];
                words[25] = sfw[6];
                words[26] = sfw[6];
                words[27] = sfw[6];
                words[28] = sfw[7];
                words[29] = sfw[7];
                words[30] = sfw[7];
                words[31] = sfw[7];
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 424 + (unsigned int)0 * 32 + lane_addr_l), "r"(words[0]), "r"(words[1]), "r"(words[2]), "r"(words[3]), "r"(words[4]), "r"(words[5]), "r"(words[6]), "r"(words[7]), "r"(words[8]), "r"(words[9]), "r"(words[10]), "r"(words[11]), "r"(words[12]), "r"(words[13]), "r"(words[14]), "r"(words[15]), "r"(words[16]), "r"(words[17]), "r"(words[18]), "r"(words[19]), "r"(words[20]), "r"(words[21]), "r"(words[22]), "r"(words[23]), "r"(words[24]), "r"(words[25]), "r"(words[26]), "r"(words[27]), "r"(words[28]), "r"(words[29]), "r"(words[30]), "r"(words[31]));
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t _cvt_f16x2_e4m3x2_21;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_21) : "h"((uint16_t)((uint16_t)(cur[0] & 65535))));
                words[0] = _cvt_f16x2_e4m3x2_21;
                uint32_t _cvt_f16x2_e4m3x2_22;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_22) : "h"((uint16_t)((uint16_t)(cur[0] >> 16))));
                words[1] = _cvt_f16x2_e4m3x2_22;
                uint32_t _cvt_f16x2_e4m3x2_23;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_23) : "h"((uint16_t)((uint16_t)(cur[1] & 65535))));
                words[2] = _cvt_f16x2_e4m3x2_23;
                uint32_t _cvt_f16x2_e4m3x2_24;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_24) : "h"((uint16_t)((uint16_t)(cur[1] >> 16))));
                words[3] = _cvt_f16x2_e4m3x2_24;
                uint32_t _cvt_f16x2_e4m3x2_25;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_25) : "h"((uint16_t)((uint16_t)(cur[2] & 65535))));
                words[4] = _cvt_f16x2_e4m3x2_25;
                uint32_t _cvt_f16x2_e4m3x2_26;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_26) : "h"((uint16_t)((uint16_t)(cur[2] >> 16))));
                words[5] = _cvt_f16x2_e4m3x2_26;
                uint32_t _cvt_f16x2_e4m3x2_27;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_27) : "h"((uint16_t)((uint16_t)(cur[3] & 65535))));
                words[6] = _cvt_f16x2_e4m3x2_27;
                uint32_t _cvt_f16x2_e4m3x2_28;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_28) : "h"((uint16_t)((uint16_t)(cur[3] >> 16))));
                words[7] = _cvt_f16x2_e4m3x2_28;
                uint32_t _cvt_f16x2_e4m3x2_29;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_29) : "h"((uint16_t)((uint16_t)(cur[4] & 65535))));
                words[8] = _cvt_f16x2_e4m3x2_29;
                uint32_t _cvt_f16x2_e4m3x2_30;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_30) : "h"((uint16_t)((uint16_t)(cur[4] >> 16))));
                words[9] = _cvt_f16x2_e4m3x2_30;
                uint32_t _cvt_f16x2_e4m3x2_31;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_31) : "h"((uint16_t)((uint16_t)(cur[5] & 65535))));
                words[10] = _cvt_f16x2_e4m3x2_31;
                uint32_t _cvt_f16x2_e4m3x2_32;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_32) : "h"((uint16_t)((uint16_t)(cur[5] >> 16))));
                words[11] = _cvt_f16x2_e4m3x2_32;
                uint32_t _cvt_f16x2_e4m3x2_33;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_33) : "h"((uint16_t)((uint16_t)(cur[6] & 65535))));
                words[12] = _cvt_f16x2_e4m3x2_33;
                uint32_t _cvt_f16x2_e4m3x2_34;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_34) : "h"((uint16_t)((uint16_t)(cur[6] >> 16))));
                words[13] = _cvt_f16x2_e4m3x2_34;
                uint32_t _cvt_f16x2_e4m3x2_35;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_35) : "h"((uint16_t)((uint16_t)(cur[7] & 65535))));
                words[14] = _cvt_f16x2_e4m3x2_35;
                uint32_t _cvt_f16x2_e4m3x2_36;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_36) : "h"((uint16_t)((uint16_t)(cur[7] >> 16))));
                words[15] = _cvt_f16x2_e4m3x2_36;
                uint32_t _cvt_f16x2_e4m3x2_37;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_37) : "h"((uint16_t)((uint16_t)(cur[8] & 65535))));
                words[16] = _cvt_f16x2_e4m3x2_37;
                uint32_t _cvt_f16x2_e4m3x2_38;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_38) : "h"((uint16_t)((uint16_t)(cur[8] >> 16))));
                words[17] = _cvt_f16x2_e4m3x2_38;
                uint32_t _cvt_f16x2_e4m3x2_39;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_39) : "h"((uint16_t)((uint16_t)(cur[9] & 65535))));
                words[18] = _cvt_f16x2_e4m3x2_39;
                uint32_t _cvt_f16x2_e4m3x2_40;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_40) : "h"((uint16_t)((uint16_t)(cur[9] >> 16))));
                words[19] = _cvt_f16x2_e4m3x2_40;
                uint32_t _cvt_f16x2_e4m3x2_41;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_41) : "h"((uint16_t)((uint16_t)(cur[10] & 65535))));
                words[20] = _cvt_f16x2_e4m3x2_41;
                uint32_t _cvt_f16x2_e4m3x2_42;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_42) : "h"((uint16_t)((uint16_t)(cur[10] >> 16))));
                words[21] = _cvt_f16x2_e4m3x2_42;
                uint32_t _cvt_f16x2_e4m3x2_43;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_43) : "h"((uint16_t)((uint16_t)(cur[11] & 65535))));
                words[22] = _cvt_f16x2_e4m3x2_43;
                uint32_t _cvt_f16x2_e4m3x2_44;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_44) : "h"((uint16_t)((uint16_t)(cur[11] >> 16))));
                words[23] = _cvt_f16x2_e4m3x2_44;
                uint32_t _cvt_f16x2_e4m3x2_45;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_45) : "h"((uint16_t)((uint16_t)(cur[12] & 65535))));
                words[24] = _cvt_f16x2_e4m3x2_45;
                uint32_t _cvt_f16x2_e4m3x2_46;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_46) : "h"((uint16_t)((uint16_t)(cur[12] >> 16))));
                words[25] = _cvt_f16x2_e4m3x2_46;
                uint32_t _cvt_f16x2_e4m3x2_47;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_47) : "h"((uint16_t)((uint16_t)(cur[13] & 65535))));
                words[26] = _cvt_f16x2_e4m3x2_47;
                uint32_t _cvt_f16x2_e4m3x2_48;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_48) : "h"((uint16_t)((uint16_t)(cur[13] >> 16))));
                words[27] = _cvt_f16x2_e4m3x2_48;
                uint32_t _cvt_f16x2_e4m3x2_49;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_49) : "h"((uint16_t)((uint16_t)(cur[14] & 65535))));
                words[28] = _cvt_f16x2_e4m3x2_49;
                uint32_t _cvt_f16x2_e4m3x2_50;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_50) : "h"((uint16_t)((uint16_t)(cur[14] >> 16))));
                words[29] = _cvt_f16x2_e4m3x2_50;
                uint32_t _cvt_f16x2_e4m3x2_51;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_51) : "h"((uint16_t)((uint16_t)(cur[15] & 65535))));
                words[30] = _cvt_f16x2_e4m3x2_51;
                uint32_t _cvt_f16x2_e4m3x2_52;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_52) : "h"((uint16_t)((uint16_t)(cur[15] >> 16))));
                words[31] = _cvt_f16x2_e4m3x2_52;
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 128 + lane_addr_l), "r"(words[0]), "r"(words[1]), "r"(words[2]), "r"(words[3]), "r"(words[4]), "r"(words[5]), "r"(words[6]), "r"(words[7]), "r"(words[8]), "r"(words[9]), "r"(words[10]), "r"(words[11]), "r"(words[12]), "r"(words[13]), "r"(words[14]), "r"(words[15]), "r"(words[16]), "r"(words[17]), "r"(words[18]), "r"(words[19]), "r"(words[20]), "r"(words[21]), "r"(words[22]), "r"(words[23]), "r"(words[24]), "r"(words[25]), "r"(words[26]), "r"(words[27]), "r"(words[28]), "r"(words[29]), "r"(words[30]), "r"(words[31]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(sfk_full_addr);
                    mbarrier_arrive(sfv_full_addr);
                    mbarrier_arrive(kr_full_addr);
                }
            }
        }
    }
    // ---- Role: sm ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
        { // sm_main
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            int lane_s = lane;
            int sq = warp % 4;
            int tid_s = sq * 32 + lane_s;
            unsigned int lane_addr_s = (unsigned int)(sq * 32) << 16;
            unsigned int qh[8];
            float qf[16];
            unsigned int qv[4];
            unsigned int q_stage = smem_p_addr;
            asm volatile("cp.async.wait_group 0;");
            #pragma unroll 1
            for (int i = 0; i < 4; i++) {
                int nq = tid_s + 128 * i;
                int head = nq >> 5;
                int d0 = (nq & 31) * 16;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&qv[0])), "=r"(*reinterpret_cast<uint32_t*>(&qv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qv[(0) + 3]))
                    : "r"(q_stage + (unsigned int)((head * 36 + (nq & 31)) * 16)));
                uint32_t _cvt_f16x2_e4m3x2_0;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_0) : "h"((uint16_t)((uint16_t)((unsigned int)qv[0] & 65535))));
                qh[0] = _cvt_f16x2_e4m3x2_0;
                uint32_t _cvt_f16x2_e4m3x2_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_1) : "h"((uint16_t)((uint16_t)((unsigned int)qv[0] >> 16))));
                qh[1] = _cvt_f16x2_e4m3x2_1;
                uint32_t _cvt_f16x2_e4m3x2_2;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_2) : "h"((uint16_t)((uint16_t)((unsigned int)qv[1] & 65535))));
                qh[2] = _cvt_f16x2_e4m3x2_2;
                uint32_t _cvt_f16x2_e4m3x2_3;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_3) : "h"((uint16_t)((uint16_t)((unsigned int)qv[1] >> 16))));
                qh[3] = _cvt_f16x2_e4m3x2_3;
                uint32_t _cvt_f16x2_e4m3x2_4;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_4) : "h"((uint16_t)((uint16_t)((unsigned int)qv[2] & 65535))));
                qh[4] = _cvt_f16x2_e4m3x2_4;
                uint32_t _cvt_f16x2_e4m3x2_5;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_5) : "h"((uint16_t)((uint16_t)((unsigned int)qv[2] >> 16))));
                qh[5] = _cvt_f16x2_e4m3x2_5;
                uint32_t _cvt_f16x2_e4m3x2_6;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_6) : "h"((uint16_t)((uint16_t)((unsigned int)qv[3] & 65535))));
                qh[6] = _cvt_f16x2_e4m3x2_6;
                uint32_t _cvt_f16x2_e4m3x2_7;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_7) : "h"((uint16_t)((uint16_t)((unsigned int)qv[3] >> 16))));
                qh[7] = _cvt_f16x2_e4m3x2_7;
                float _cvt_f32_f16_8;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_8) : "h"((uint16_t)(qh[0])));
                qf[0] = _cvt_f32_f16_8;
                float _cvt_f32_f16_9;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_9) : "h"((uint16_t)(qh[0] >> 16)));
                qf[1] = _cvt_f32_f16_9;
                float _cvt_f32_f16_10;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_10) : "h"((uint16_t)(qh[1])));
                qf[2] = _cvt_f32_f16_10;
                float _cvt_f32_f16_11;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_11) : "h"((uint16_t)(qh[1] >> 16)));
                qf[3] = _cvt_f32_f16_11;
                float _cvt_f32_f16_12;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_12) : "h"((uint16_t)(qh[2])));
                qf[4] = _cvt_f32_f16_12;
                float _cvt_f32_f16_13;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_13) : "h"((uint16_t)(qh[2] >> 16)));
                qf[5] = _cvt_f32_f16_13;
                float _cvt_f32_f16_14;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_14) : "h"((uint16_t)(qh[3])));
                qf[6] = _cvt_f32_f16_14;
                float _cvt_f32_f16_15;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_15) : "h"((uint16_t)(qh[3] >> 16)));
                qf[7] = _cvt_f32_f16_15;
                float _cvt_f32_f16_16;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_16) : "h"((uint16_t)(qh[4])));
                qf[8] = _cvt_f32_f16_16;
                float _cvt_f32_f16_17;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_17) : "h"((uint16_t)(qh[4] >> 16)));
                qf[9] = _cvt_f32_f16_17;
                float _cvt_f32_f16_18;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_18) : "h"((uint16_t)(qh[5])));
                qf[10] = _cvt_f32_f16_18;
                float _cvt_f32_f16_19;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_19) : "h"((uint16_t)(qh[5] >> 16)));
                qf[11] = _cvt_f32_f16_19;
                float _cvt_f32_f16_20;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_20) : "h"((uint16_t)(qh[6])));
                qf[12] = _cvt_f32_f16_20;
                float _cvt_f32_f16_21;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_21) : "h"((uint16_t)(qh[6] >> 16)));
                qf[13] = _cvt_f32_f16_21;
                float _cvt_f32_f16_22;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_22) : "h"((uint16_t)(qh[7])));
                qf[14] = _cvt_f32_f16_22;
                float _cvt_f32_f16_23;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_23) : "h"((uint16_t)(qh[7] >> 16)));
                qf[15] = _cvt_f32_f16_23;
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((smem_qn_addr + (unsigned int)(d0 / 32 * 2048 + head * 128 + d0 % 32 * 4 ^ (d0 / 32 * 2048 + head * 128 + d0 % 32 * 4 >> 7 & 7) << 4))), "f"(qf[0]), "f"(qf[1]), "f"(qf[2]), "f"(qf[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((smem_qn_addr + (unsigned int)((d0 + 4) / 32 * 2048 + head * 128 + (d0 + 4) % 32 * 4 ^ ((d0 + 4) / 32 * 2048 + head * 128 + (d0 + 4) % 32 * 4 >> 7 & 7) << 4))), "f"(qf[4]), "f"(qf[5]), "f"(qf[6]), "f"(qf[7]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((smem_qn_addr + (unsigned int)((d0 + 8) / 32 * 2048 + head * 128 + (d0 + 8) % 32 * 4 ^ ((d0 + 8) / 32 * 2048 + head * 128 + (d0 + 8) % 32 * 4 >> 7 & 7) << 4))), "f"(qf[8]), "f"(qf[9]), "f"(qf[10]), "f"(qf[11]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((smem_qn_addr + (unsigned int)((d0 + 12) / 32 * 2048 + head * 128 + (d0 + 12) % 32 * 4 ^ ((d0 + 12) / 32 * 2048 + head * 128 + (d0 + 12) % 32 * 4 >> 7 & 7) << 4))), "f"(qf[12]), "f"(qf[13]), "f"(qf[14]), "f"(qf[15]) : "memory");
            }
            if (tid_s < 64) {
                int hr = tid_s >> 2;
                int dr = (tid_s & 3) * 16;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&qv[0])), "=r"(*reinterpret_cast<uint32_t*>(&qv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qv[(0) + 3]))
                    : "r"(q_stage + (unsigned int)((hr * 36 + 32 + (tid_s & 3)) * 16)));
                uint32_t _cvt_f16x2_e4m3x2_24_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_24_1) : "h"((uint16_t)((uint16_t)((unsigned int)qv[0] & 65535))));
                qh[0] = _cvt_f16x2_e4m3x2_24_1;
                uint32_t _cvt_f16x2_e4m3x2_25_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_25_1) : "h"((uint16_t)((uint16_t)((unsigned int)qv[0] >> 16))));
                qh[1] = _cvt_f16x2_e4m3x2_25_1;
                uint32_t _cvt_f16x2_e4m3x2_26_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_26_1) : "h"((uint16_t)((uint16_t)((unsigned int)qv[1] & 65535))));
                qh[2] = _cvt_f16x2_e4m3x2_26_1;
                uint32_t _cvt_f16x2_e4m3x2_27_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_27_1) : "h"((uint16_t)((uint16_t)((unsigned int)qv[1] >> 16))));
                qh[3] = _cvt_f16x2_e4m3x2_27_1;
                uint32_t _cvt_f16x2_e4m3x2_28_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_28_1) : "h"((uint16_t)((uint16_t)((unsigned int)qv[2] & 65535))));
                qh[4] = _cvt_f16x2_e4m3x2_28_1;
                uint32_t _cvt_f16x2_e4m3x2_29_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_29_1) : "h"((uint16_t)((uint16_t)((unsigned int)qv[2] >> 16))));
                qh[5] = _cvt_f16x2_e4m3x2_29_1;
                uint32_t _cvt_f16x2_e4m3x2_30_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_30_1) : "h"((uint16_t)((uint16_t)((unsigned int)qv[3] & 65535))));
                qh[6] = _cvt_f16x2_e4m3x2_30_1;
                uint32_t _cvt_f16x2_e4m3x2_31_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_31_1) : "h"((uint16_t)((uint16_t)((unsigned int)qv[3] >> 16))));
                qh[7] = _cvt_f16x2_e4m3x2_31_1;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_qr_addr + (unsigned int)(dr / 64 * 2048 + hr * 128 + dr % 64 * 2 ^ (dr / 64 * 2048 + hr * 128 + dr % 64 * 2 >> 7 & 7) << 4))), "r"(qh[0]), "r"(qh[1]), "r"(qh[2]), "r"(qh[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_qr_addr + (unsigned int)((dr + 8) / 64 * 2048 + hr * 128 + (dr + 8) % 64 * 2 ^ ((dr + 8) / 64 * 2048 + hr * 128 + (dr + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(qh[4]), "r"(qh[5]), "r"(qh[6]), "r"(qh[7]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(q_full_addr);
            }
            if (tid_s == 0) {
                int ngr = (16 - (cta + 64) % 4 + 3) / 4 + (16 - (cta - 1 + 64) % 4 + 3) / 4 + (16 - (cta - 2 + 64) % 4 + 3) / 4 + (16 - (cta - 3 + 64) % 4 + 3) / 4;
                mbarrier_arrive_expect_tx(zone_bar_addr, 3 * ngr * 512);
                mbarrier_arrive_expect_tx(ml_bar_addr, 384);
            }
            float m_run[16];
            float wm[16];
            float m_lane = -1e+30f;
            float l_run[16];
            float alpha[16];
            float sv[16];
            m_run[0] = -1e+30f;
            l_run[0] = 0.0f;
            m_run[1] = -1e+30f;
            l_run[1] = 0.0f;
            m_run[2] = -1e+30f;
            l_run[2] = 0.0f;
            m_run[3] = -1e+30f;
            l_run[3] = 0.0f;
            m_run[4] = -1e+30f;
            l_run[4] = 0.0f;
            m_run[5] = -1e+30f;
            l_run[5] = 0.0f;
            m_run[6] = -1e+30f;
            l_run[6] = 0.0f;
            m_run[7] = -1e+30f;
            l_run[7] = 0.0f;
            m_run[8] = -1e+30f;
            l_run[8] = 0.0f;
            m_run[9] = -1e+30f;
            l_run[9] = 0.0f;
            m_run[10] = -1e+30f;
            l_run[10] = 0.0f;
            m_run[11] = -1e+30f;
            l_run[11] = 0.0f;
            m_run[12] = -1e+30f;
            l_run[12] = 0.0f;
            m_run[13] = -1e+30f;
            l_run[13] = 0.0f;
            m_run[14] = -1e+30f;
            l_run[14] = 0.0f;
            m_run[15] = -1e+30f;
            l_run[15] = 0.0f;
            int vid[1];
            unsigned int _phase_s_full = 0;
            #pragma unroll
            for (int t_1 = 0; t_1 < 1; t_1++) {
                int par = t_1 & 1;
                int e_s = t_1 * 128 + tid_s;
                vid[0] = -1;
                if (e_s < n_keys) {
                    int _vec_load_30[1];
                    {
                        uint32_t _scalar_bits_32;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_32) : "l"((const void*)(indices + (token * topk + key_lo + e_s))) : "memory");
                        _vec_load_30[0] = (int32_t)_scalar_bits_32;
                    }
                    vid[0] = _vec_load_30[0];
                }
                int valid = ((vid[0] >= 0) ? 1 : 0);
                mbarrier_wait(s_full_addr, _phase_s_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], taddr + 160 + (unsigned int)0 * 16 + lane_addr_s);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                _phase_s_full ^= 1;
                sv[0] = ((valid != 0) ? _tmem_load_0[0] * qk_scale : -1e+30f);
                float _warp_redux_33;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_33) : "f"(sv[0]));
                wm[0] = _warp_redux_33;
                sv[1] = ((valid != 0) ? _tmem_load_0[1] * qk_scale : -1e+30f);
                float _warp_redux_34;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_34) : "f"(sv[1]));
                wm[1] = _warp_redux_34;
                sv[2] = ((valid != 0) ? _tmem_load_0[2] * qk_scale : -1e+30f);
                float _warp_redux_35;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_35) : "f"(sv[2]));
                wm[2] = _warp_redux_35;
                sv[3] = ((valid != 0) ? _tmem_load_0[3] * qk_scale : -1e+30f);
                float _warp_redux_36;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_36) : "f"(sv[3]));
                wm[3] = _warp_redux_36;
                sv[4] = ((valid != 0) ? _tmem_load_0[4] * qk_scale : -1e+30f);
                float _warp_redux_37;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_37) : "f"(sv[4]));
                wm[4] = _warp_redux_37;
                sv[5] = ((valid != 0) ? _tmem_load_0[5] * qk_scale : -1e+30f);
                float _warp_redux_38;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_38) : "f"(sv[5]));
                wm[5] = _warp_redux_38;
                sv[6] = ((valid != 0) ? _tmem_load_0[6] * qk_scale : -1e+30f);
                float _warp_redux_39;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_39) : "f"(sv[6]));
                wm[6] = _warp_redux_39;
                sv[7] = ((valid != 0) ? _tmem_load_0[7] * qk_scale : -1e+30f);
                float _warp_redux_40;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_40) : "f"(sv[7]));
                wm[7] = _warp_redux_40;
                sv[8] = ((valid != 0) ? _tmem_load_0[8] * qk_scale : -1e+30f);
                float _warp_redux_41;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_41) : "f"(sv[8]));
                wm[8] = _warp_redux_41;
                sv[9] = ((valid != 0) ? _tmem_load_0[9] * qk_scale : -1e+30f);
                float _warp_redux_42;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_42) : "f"(sv[9]));
                wm[9] = _warp_redux_42;
                sv[10] = ((valid != 0) ? _tmem_load_0[10] * qk_scale : -1e+30f);
                float _warp_redux_43;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_43) : "f"(sv[10]));
                wm[10] = _warp_redux_43;
                sv[11] = ((valid != 0) ? _tmem_load_0[11] * qk_scale : -1e+30f);
                float _warp_redux_44;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_44) : "f"(sv[11]));
                wm[11] = _warp_redux_44;
                sv[12] = ((valid != 0) ? _tmem_load_0[12] * qk_scale : -1e+30f);
                float _warp_redux_45;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_45) : "f"(sv[12]));
                wm[12] = _warp_redux_45;
                sv[13] = ((valid != 0) ? _tmem_load_0[13] * qk_scale : -1e+30f);
                float _warp_redux_46;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_46) : "f"(sv[13]));
                wm[13] = _warp_redux_46;
                sv[14] = ((valid != 0) ? _tmem_load_0[14] * qk_scale : -1e+30f);
                float _warp_redux_47;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_47) : "f"(sv[14]));
                wm[14] = _warp_redux_47;
                sv[15] = ((valid != 0) ? _tmem_load_0[15] * qk_scale : -1e+30f);
                float _warp_redux_48;
                asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_48) : "f"(sv[15]));
                wm[15] = _warp_redux_48;
                if (lane_s == 0) {
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(xmax_addr + (unsigned int)((par * 64 + sq * 16) * 4)), "f"(wm[0]), "f"(wm[1]), "f"(wm[2]), "f"(wm[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(xmax_addr + (unsigned int)((par * 64 + sq * 16 + 4) * 4)), "f"(wm[4]), "f"(wm[5]), "f"(wm[6]), "f"(wm[7]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(xmax_addr + (unsigned int)((par * 64 + sq * 16 + 8) * 4)), "f"(wm[8]), "f"(wm[9]), "f"(wm[10]), "f"(wm[11]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(xmax_addr + (unsigned int)((par * 64 + sq * 16 + 12) * 4)), "f"(wm[12]), "f"(wm[13]), "f"(wm[14]), "f"(wm[15]) : "memory");
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                float hmax = -1e+30f;
                if (lane_s < 16) {
                    float a0 = xmax[par * 64 + lane_s];
                    float a1 = xmax[par * 64 + 16 + lane_s];
                    float a2 = xmax[par * 64 + 32 + lane_s];
                    float a3 = xmax[par * 64 + 48 + lane_s];
                    hmax = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));
                }
                m_lane = fmaxf(m_lane, hmax);
                int grow = 0;
                unsigned int p_base = smem_p_addr + 0 * 8192;
                float m_new = __shfl_sync(0xFFFFFFFF, m_lane, 0);
                float _exp2_49 = approx_exp2(sv[0] - m_new);
                float p = ((valid != 0) ? _exp2_49 : 0.0f);
                l_run[0] = p;
                m_run[0] = m_new;
                float p16 = (float)(__half)p;
                {
                    uint32_t _addr_50 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_50), "f"(p16) : "memory");
                }
                float m_new_0 = __shfl_sync(0xFFFFFFFF, m_lane, 1);
                float _exp2_51 = approx_exp2(sv[1] - m_new_0);
                float p_1 = ((valid != 0) ? _exp2_51 : 0.0f);
                l_run[1] = p_1;
                m_run[1] = m_new_0;
                float p16_2 = (float)(__half)p_1;
                {
                    uint32_t _addr_52 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + 128 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + 128 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_52), "f"(p16_2) : "memory");
                }
                float m_new_3 = __shfl_sync(0xFFFFFFFF, m_lane, 2);
                float _exp2_53 = approx_exp2(sv[2] - m_new_3);
                float p_4 = ((valid != 0) ? _exp2_53 : 0.0f);
                l_run[2] = p_4;
                m_run[2] = m_new_3;
                float p16_5 = (float)(__half)p_4;
                {
                    uint32_t _addr_54 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + 256 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + 256 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_54), "f"(p16_5) : "memory");
                }
                float m_new_6 = __shfl_sync(0xFFFFFFFF, m_lane, 3);
                float _exp2_55 = approx_exp2(sv[3] - m_new_6);
                float p_7 = ((valid != 0) ? _exp2_55 : 0.0f);
                l_run[3] = p_7;
                m_run[3] = m_new_6;
                float p16_8 = (float)(__half)p_7;
                {
                    uint32_t _addr_56 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + 384 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + 384 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_56), "f"(p16_8) : "memory");
                }
                float m_new_9 = __shfl_sync(0xFFFFFFFF, m_lane, 4);
                float _exp2_57 = approx_exp2(sv[4] - m_new_9);
                float p_10 = ((valid != 0) ? _exp2_57 : 0.0f);
                l_run[4] = p_10;
                m_run[4] = m_new_9;
                float p16_11 = (float)(__half)p_10;
                {
                    uint32_t _addr_58 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + 512 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + 512 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_58), "f"(p16_11) : "memory");
                }
                float m_new_12 = __shfl_sync(0xFFFFFFFF, m_lane, 5);
                float _exp2_59 = approx_exp2(sv[5] - m_new_12);
                float p_13 = ((valid != 0) ? _exp2_59 : 0.0f);
                l_run[5] = p_13;
                m_run[5] = m_new_12;
                float p16_14 = (float)(__half)p_13;
                {
                    uint32_t _addr_60 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + 640 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + 640 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_60), "f"(p16_14) : "memory");
                }
                float m_new_15 = __shfl_sync(0xFFFFFFFF, m_lane, 6);
                float _exp2_61 = approx_exp2(sv[6] - m_new_15);
                float p_16 = ((valid != 0) ? _exp2_61 : 0.0f);
                l_run[6] = p_16;
                m_run[6] = m_new_15;
                float p16_17 = (float)(__half)p_16;
                {
                    uint32_t _addr_62 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + 768 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + 768 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_62), "f"(p16_17) : "memory");
                }
                float m_new_18 = __shfl_sync(0xFFFFFFFF, m_lane, 7);
                float _exp2_63 = approx_exp2(sv[7] - m_new_18);
                float p_19 = ((valid != 0) ? _exp2_63 : 0.0f);
                l_run[7] = p_19;
                m_run[7] = m_new_18;
                float p16_20 = (float)(__half)p_19;
                {
                    uint32_t _addr_64 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + 896 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + 896 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_64), "f"(p16_20) : "memory");
                }
                float m_new_21 = __shfl_sync(0xFFFFFFFF, m_lane, 8);
                float _exp2_65 = approx_exp2(sv[8] - m_new_21);
                float p_22 = ((valid != 0) ? _exp2_65 : 0.0f);
                l_run[8] = p_22;
                m_run[8] = m_new_21;
                float p16_23 = (float)(__half)p_22;
                {
                    uint32_t _addr_66 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + 1024 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + 1024 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_66), "f"(p16_23) : "memory");
                }
                float m_new_24 = __shfl_sync(0xFFFFFFFF, m_lane, 9);
                float _exp2_67 = approx_exp2(sv[9] - m_new_24);
                float p_25 = ((valid != 0) ? _exp2_67 : 0.0f);
                l_run[9] = p_25;
                m_run[9] = m_new_24;
                float p16_26 = (float)(__half)p_25;
                {
                    uint32_t _addr_68 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + 1152 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + 1152 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_68), "f"(p16_26) : "memory");
                }
                float m_new_27 = __shfl_sync(0xFFFFFFFF, m_lane, 10);
                float _exp2_69 = approx_exp2(sv[10] - m_new_27);
                float p_28 = ((valid != 0) ? _exp2_69 : 0.0f);
                l_run[10] = p_28;
                m_run[10] = m_new_27;
                float p16_29 = (float)(__half)p_28;
                {
                    uint32_t _addr_70 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + 1280 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + 1280 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_70), "f"(p16_29) : "memory");
                }
                float m_new_30 = __shfl_sync(0xFFFFFFFF, m_lane, 11);
                float _exp2_71 = approx_exp2(sv[11] - m_new_30);
                float p_31 = ((valid != 0) ? _exp2_71 : 0.0f);
                l_run[11] = p_31;
                m_run[11] = m_new_30;
                float p16_32 = (float)(__half)p_31;
                {
                    uint32_t _addr_72 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + 1408 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + 1408 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_72), "f"(p16_32) : "memory");
                }
                float m_new_33 = __shfl_sync(0xFFFFFFFF, m_lane, 12);
                float _exp2_73 = approx_exp2(sv[12] - m_new_33);
                float p_34 = ((valid != 0) ? _exp2_73 : 0.0f);
                l_run[12] = p_34;
                m_run[12] = m_new_33;
                float p16_35 = (float)(__half)p_34;
                {
                    uint32_t _addr_74 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + 1536 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + 1536 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_74), "f"(p16_35) : "memory");
                }
                float m_new_36 = __shfl_sync(0xFFFFFFFF, m_lane, 13);
                float _exp2_75 = approx_exp2(sv[13] - m_new_36);
                float p_37 = ((valid != 0) ? _exp2_75 : 0.0f);
                l_run[13] = p_37;
                m_run[13] = m_new_36;
                float p16_38 = (float)(__half)p_37;
                {
                    uint32_t _addr_76 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + 1664 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + 1664 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_76), "f"(p16_38) : "memory");
                }
                float m_new_39 = __shfl_sync(0xFFFFFFFF, m_lane, 14);
                float _exp2_77 = approx_exp2(sv[14] - m_new_39);
                float p_40 = ((valid != 0) ? _exp2_77 : 0.0f);
                l_run[14] = p_40;
                m_run[14] = m_new_39;
                float p16_41 = (float)(__half)p_40;
                {
                    uint32_t _addr_78 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + 1792 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + 1792 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_78), "f"(p16_41) : "memory");
                }
                float m_new_42 = __shfl_sync(0xFFFFFFFF, m_lane, 15);
                float _exp2_79 = approx_exp2(sv[15] - m_new_42);
                float p_43 = ((valid != 0) ? _exp2_79 : 0.0f);
                l_run[15] = p_43;
                m_run[15] = m_new_42;
                float p16_44 = (float)(__half)p_43;
                {
                    uint32_t _addr_80 = static_cast<uint32_t>((p_base + (unsigned int)(tid_s / 32 * 2048 + 1920 + tid_s % 32 * 4 ^ (tid_s / 32 * 2048 + 1920 + tid_s % 32 * 4 >> 7 & 7) << 4)));
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_80), "f"(p16_44) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(p_full_addr);
                }
            }
            float red[16];
            red[0] = l_run[0];
            red[1] = l_run[1];
            red[2] = l_run[2];
            red[3] = l_run[3];
            red[4] = l_run[4];
            red[5] = l_run[5];
            red[6] = l_run[6];
            red[7] = l_run[7];
            red[8] = l_run[8];
            red[9] = l_run[9];
            red[10] = l_run[10];
            red[11] = l_run[11];
            red[12] = l_run[12];
            red[13] = l_run[13];
            red[14] = l_run[14];
            red[15] = l_run[15];
            int up = lane_s & 16;
            float keep = ((up != 0) ? red[8] : red[0]);
            float send = ((up != 0) ? red[0] : red[8]);
            float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
            red[0] = keep + _shfl_xor_sync_81;
            float keep_0 = ((up != 0) ? red[9] : red[1]);
            float send_1 = ((up != 0) ? red[1] : red[9]);
            float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, send_1, 16);
            red[1] = keep_0 + _shfl_xor_sync_82;
            float keep_2 = ((up != 0) ? red[10] : red[2]);
            float send_3 = ((up != 0) ? red[2] : red[10]);
            float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, send_3, 16);
            red[2] = keep_2 + _shfl_xor_sync_83;
            float keep_4 = ((up != 0) ? red[11] : red[3]);
            float send_5 = ((up != 0) ? red[3] : red[11]);
            float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, send_5, 16);
            red[3] = keep_4 + _shfl_xor_sync_84;
            float keep_6 = ((up != 0) ? red[12] : red[4]);
            float send_7 = ((up != 0) ? red[4] : red[12]);
            float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, send_7, 16);
            red[4] = keep_6 + _shfl_xor_sync_85;
            float keep_8 = ((up != 0) ? red[13] : red[5]);
            float send_9 = ((up != 0) ? red[5] : red[13]);
            float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, send_9, 16);
            red[5] = keep_8 + _shfl_xor_sync_86;
            float keep_10 = ((up != 0) ? red[14] : red[6]);
            float send_11 = ((up != 0) ? red[6] : red[14]);
            float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, send_11, 16);
            red[6] = keep_10 + _shfl_xor_sync_87;
            float keep_12 = ((up != 0) ? red[15] : red[7]);
            float send_13 = ((up != 0) ? red[7] : red[15]);
            float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, send_13, 16);
            red[7] = keep_12 + _shfl_xor_sync_88;
            int up_14 = lane_s & 8;
            float keep_15 = ((up_14 != 0) ? red[4] : red[0]);
            float send_16 = ((up_14 != 0) ? red[0] : red[4]);
            float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, send_16, 8);
            red[0] = keep_15 + _shfl_xor_sync_89;
            float keep_17 = ((up_14 != 0) ? red[5] : red[1]);
            float send_18 = ((up_14 != 0) ? red[1] : red[5]);
            float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, send_18, 8);
            red[1] = keep_17 + _shfl_xor_sync_90;
            float keep_19 = ((up_14 != 0) ? red[6] : red[2]);
            float send_20 = ((up_14 != 0) ? red[2] : red[6]);
            float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, send_20, 8);
            red[2] = keep_19 + _shfl_xor_sync_91;
            float keep_21 = ((up_14 != 0) ? red[7] : red[3]);
            float send_22 = ((up_14 != 0) ? red[3] : red[7]);
            float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, send_22, 8);
            red[3] = keep_21 + _shfl_xor_sync_92;
            int up_23 = lane_s & 4;
            float keep_24 = ((up_23 != 0) ? red[2] : red[0]);
            float send_25 = ((up_23 != 0) ? red[0] : red[2]);
            float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, send_25, 4);
            red[0] = keep_24 + _shfl_xor_sync_93;
            float keep_26 = ((up_23 != 0) ? red[3] : red[1]);
            float send_27 = ((up_23 != 0) ? red[1] : red[3]);
            float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, send_27, 4);
            red[1] = keep_26 + _shfl_xor_sync_94;
            int up_28 = lane_s & 2;
            float keep_29 = ((up_28 != 0) ? red[1] : red[0]);
            float send_30 = ((up_28 != 0) ? red[0] : red[1]);
            float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, send_30, 2);
            red[0] = keep_29 + _shfl_xor_sync_95;
            float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, red[0], 1);
            float hsum = red[0] + _shfl_xor_sync_96;
            if ((lane_s & 1) == 0) {
                xsum[sq * 16 + (lane_s >> 1)] = hsum;
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            float inv_l = 0.0f;
            if (lane_s < 16) {
                float tot = xsum[lane_s] + xsum[16 + lane_s] + (xsum[32 + lane_s] + xsum[48 + lane_s]);
                inv_l = ((tot > 0.0f) ? out_scale / tot : 0.0f);
            }
            unsigned int zone_addr_31 = zone_addr;
            if (sq == 0) {
                float ma = __shfl_sync(0xFFFFFFFF, m_lane, 2 * lane_s & 31);
                float mb = __shfl_sync(0xFFFFFFFF, m_lane, 2 * lane_s + 1 & 31);
                if (tid_s < 8) {
                    int ha = 2 * tid_s;
                    float la = xsum[ha] + xsum[16 + ha] + (xsum[32 + ha] + xsum[48 + ha]);
                    float lb = xsum[ha + 1] + xsum[16 + ha + 1] + (xsum[32 + ha + 1] + xsum[48 + ha + 1]);
                    unsigned int ml_dst = zone_addr_31 + (unsigned int)(cta * 8320 + 8192 + tid_s * 16);
                    if (cta != 0) {
                        uint32_t _cluster_mapa_97;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_97) : "r"(ml_dst), "r"(0));
                        uint32_t _cluster_mapa_98;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_98) : "r"(ml_bar_addr), "r"(0));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_97), "r"(__float_as_uint(ma)), "r"(__float_as_uint(la)), "r"(__float_as_uint(mb)), "r"(__float_as_uint(lb)), "r"(_cluster_mapa_98) : "memory");
                    }
                    if (cta != 1) {
                        uint32_t _cluster_mapa_99;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_99) : "r"(ml_dst), "r"(1));
                        uint32_t _cluster_mapa_100;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_100) : "r"(ml_bar_addr), "r"(1));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_99), "r"(__float_as_uint(ma)), "r"(__float_as_uint(la)), "r"(__float_as_uint(mb)), "r"(__float_as_uint(lb)), "r"(_cluster_mapa_100) : "memory");
                    }
                    if (cta != 2) {
                        uint32_t _cluster_mapa_101;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_101) : "r"(ml_dst), "r"(2));
                        uint32_t _cluster_mapa_102;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_102) : "r"(ml_bar_addr), "r"(2));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_101), "r"(__float_as_uint(ma)), "r"(__float_as_uint(la)), "r"(__float_as_uint(mb)), "r"(__float_as_uint(lb)), "r"(_cluster_mapa_102) : "memory");
                    }
                    if (cta != 3) {
                        uint32_t _cluster_mapa_103;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_103) : "r"(ml_dst), "r"(3));
                        uint32_t _cluster_mapa_104;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_104) : "r"(ml_bar_addr), "r"(3));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_103), "r"(__float_as_uint(ma)), "r"(__float_as_uint(la)), "r"(__float_as_uint(mb)), "r"(__float_as_uint(lb)), "r"(_cluster_mapa_104) : "memory");
                    }
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ml_dst), "f"(ma), "f"(la), "f"(mb), "f"(lb) : "memory");
                }
                __syncwarp();
            }
            int gl = tid_s >> 3;
            int jd = tid_s & 7;
            #pragma unroll
            for (int b_1 = 0; b_1 < 4; b_1++) {
                int _stage_oblk = (int)((unsigned int)b_1 % 4);
                int _parity_oblk = (int)((unsigned int)b_1 / 4 % 2);
                mbarrier_wait(oblk_addr + (_stage_oblk) * 8, _parity_oblk);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + 192 + (unsigned int)(b_1 * 16) + lane_addr_s);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int owner = (gl + b_1) % 4;
                int lg = gl / 4;
                if (b_1 > 0) {
                    lg = lg + (16 - (owner + 64) % 4 + 3) / 4;
                }
                if (b_1 > 1) {
                    lg = lg + (16 - (owner - 1 + 64) % 4 + 3) / 4;
                }
                if (b_1 > 2) {
                    lg = lg + (16 - (owner - 2 + 64) % 4 + 3) / 4;
                }
                unsigned int pdst = zone_addr_31 + (unsigned int)(cta * 8320 + lg * 512 + jd * 16);
                if (owner != cta) {
                    uint32_t _mapa_0;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_0) : "r"(pdst), "r"(owner));
                    uint32_t _mapa_1;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_1) : "r"(zone_bar_addr), "r"(owner));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0), "r"(__float_as_uint(_tmem_load_1[0])), "r"(__float_as_uint(_tmem_load_1[1])), "r"(__float_as_uint(_tmem_load_1[2])), "r"(__float_as_uint(_tmem_load_1[3])), "r"(_mapa_1) : "memory");
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 128), "r"(__float_as_uint(_tmem_load_1[4])), "r"(__float_as_uint(_tmem_load_1[5])), "r"(__float_as_uint(_tmem_load_1[6])), "r"(__float_as_uint(_tmem_load_1[7])), "r"(_mapa_1) : "memory");
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 256), "r"(__float_as_uint(_tmem_load_1[8])), "r"(__float_as_uint(_tmem_load_1[9])), "r"(__float_as_uint(_tmem_load_1[10])), "r"(__float_as_uint(_tmem_load_1[11])), "r"(_mapa_1) : "memory");
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 384), "r"(__float_as_uint(_tmem_load_1[12])), "r"(__float_as_uint(_tmem_load_1[13])), "r"(__float_as_uint(_tmem_load_1[14])), "r"(__float_as_uint(_tmem_load_1[15])), "r"(_mapa_1) : "memory");
                } else {
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pdst), "f"(_tmem_load_1[0]), "f"(_tmem_load_1[1]), "f"(_tmem_load_1[2]), "f"(_tmem_load_1[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pdst + 128), "f"(_tmem_load_1[4]), "f"(_tmem_load_1[5]), "f"(_tmem_load_1[6]), "f"(_tmem_load_1[7]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pdst + 256), "f"(_tmem_load_1[8]), "f"(_tmem_load_1[9]), "f"(_tmem_load_1[10]), "f"(_tmem_load_1[11]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(pdst + 384), "f"(_tmem_load_1[12]), "f"(_tmem_load_1[13]), "f"(_tmem_load_1[14]), "f"(_tmem_load_1[15]) : "memory");
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(tmem_free_addr);
            }
            if (sq == 0) {
                mbarrier_wait_cluster(ml_bar_addr, 0);
            }
            if (tid_s < 16) {
                float mv[4];
                float lv[4];
                unsigned int mlw[2];
                int ml_off = 8192 + (tid_s >> 1) * 16 + (tid_s & 1) * 8;
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mlw[0])), "=r"(*reinterpret_cast<uint32_t*>(&mlw[(0) + 1]))
                    : "r"(zone_addr_31 + (unsigned int)ml_off));
                mv[0] = __uint_as_float(mlw[0]);
                lv[0] = __uint_as_float(mlw[1]);
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mlw[0])), "=r"(*reinterpret_cast<uint32_t*>(&mlw[(0) + 1]))
                    : "r"(zone_addr_31 + (unsigned int)(8320 + ml_off)));
                mv[1] = __uint_as_float(mlw[0]);
                lv[1] = __uint_as_float(mlw[1]);
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mlw[0])), "=r"(*reinterpret_cast<uint32_t*>(&mlw[(0) + 1]))
                    : "r"(zone_addr_31 + (unsigned int)(16640 + ml_off)));
                mv[2] = __uint_as_float(mlw[0]);
                lv[2] = __uint_as_float(mlw[1]);
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mlw[0])), "=r"(*reinterpret_cast<uint32_t*>(&mlw[(0) + 1]))
                    : "r"(zone_addr_31 + (unsigned int)(24960 + ml_off)));
                mv[3] = __uint_as_float(mlw[0]);
                lv[3] = __uint_as_float(mlw[1]);
                float mmax = mv[0];
                mmax = fmaxf(mmax, mv[1]);
                mmax = fmaxf(mmax, mv[2]);
                mmax = fmaxf(mmax, mv[3]);
                float lsum = 0.0f;
                float _exp2_105 = approx_exp2(mv[0] - mmax);
                float wj = ((mv[0] > -1e+29f) ? _exp2_105 : 0.0f);
                lsum = __fmaf_rn(wj, lv[0], lsum);
                wtab[tid_s] = wj;
                float _exp2_106 = approx_exp2(mv[1] - mmax);
                float wj_0 = ((mv[1] > -1e+29f) ? _exp2_106 : 0.0f);
                lsum = __fmaf_rn(wj_0, lv[1], lsum);
                wtab[16 + tid_s] = wj_0;
                float _exp2_107 = approx_exp2(mv[2] - mmax);
                float wj_1 = ((mv[2] > -1e+29f) ? _exp2_107 : 0.0f);
                lsum = __fmaf_rn(wj_1, lv[2], lsum);
                wtab[32 + tid_s] = wj_1;
                float _exp2_108 = approx_exp2(mv[3] - mmax);
                float wj_2 = ((mv[3] > -1e+29f) ? _exp2_108 : 0.0f);
                lsum = __fmaf_rn(wj_2, lv[3], lsum);
                wtab[48 + tid_s] = wj_2;
                invt[tid_s] = ((lsum > 0.0f) ? out_scale / lsum : 0.0f);
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            mbarrier_wait_cluster(zone_bar_addr, 0);
            float accv[4];
            unsigned int piece[16];
            unsigned int wq[16];
            int n_items = ((16 - (cta + 64) % 4 + 3) / 4 + (16 - (cta - 1 + 64) % 4 + 3) / 4 + (16 - (cta - 2 + 64) % 4 + 3) / 4 + (16 - (cta - 3 + 64) % 4 + 3) / 4) * 32;
            #pragma unroll 1
            for (int rr = 0; rr < 4; rr++) {
                int item = rr * 128 + tid_s;
                if (item < n_items) {
                    int lgm = item >> 5;
                    int qdm = item >> 3 & 3;
                    int jdm = item & 7;
                    int bm = 0;
                    int mm = lgm;
                    int pb = (16 - (cta + 64) % 4 + 3) / 4;
                    if (lgm >= pb) {
                        bm = 1;
                        mm = lgm - pb;
                    }
                    int pb_0 = (16 - (cta + 64) % 4 + 3) / 4 + (16 - (cta - 1 + 64) % 4 + 3) / 4;
                    if (lgm >= pb_0) {
                        bm = 2;
                        mm = lgm - pb_0;
                    }
                    int pb_1 = (16 - (cta + 64) % 4 + 3) / 4 + (16 - (cta - 1 + 64) % 4 + 3) / 4 + (16 - (cta - 2 + 64) % 4 + 3) / 4;
                    if (lgm >= pb_1) {
                        bm = 3;
                        mm = lgm - pb_1;
                    }
                    int glm = (cta - bm + 64) % 4 + 4 * mm;
                    int dimm = 128 * bm + 8 * glm + jdm;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&piece[0])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(0) + 3]))
                        : "r"(zone_addr_31 + (unsigned int)(lgm * 512 + qdm * 128 + jdm * 16)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wq[0])), "=r"(*reinterpret_cast<uint32_t*>(&wq[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wq[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wq[(0) + 3]))
                        : "r"(wtab_addr + (unsigned int)(4 * qdm * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&piece[4])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(4) + 3]))
                        : "r"(zone_addr_31 + (unsigned int)(8320 + lgm * 512 + qdm * 128 + jdm * 16)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wq[4])), "=r"(*reinterpret_cast<uint32_t*>(&wq[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wq[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wq[(4) + 3]))
                        : "r"(wtab_addr + (unsigned int)((16 + 4 * qdm) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&piece[8])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(8) + 3]))
                        : "r"(zone_addr_31 + (unsigned int)(16640 + lgm * 512 + qdm * 128 + jdm * 16)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wq[8])), "=r"(*reinterpret_cast<uint32_t*>(&wq[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wq[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wq[(8) + 3]))
                        : "r"(wtab_addr + (unsigned int)((32 + 4 * qdm) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&piece[12])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(12) + 3]))
                        : "r"(zone_addr_31 + (unsigned int)(24960 + lgm * 512 + qdm * 128 + jdm * 16)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wq[12])), "=r"(*reinterpret_cast<uint32_t*>(&wq[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wq[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wq[(12) + 3]))
                        : "r"(wtab_addr + (unsigned int)((48 + 4 * qdm) * 4)));
                    accv[0] = 0.0f;
                    accv[1] = 0.0f;
                    accv[2] = 0.0f;
                    accv[3] = 0.0f;
                    accv[0] = __fmaf_rn(__uint_as_float(piece[0]), __uint_as_float(wq[0]), accv[0]);
                    accv[1] = __fmaf_rn(__uint_as_float(piece[1]), __uint_as_float(wq[1]), accv[1]);
                    accv[2] = __fmaf_rn(__uint_as_float(piece[2]), __uint_as_float(wq[2]), accv[2]);
                    accv[3] = __fmaf_rn(__uint_as_float(piece[3]), __uint_as_float(wq[3]), accv[3]);
                    accv[0] = __fmaf_rn(__uint_as_float(piece[4]), __uint_as_float(wq[4]), accv[0]);
                    accv[1] = __fmaf_rn(__uint_as_float(piece[5]), __uint_as_float(wq[5]), accv[1]);
                    accv[2] = __fmaf_rn(__uint_as_float(piece[6]), __uint_as_float(wq[6]), accv[2]);
                    accv[3] = __fmaf_rn(__uint_as_float(piece[7]), __uint_as_float(wq[7]), accv[3]);
                    accv[0] = __fmaf_rn(__uint_as_float(piece[8]), __uint_as_float(wq[8]), accv[0]);
                    accv[1] = __fmaf_rn(__uint_as_float(piece[9]), __uint_as_float(wq[9]), accv[1]);
                    accv[2] = __fmaf_rn(__uint_as_float(piece[10]), __uint_as_float(wq[10]), accv[2]);
                    accv[3] = __fmaf_rn(__uint_as_float(piece[11]), __uint_as_float(wq[11]), accv[3]);
                    accv[0] = __fmaf_rn(__uint_as_float(piece[12]), __uint_as_float(wq[12]), accv[0]);
                    accv[1] = __fmaf_rn(__uint_as_float(piece[13]), __uint_as_float(wq[13]), accv[1]);
                    accv[2] = __fmaf_rn(__uint_as_float(piece[14]), __uint_as_float(wq[14]), accv[2]);
                    accv[3] = __fmaf_rn(__uint_as_float(piece[15]), __uint_as_float(wq[15]), accv[3]);
                    *(reinterpret_cast<__nv_bfloat16*>(out + ((token * 16 + 4 * qdm) * 512 + dimm)) + (0)) = (__nv_bfloat16)(accv[0] * invt[4 * qdm]);
                    *(reinterpret_cast<__nv_bfloat16*>(out + ((token * 16 + 4 * qdm + 1) * 512 + dimm)) + (0)) = (__nv_bfloat16)(accv[1] * invt[4 * qdm + 1]);
                    *(reinterpret_cast<__nv_bfloat16*>(out + ((token * 16 + 4 * qdm + 2) * 512 + dimm)) + (0)) = (__nv_bfloat16)(accv[2] * invt[4 * qdm + 2]);
                    *(reinterpret_cast<__nv_bfloat16*>(out + ((token * 16 + 4 * qdm + 3) * 512 + dimm)) + (0)) = (__nv_bfloat16)(accv[3] * invt[4 * qdm + 3]);
                }
            }
        }
    }
    // ---- Role: mma_k ----
    if (warp == 8) {
        { // mma_k_main
            mbarrier_wait(const_ready_addr, 0);
            mbarrier_wait(q_full_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int kc = 0;
            unsigned int kq = 0;
            unsigned int _phase_nope_full_1 = 0;
            unsigned int _phase_sfk_full = 0;
            unsigned int _phase_kr_full = 0;
            unsigned int _phase_kd_full = 0;
            if (elect_sync()) {
                #pragma unroll
                for (int t_2 = 0; t_2 < 1; t_2++) {
                    mbarrier_wait(nope_full_addr, _phase_nope_full_1);
#if __CUDA_ARCH__ == 1030
#else
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
                    mbarrier_wait(sfk_full_addr, _phase_sfk_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _smem_desc_lo_0 = (((smem_nope_p_addr) >> 4) & 0x3FFF) + ((int)0 * 2) * 1024;
                    int _smem_desc_lo_1 = (((smem_idb_addr) >> 4) & 0x3FFF) + (0) * 128;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_0) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0xC0004010 << 32);

                        tcgen05_mma_mxf4nvf4_bs((tmem_tmem_kd + ((int)kc * 64)), a_desc + 0, b_desc + 0,
                            0x8100480U, tmem_tmem_sfak + (int)0 * 32 + 0, tmem_tmem_sfbk + 0, 0);
                    }
                    tcgen05_commit(kd_full_addr + (kc) * 8);
                    kc += 1;
                    if (kc == 2) { kc = 0; }
                    int _smem_desc_lo_2 = (((smem_nope_p_addr + 32) >> 4) & 0x3FFF) + ((int)0 * 2) * 1024;
                    int _smem_desc_lo_3 = (((smem_idb_addr) >> 4) & 0x3FFF) + (0) * 128;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0xC0004010 << 32);

                        tcgen05_mma_mxf4nvf4_bs((tmem_tmem_kd + ((int)kc * 64)), a_desc + 0, b_desc + 0,
                            0x8100480U, tmem_tmem_sfak + ((int)0 * 32 + 4) + 0, tmem_tmem_sfbk + 0, 0);
                    }
                    tcgen05_commit(kd_full_addr + (kc) * 8);
                    kc += 1;
                    if (kc == 2) { kc = 0; }
                    mbarrier_wait(kr_full_addr, _phase_kr_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _smem_desc_lo_4 = (((smem_qr_addr) >> 4) & 0x3FFF) + (0) * 128;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134479888;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + ((int)0 * 16))), "r"(_smem_desc_lo_4), "r"(tmem_tmem_kr), "r"(0) : "memory");
                    _phase_kr_full ^= 1;
                    #pragma unroll 1
                    for (int p_2 = 0; p_2 < 2; p_2++) {
                        mbarrier_wait(kd_full_addr + (kq) * 8, _phase_kd_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_5 = (((smem_qnk_addr) >> 4) & 0x3FFF) + (4 * p_2) * 256;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + ((int)0 * 16))), "r"(_smem_desc_lo_5), "r"(tmem_tmem_kd + (int)kq * 64), "r"(1) : "memory");
                        kq += 1;
                        if (kq == 2) { kq = 0; _phase_kd_full ^= 1; }
                        int _smem_desc_lo_6 = (((smem_nope_p_addr + 64) >> 4) & 0x3FFF) + ((int)0 * 2 + p_2) * 1024;
                        int _smem_desc_lo_7 = (((smem_idb_addr) >> 4) & 0x3FFF) + (0) * 128;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_6) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_7) | ((uint64_t)0xC0004010 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_tmem_kd + ((int)kc * 64)), a_desc + 0, b_desc + 0,
                                0x8100480U, tmem_tmem_sfak + ((int)0 * 32 + 16 * p_2 + 8) + 0, tmem_tmem_sfbk + 0, 0);
                        }
                        tcgen05_commit(kd_full_addr + (kc) * 8);
                        kc += 1;
                        if (kc == 2) { kc = 0; }
                        mbarrier_wait(kd_full_addr + (kq) * 8, _phase_kd_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_8 = (((smem_qnk_addr) >> 4) & 0x3FFF) + (4 * p_2 + 1) * 256;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + ((int)0 * 16))), "r"(_smem_desc_lo_8), "r"(tmem_tmem_kd + (int)kq * 64), "r"(1) : "memory");
                        kq += 1;
                        if (kq == 2) { kq = 0; _phase_kd_full ^= 1; }
                        int _smem_desc_lo_9 = (((smem_nope_p_addr + 96) >> 4) & 0x3FFF) + ((int)0 * 2 + p_2) * 1024;
                        int _smem_desc_lo_10 = (((smem_idb_addr) >> 4) & 0x3FFF) + (0) * 128;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_9) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_10) | ((uint64_t)0xC0004010 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_tmem_kd + ((int)kc * 64)), a_desc + 0, b_desc + 0,
                                0x8100480U, tmem_tmem_sfak + ((int)0 * 32 + 16 * p_2 + 12) + 0, tmem_tmem_sfbk + 0, 0);
                        }
                        tcgen05_commit(kd_full_addr + (kc) * 8);
                        kc += 1;
                        if (kc == 2) { kc = 0; }
                        mbarrier_wait(kd_full_addr + (kq) * 8, _phase_kd_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_11 = (((smem_qnk_addr) >> 4) & 0x3FFF) + (4 * p_2 + 2) * 256;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + ((int)0 * 16))), "r"(_smem_desc_lo_11), "r"(tmem_tmem_kd + (int)kq * 64), "r"(1) : "memory");
                        kq += 1;
                        if (kq == 2) { kq = 0; _phase_kd_full ^= 1; }
                        if (p_2 + 1 < 2) {
                            int _smem_desc_lo_12 = (((smem_nope_p_addr) >> 4) & 0x3FFF) + ((int)0 * 2 + (p_2 + 1)) * 1024;
                            int _smem_desc_lo_13 = (((smem_idb_addr) >> 4) & 0x3FFF) + (0) * 128;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_12) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_13) | ((uint64_t)0xC0004010 << 32);

                                tcgen05_mma_mxf4nvf4_bs((tmem_tmem_kd + ((int)kc * 64)), a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfak + ((int)0 * 32 + 16 * (p_2 + 1)) + 0, tmem_tmem_sfbk + 0, 0);
                            }
                            tcgen05_commit(kd_full_addr + (kc) * 8);
                            kc += 1;
                            if (kc == 2) { kc = 0; }
                        }
                        mbarrier_wait(kd_full_addr + (kq) * 8, _phase_kd_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_14 = (((smem_qnk_addr) >> 4) & 0x3FFF) + (4 * p_2 + 3) * 256;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + ((int)0 * 16))), "r"(_smem_desc_lo_14), "r"(tmem_tmem_kd + (int)kq * 64), "r"(1) : "memory");
                        kq += 1;
                        if (kq == 2) { kq = 0; _phase_kd_full ^= 1; }
                        if (p_2 + 1 < 2) {
                            int _smem_desc_lo_15 = (((smem_nope_p_addr + 32) >> 4) & 0x3FFF) + ((int)0 * 2 + (p_2 + 1)) * 1024;
                            int _smem_desc_lo_16 = (((smem_idb_addr) >> 4) & 0x3FFF) + (0) * 128;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_15) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_16) | ((uint64_t)0xC0004010 << 32);

                                tcgen05_mma_mxf4nvf4_bs((tmem_tmem_kd + ((int)kc * 64)), a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfak + ((int)0 * 32 + 16 * (p_2 + 1) + 4) + 0, tmem_tmem_sfbk + 0, 0);
                            }
                            tcgen05_commit(kd_full_addr + (kc) * 8);
                            kc += 1;
                            if (kc == 2) { kc = 0; }
                        }
                    }
                    tcgen05_commit(s_full_addr);
                    _phase_sfk_full ^= 1;
                    _phase_nope_full_1 ^= 1;
                }
            }
        }
    }
    // ---- Role: mma_v ----
    if (warp == 9) {
        { // mma_v_main
            mbarrier_wait(const_ready_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int vs = 0;
            unsigned int vp = 0;
            unsigned int _phase_nope_full_2 = 0;
            unsigned int _phase_sfv_full = 0;
            unsigned int _phase_p_full = 0;
            unsigned int _phase_vt_full = 0;
            if (elect_sync()) {
                #pragma unroll
                for (int t_3 = 0; t_3 < 1; t_3++) {
                    mbarrier_wait(nope_full_addr, _phase_nope_full_2);
#if __CUDA_ARCH__ == 1030
#else
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
                    mbarrier_wait(sfv_full_addr, _phase_sfv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    bool pv_first = t_3 == 0;
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfbv + (int)vs * 2 * 4, make_sf_cp_desc_sbo128(smem_sfh_addr + (unsigned int)((int)0 * 2 * 4096)));
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfbv + ((int)vs * 2 + 1) * 4, make_sf_cp_desc_sbo128(smem_sfh_addr + (unsigned int)((int)0 * 2 * 4096) + 1024));
                    int _smem_desc_lo_0_1 = (((smem_nope_p_addr) >> 4) & 0x3FFF) + ((int)0 * 2) * 1024;
                    {
                        const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_0_1);
                        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_vt + ((int)vs * 64))), "r"(tmem_tmem_id), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfav), "r"(tmem_tmem_sfbv + (int)vs * 8), "r"(0) : "memory");
                    }
                    int _smem_desc_lo_1_1 = (((smem_nope_p_addr + 32) >> 4) & 0x3FFF) + ((int)0 * 2) * 1024;
                    {
                        const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_1_1);
                        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_vt + ((int)vs * 64))), "r"(tmem_tmem_id + 8), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfav), "r"(tmem_tmem_sfbv + ((int)vs * 8 + 4)), "r"(1) : "memory");
                    }
                    tcgen05_commit(vt_full_addr + (vs) * 8);
                    vs += 1;
                    if (vs == 2) { vs = 0; }
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfbv + (int)vs * 2 * 4, make_sf_cp_desc_sbo128(smem_sfh_addr + (unsigned int)((int)0 * 2 * 4096) + 512));
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfbv + ((int)vs * 2 + 1) * 4, make_sf_cp_desc_sbo128(smem_sfh_addr + (unsigned int)((int)0 * 2 * 4096) + 1536));
                    int _smem_desc_lo_2_1 = (((smem_nope_p_addr + 8192) >> 4) & 0x3FFF) + ((int)0 * 2) * 1024;
                    {
                        const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_2_1);
                        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_vt + ((int)vs * 64))), "r"(tmem_tmem_id), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfav), "r"(tmem_tmem_sfbv + (int)vs * 8), "r"(0) : "memory");
                    }
                    int _smem_desc_lo_3_1 = (((smem_nope_p_addr + 8224) >> 4) & 0x3FFF) + ((int)0 * 2) * 1024;
                    {
                        const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_3_1);
                        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_vt + ((int)vs * 64))), "r"(tmem_tmem_id + 8), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfav), "r"(tmem_tmem_sfbv + ((int)vs * 8 + 4)), "r"(1) : "memory");
                    }
                    tcgen05_commit(vt_full_addr + (vs) * 8);
                    vs += 1;
                    if (vs == 2) { vs = 0; }
                    mbarrier_wait(p_full_addr, _phase_p_full);
                    #pragma unroll 1
                    for (int q_1 = 0; q_1 < 2; q_1++) {
                        mbarrier_wait(vt_full_addr + (vp) * 8, _phase_vt_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_4_1 = (((smem_ph_addr) >> 4) & 0x3FFF) + ((int)0 * 2) * 256;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_o + (2 * q_1 * 16))), "r"(_smem_desc_lo_4_1), "r"(tmem_tmem_vt + (int)vp * 64), "r"(((pv_first) ? 0 : 1)) : "memory");
                        vp += 1;
                        if (vp == 2) { vp = 0; _phase_vt_full ^= 1; }
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfbv + (int)vs * 2 * 4, make_sf_cp_desc_sbo128(smem_sfh_addr + (unsigned int)(((int)0 * 2 + q_1) * 4096) + 2048));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfbv + ((int)vs * 2 + 1) * 4, make_sf_cp_desc_sbo128(smem_sfh_addr + (unsigned int)(((int)0 * 2 + q_1) * 4096) + 3072));
                        int _smem_desc_lo_5_1 = (((smem_nope_p_addr + 64) >> 4) & 0x3FFF) + ((int)0 * 2 + q_1) * 1024;
                        {
                            const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_5_1);
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_vt + ((int)vs * 64))), "r"(tmem_tmem_id), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfav), "r"(tmem_tmem_sfbv + (int)vs * 8), "r"(0) : "memory");
                        }
                        int _smem_desc_lo_6_1 = (((smem_nope_p_addr + 96) >> 4) & 0x3FFF) + ((int)0 * 2 + q_1) * 1024;
                        {
                            const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_6_1);
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_vt + ((int)vs * 64))), "r"(tmem_tmem_id + 8), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfav), "r"(tmem_tmem_sfbv + ((int)vs * 8 + 4)), "r"(1) : "memory");
                        }
                        tcgen05_commit(vt_full_addr + (vs) * 8);
                        vs += 1;
                        if (vs == 2) { vs = 0; }
                        mbarrier_wait(vt_full_addr + (vp) * 8, _phase_vt_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_7_1 = (((smem_ph_addr) >> 4) & 0x3FFF) + ((int)0 * 2 + 1) * 256;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_o + (2 * q_1 * 16))), "r"(_smem_desc_lo_7_1), "r"(tmem_tmem_vt + (int)vp * 64), "r"(1) : "memory");
                        vp += 1;
                        if (vp == 2) { vp = 0; _phase_vt_full ^= 1; }
                        tcgen05_commit(oblk_addr + (2 * q_1) * 8);
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfbv + (int)vs * 2 * 4, make_sf_cp_desc_sbo128(smem_sfh_addr + (unsigned int)(((int)0 * 2 + q_1) * 4096) + 2560));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfbv + ((int)vs * 2 + 1) * 4, make_sf_cp_desc_sbo128(smem_sfh_addr + (unsigned int)(((int)0 * 2 + q_1) * 4096) + 3584));
                        int _smem_desc_lo_8_1 = (((smem_nope_p_addr + 8256) >> 4) & 0x3FFF) + ((int)0 * 2 + q_1) * 1024;
                        {
                            const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_8_1);
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_vt + ((int)vs * 64))), "r"(tmem_tmem_id), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfav), "r"(tmem_tmem_sfbv + (int)vs * 8), "r"(0) : "memory");
                        }
                        int _smem_desc_lo_9_1 = (((smem_nope_p_addr + 8288) >> 4) & 0x3FFF) + ((int)0 * 2 + q_1) * 1024;
                        {
                            const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_9_1);
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_vt + ((int)vs * 64))), "r"(tmem_tmem_id + 8), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfav), "r"(tmem_tmem_sfbv + ((int)vs * 8 + 4)), "r"(1) : "memory");
                        }
                        tcgen05_commit(vt_full_addr + (vs) * 8);
                        vs += 1;
                        if (vs == 2) { vs = 0; }
                        mbarrier_wait(vt_full_addr + (vp) * 8, _phase_vt_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_10_1 = (((smem_ph_addr) >> 4) & 0x3FFF) + ((int)0 * 2) * 256;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_o + ((2 * q_1 + 1) * 16))), "r"(_smem_desc_lo_10_1), "r"(tmem_tmem_vt + (int)vp * 64), "r"(((pv_first) ? 0 : 1)) : "memory");
                        vp += 1;
                        if (vp == 2) { vp = 0; _phase_vt_full ^= 1; }
                        if (q_1 + 1 < 2) {
                            tcgen05_cp_32x128b_warpx4(tmem_tmem_sfbv + (int)vs * 2 * 4, make_sf_cp_desc_sbo128(smem_sfh_addr + (unsigned int)(((int)0 * 2 + (q_1 + 1)) * 4096)));
                            tcgen05_cp_32x128b_warpx4(tmem_tmem_sfbv + ((int)vs * 2 + 1) * 4, make_sf_cp_desc_sbo128(smem_sfh_addr + (unsigned int)(((int)0 * 2 + (q_1 + 1)) * 4096) + 1024));
                            int _smem_desc_lo_11_1 = (((smem_nope_p_addr) >> 4) & 0x3FFF) + ((int)0 * 2 + (q_1 + 1)) * 1024;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_11_1);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_vt + ((int)vs * 64))), "r"(tmem_tmem_id), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfav), "r"(tmem_tmem_sfbv + (int)vs * 8), "r"(0) : "memory");
                            }
                            int _smem_desc_lo_12_1 = (((smem_nope_p_addr + 32) >> 4) & 0x3FFF) + ((int)0 * 2 + (q_1 + 1)) * 1024;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_12_1);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_vt + ((int)vs * 64))), "r"(tmem_tmem_id + 8), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfav), "r"(tmem_tmem_sfbv + ((int)vs * 8 + 4)), "r"(1) : "memory");
                            }
                            tcgen05_commit(vt_full_addr + (vs) * 8);
                            vs += 1;
                            if (vs == 2) { vs = 0; }
                        }
                        mbarrier_wait(vt_full_addr + (vp) * 8, _phase_vt_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_13_1 = (((smem_ph_addr) >> 4) & 0x3FFF) + ((int)0 * 2 + 1) * 256;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134482192;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_o + ((2 * q_1 + 1) * 16))), "r"(_smem_desc_lo_13_1), "r"(tmem_tmem_vt + (int)vp * 64), "r"(1) : "memory");
                        vp += 1;
                        if (vp == 2) { vp = 0; _phase_vt_full ^= 1; }
                        tcgen05_commit(oblk_addr + (2 * q_1 + 1) * 8);
                        if (q_1 + 1 < 2) {
                            tcgen05_cp_32x128b_warpx4(tmem_tmem_sfbv + (int)vs * 2 * 4, make_sf_cp_desc_sbo128(smem_sfh_addr + (unsigned int)(((int)0 * 2 + (q_1 + 1)) * 4096) + 512));
                            tcgen05_cp_32x128b_warpx4(tmem_tmem_sfbv + ((int)vs * 2 + 1) * 4, make_sf_cp_desc_sbo128(smem_sfh_addr + (unsigned int)(((int)0 * 2 + (q_1 + 1)) * 4096) + 1536));
                            int _smem_desc_lo_14_1 = (((smem_nope_p_addr + 8192) >> 4) & 0x3FFF) + ((int)0 * 2 + (q_1 + 1)) * 1024;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_14_1);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_vt + ((int)vs * 64))), "r"(tmem_tmem_id), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfav), "r"(tmem_tmem_sfbv + (int)vs * 8), "r"(0) : "memory");
                            }
                            int _smem_desc_lo_15_1 = (((smem_nope_p_addr + 8224) >> 4) & 0x3FFF) + ((int)0 * 2 + (q_1 + 1)) * 1024;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_15_1);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_vt + ((int)vs * 64))), "r"(tmem_tmem_id + 8), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfav), "r"(tmem_tmem_sfbv + ((int)vs * 8 + 4)), "r"(1) : "memory");
                            }
                            tcgen05_commit(vt_full_addr + (vs) * 8);
                            vs += 1;
                            if (vs == 2) { vs = 0; }
                        }
                    }
                    _phase_sfv_full ^= 1;
                    _phase_nope_full_2 ^= 1;
                    _phase_p_full ^= 1;
                }
            }
            mbarrier_wait(tmem_free_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
    // ---- Role: io ----
    if (warp >= 12 && warp <= 13) {
        { // io_main
            int lane_i = lane;
            int row_i = (warp & 1) * 64 + lane_i * 4;
            int ic[4];
            int inx_1[4];
        }
    }
    // ---- Role: cp0 ----
    if (warp == 10 || warp == 11 || warp == 14 || warp == 15) {
        { // cp0_main
            __syncwarp();
        }
    }

    // Cleanup
}

} // extern "C"
