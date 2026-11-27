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





__device__ __forceinline__ void tcgen05_mma_mxf8f6f4_bs_k64_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
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


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
}





__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xFFFFFFFF;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(224, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_batch_deepgemm_fp8_6a369701cc56c41c807e(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, int* __restrict__ SFA, int* __restrict__ SFB, int* __restrict__ masked_m, const __grid_constant__ CUtensorMap C, __nv_bfloat16* __restrict__ C_rows, int G, int M, int N, int K, CakeFastDivmod fd_w)
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
    #define af_addr (mbar_base + 0)
    #define sf_addr (mbar_base + 72)
    #define ae_addr (mbar_base + 144)
    #define cf_addr (mbar_base + 216)
    #define ce_addr (mbar_base + 232)
    #define tm_ready_addr (mbar_base + 248)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 256);
    uint8_t* sm_w = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int sm_w_addr = smem + 1024;
    uint8_t* sm_t = reinterpret_cast<uint8_t*>(smem_raw + 148480);
    const int sm_t_addr = smem + 148480;
    __nv_bfloat16* _sm_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 295936);
    const int sm_c_addr = smem + 295936;
    uint8_t* _sm_wsf = reinterpret_cast<uint8_t*>(smem_raw + 312320);
    const int sm_wsf_addr = smem + 312320;
    uint8_t* _sm_tsf = reinterpret_cast<uint8_t*>(smem_raw + 316928);
    const int sm_tsf_addr = smem + 316928;
    int* plist = reinterpret_cast<int*>(smem_raw + 326144);
    const int _plist_addr = smem + 326144;

    // Mbarrier init (32 barriers at smem_raw[0..256))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ap' ---
            // af: 9 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // sf: 9 barriers, init_count=2
            mbarrier_init(smem + 72, 2);
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            mbarrier_init(smem + 96, 2);
            mbarrier_init(smem + 104, 2);
            mbarrier_init(smem + 112, 2);
            mbarrier_init(smem + 120, 2);
            mbarrier_init(smem + 128, 2);
            mbarrier_init(smem + 136, 2);
            // ae: 9 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // --- pipeline 'cp' ---
            // cf: 2 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // ce: 2 barriers, init_count=8
            mbarrier_init(smem + 232, 8);
            mbarrier_init(smem + 240, 8);
            // tm_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 248, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    if (warp == 0) {
        int done = 0;
        #pragma unroll 1
        for (int r = 0; r < G; r += 32) {
            int gi = r + lane;
            int raw = masked_m[((gi < G) ? gi : 0)];
            int _max_1 = ((raw) > (0) ? (raw) : (0));
            int _min_0 = ((_max_1) < (M) ? (_max_1) : (M));
            int rows = ((gi < G) ? _min_0 : 0);
            int npass = (rows + 255) / 256;
            int _warp_scan_sum_2 = npass;
            {
                uint32_t _warp_scan_sum_2_scan = (uint32_t)_warp_scan_sum_2;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2_scan, 1, 32); if (lane >= 1) { _warp_scan_sum_2_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2_scan, 2, 32); if (lane >= 2) { _warp_scan_sum_2_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2_scan, 4, 32); if (lane >= 4) { _warp_scan_sum_2_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2_scan, 8, 32); if (lane >= 8) { _warp_scan_sum_2_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2_scan, 16, 32); if (lane >= 16) { _warp_scan_sum_2_scan += peer; } }
                _warp_scan_sum_2_scan -= (uint32_t)_warp_scan_sum_2;
                _warp_scan_sum_2 = (int)_warp_scan_sum_2_scan;
            }
            int first = done + _warp_scan_sum_2;
            if (gi < G) {
                plist[gi] = first;
                plist[1025 + gi] = rows;
            }
            int _warp_redux_3;
            asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_3) : "r"(npass));
            done = done + _warp_redux_3;
        }
        if (lane == 0) {
            plist[G] = done;
        }
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    const unsigned int taddr = 0;
    const unsigned int accum = taddr;
    const unsigned int tm_wsf = taddr + 512;
    const unsigned int tm_tsf = taddr + 520;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            unsigned int acs = 0;
            const int epi_warp = warp % 4;
            int epi_wg = warp / 4;
            int v_e = cta_rank & 1;
            int leader = cta_rank - v_e;
            int q_row = lane / 4;
            int q_chunk = lane % 4;
            int n_items = N / 256 * plist[G];
            unsigned int _phase_cf = 0;
            #pragma unroll 1
            for (int u = (int)cluster_id; u < n_items; u += (int)num_clusters) {
                uint32_t su = (fd_w.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)u), fd_w.multiplier) >> fd_w.shift_right) : (uint32_t)((unsigned int)u);
                uint32_t wu = (uint32_t)((unsigned int)u) - su * (uint32_t)(fd_w.divisor);
                int grp = 0;
                int cand_30 = grp + 512;
                int _min_0_1 = ((cand_30) < (G) ? (cand_30) : (G));
                int cand_first_30 = plist[_min_0_1];
                grp = ((cand_first_30 <= (int)su) ? cand_30 : grp);
                int cand_31 = grp + 256;
                int _min_1 = ((cand_31) < (G) ? (cand_31) : (G));
                int cand_first_31 = plist[_min_1];
                grp = ((cand_first_31 <= (int)su) ? cand_31 : grp);
                int cand_32 = grp + 128;
                int _min_2 = ((cand_32) < (G) ? (cand_32) : (G));
                int cand_first_32 = plist[_min_2];
                grp = ((cand_first_32 <= (int)su) ? cand_32 : grp);
                int cand_33 = grp + 64;
                int _min_3 = ((cand_33) < (G) ? (cand_33) : (G));
                int cand_first_33 = plist[_min_3];
                grp = ((cand_first_33 <= (int)su) ? cand_33 : grp);
                int cand_34 = grp + 32;
                int _min_4 = ((cand_34) < (G) ? (cand_34) : (G));
                int cand_first_34 = plist[_min_4];
                grp = ((cand_first_34 <= (int)su) ? cand_34 : grp);
                int cand_35 = grp + 16;
                int _min_5 = ((cand_35) < (G) ? (cand_35) : (G));
                int cand_first_35 = plist[_min_5];
                grp = ((cand_first_35 <= (int)su) ? cand_35 : grp);
                int cand_36 = grp + 8;
                int _min_6 = ((cand_36) < (G) ? (cand_36) : (G));
                int cand_first_36 = plist[_min_6];
                grp = ((cand_first_36 <= (int)su) ? cand_36 : grp);
                int cand_37 = grp + 4;
                int _min_7 = ((cand_37) < (G) ? (cand_37) : (G));
                int cand_first_37 = plist[_min_7];
                grp = ((cand_first_37 <= (int)su) ? cand_37 : grp);
                int cand_38 = grp + 2;
                int _min_8 = ((cand_38) < (G) ? (cand_38) : (G));
                int cand_first_38 = plist[_min_8];
                grp = ((cand_first_38 <= (int)su) ? cand_38 : grp);
                int cand_39 = grp + 1;
                int _min_9 = ((cand_39) < (G) ? (cand_39) : (G));
                int cand_first_39 = plist[_min_9];
                grp = ((cand_first_39 <= (int)su) ? cand_39 : grp);
                int pj = (int)su - plist[grp];
                int p_left = plist[1025 + grp] - pj * 256;
                int p_ntok = ((p_left > 128) ? 256 : 128);
                int row0 = make_warp_uniform(pj * 2 * 128);
                int n0 = make_warp_uniform((int)wu * 256 + v_e * 128);
                int ge = make_warp_uniform(grp);
                int strips = p_ntok / 16;
                mbarrier_wait_cluster_hint(cf_addr + (acs) * 8, _phase_cf, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll 1
                for (int strip_0 = epi_wg; strip_0 < strips; strip_0++) {
                    unsigned int slot_0 = (unsigned int)(strip_0 % 4);
                    int valid_0 = p_left - strip_0 * 16;
                    asm volatile("cp.async.bulk.wait_group.read 3;");
                    __syncwarp();
                    if (valid_0 > 0) {
                        unsigned int s_base_0 = sm_c_addr + (unsigned int)(warp * 4 * 1024) + slot_0 * 1024;
                        int t_addr_0 = taddr + (unsigned int)(epi_warp * 32 << 16) + acs * 256 + (unsigned int)(strip_0 * 16);
                        float acc_0[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&acc_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc_0[3]))
                            : "r"(t_addr_0));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&acc_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc_0[7]))
                            : "r"(t_addr_0 + 1048576));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&acc_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc_0[11]))
                            : "r"(t_addr_0 + 8));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&acc_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc_0[15]))
                            : "r"(t_addr_0 + 8 + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        __nv_bfloat162 p0_0 = __float22bfloat162_rn(make_float2(acc_0[0], acc_0[1]));
                        __nv_bfloat162 p1_0 = __float22bfloat162_rn(make_float2(acc_0[2], acc_0[3]));
                        __nv_bfloat162 p2_0 = __float22bfloat162_rn(make_float2(acc_0[4], acc_0[5]));
                        __nv_bfloat162 p3_0 = __float22bfloat162_rn(make_float2(acc_0[6], acc_0[7]));
                        int row_0 = lane % 8;
                        int chunk_0 = lane / 8;
                        uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(s_base_0 + (unsigned int)(row_0 * 64 + (chunk_0 ^ row_0 >> 1 & 3) * 16));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&p0_0)), "r"(*reinterpret_cast<const uint32_t*>(&p1_0)), "r"(*reinterpret_cast<const uint32_t*>(&p2_0)), "r"(*reinterpret_cast<const uint32_t*>(&p3_0))
                            : "memory");
                        __nv_bfloat162 p0_1 = __float22bfloat162_rn(make_float2(acc_0[8], acc_0[9]));
                        __nv_bfloat162 p1_1 = __float22bfloat162_rn(make_float2(acc_0[10], acc_0[11]));
                        __nv_bfloat162 p2_1 = __float22bfloat162_rn(make_float2(acc_0[12], acc_0[13]));
                        __nv_bfloat162 p3_1 = __float22bfloat162_rn(make_float2(acc_0[14], acc_0[15]));
                        int row_1 = 8 + lane % 8;
                        int chunk_1 = lane / 8;
                        uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(s_base_0 + (unsigned int)(row_1 * 64 + (chunk_1 ^ row_1 >> 1 & 3) * 16));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&p0_1)), "r"(*reinterpret_cast<const uint32_t*>(&p1_1)), "r"(*reinterpret_cast<const uint32_t*>(&p2_1)), "r"(*reinterpret_cast<const uint32_t*>(&p3_1))
                            : "memory");
                        if (valid_0 >= 16) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            __syncwarp();
                            if (elect_sync()) {
                                tma_store_3d((&C), n0 + epi_warp * 32, row0 + strip_0 * 16, ge, s_base_0);
                            }
                        } else {
                            __syncwarp();
                            int qr_0 = q_row;
                            unsigned int qv_0[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qv_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&qv_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qv_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qv_0[(0) + 3]))
                                : "r"(s_base_0 + (unsigned int)(qr_0 * 64 + (q_chunk ^ qr_0 >> 1 & 3) * 16)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(C_rows + (((long long)ge * (long long)M + (long long)(row0 + strip_0 * 16 + qr_0)) * (long long)N + (long long)(n0 + epi_warp * 32 + q_chunk * 8))), "r"(qv_0[(0) + 0]), "r"(qv_0[(0) + 1]), "r"(qv_0[(0) + 2]), "r"(qv_0[(0) + 3]), "r"((unsigned int)(qr_0 < valid_0)) : "memory");
                            int qr_1 = q_row + 8;
                            unsigned int qv_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qv_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qv_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qv_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qv_1[(0) + 3]))
                                : "r"(s_base_0 + (unsigned int)(qr_1 * 64 + (q_chunk ^ qr_1 >> 1 & 3) * 16)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(C_rows + (((long long)ge * (long long)M + (long long)(row0 + strip_0 * 16 + qr_1)) * (long long)N + (long long)(n0 + epi_warp * 32 + q_chunk * 8))), "r"(qv_1[(0) + 0]), "r"(qv_1[(0) + 1]), "r"(qv_1[(0) + 2]), "r"(qv_1[(0) + 3]), "r"((unsigned int)(qr_1 < valid_0)) : "memory");
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(ce_addr + (acs) * 8), "r"(leader) : "memory");
                }
                acs += 1;
                if (acs == 2) { acs = 0; _phase_cf ^= 1; }
            }
            asm volatile("cp.async.bulk.wait_group.read 0;");
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                mbarrier_wait(tm_ready_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.dealloc.exclusive.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(576));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            unsigned int stage = 0;
            unsigned int acs_1 = 0;
            int k_tiles_m = K / 128;
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            int _tmem_alloc_hold = smem + 256;
            asm volatile("tcgen05.alloc.exclusive.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_alloc_hold), "r"(576) : "memory");
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (elect_sync()) {
                mbarrier_arrive(tm_ready_addr);
            }
            unsigned int _phase_ce = 1;
            unsigned int _phase_af = 0;
            if ((cta_rank & 1) == 0) {
                int n_items_1 = N / 256 * plist[G];
                #pragma unroll 1
                for (int u_1 = (int)cluster_id; u_1 < n_items_1; u_1 += (int)num_clusters) {
                    uint32_t su_1 = (fd_w.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)u_1), fd_w.multiplier) >> fd_w.shift_right) : (uint32_t)((unsigned int)u_1);
                    uint32_t wu_1 = (uint32_t)((unsigned int)u_1) - su_1 * (uint32_t)(fd_w.divisor);
                    int grp_1 = 0;
                    int cand_20 = grp_1 + 512;
                    int _min_0_2 = ((cand_20) < (G) ? (cand_20) : (G));
                    int cand_first_20 = plist[_min_0_2];
                    grp_1 = ((cand_first_20 <= (int)su_1) ? cand_20 : grp_1);
                    int cand_21 = grp_1 + 256;
                    int _min_1_1 = ((cand_21) < (G) ? (cand_21) : (G));
                    int cand_first_21 = plist[_min_1_1];
                    grp_1 = ((cand_first_21 <= (int)su_1) ? cand_21 : grp_1);
                    int cand_22 = grp_1 + 128;
                    int _min_2_1 = ((cand_22) < (G) ? (cand_22) : (G));
                    int cand_first_22 = plist[_min_2_1];
                    grp_1 = ((cand_first_22 <= (int)su_1) ? cand_22 : grp_1);
                    int cand_23 = grp_1 + 64;
                    int _min_3_1 = ((cand_23) < (G) ? (cand_23) : (G));
                    int cand_first_23 = plist[_min_3_1];
                    grp_1 = ((cand_first_23 <= (int)su_1) ? cand_23 : grp_1);
                    int cand_24 = grp_1 + 32;
                    int _min_4_1 = ((cand_24) < (G) ? (cand_24) : (G));
                    int cand_first_24 = plist[_min_4_1];
                    grp_1 = ((cand_first_24 <= (int)su_1) ? cand_24 : grp_1);
                    int cand_25 = grp_1 + 16;
                    int _min_5_1 = ((cand_25) < (G) ? (cand_25) : (G));
                    int cand_first_25 = plist[_min_5_1];
                    grp_1 = ((cand_first_25 <= (int)su_1) ? cand_25 : grp_1);
                    int cand_26 = grp_1 + 8;
                    int _min_6_1 = ((cand_26) < (G) ? (cand_26) : (G));
                    int cand_first_26 = plist[_min_6_1];
                    grp_1 = ((cand_first_26 <= (int)su_1) ? cand_26 : grp_1);
                    int cand_27 = grp_1 + 4;
                    int _min_7_1 = ((cand_27) < (G) ? (cand_27) : (G));
                    int cand_first_27 = plist[_min_7_1];
                    grp_1 = ((cand_first_27 <= (int)su_1) ? cand_27 : grp_1);
                    int cand_28 = grp_1 + 2;
                    int _min_8_1 = ((cand_28) < (G) ? (cand_28) : (G));
                    int cand_first_28 = plist[_min_8_1];
                    grp_1 = ((cand_first_28 <= (int)su_1) ? cand_28 : grp_1);
                    int cand_29 = grp_1 + 1;
                    int _min_9_1 = ((cand_29) < (G) ? (cand_29) : (G));
                    int cand_first_29 = plist[_min_9_1];
                    grp_1 = ((cand_first_29 <= (int)su_1) ? cand_29 : grp_1);
                    int pj_1 = (int)su_1 - plist[grp_1];
                    int p_left_1 = plist[1025 + grp_1] - pj_1 * 256;
                    int p_ntok_1 = ((p_left_1 > 128) ? 256 : 128);
                    mbarrier_wait_cluster_hint(ce_addr + (acs_1) * 8, _phase_ce, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int acc_col = acs_1 * 256;
                    #pragma unroll 1
                    for (int kt = 0; kt < k_tiles_m; kt++) {
                        mbarrier_wait_cluster_hint(af_addr + (stage) * 8, _phase_af, 10000000);
                        mbarrier_wait_cluster_hint(sf_addr + (stage) * 8, _phase_af, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_10 = elect_sync();
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(sm_wsf_addr + stage * 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(taddr + 512)), "l"(_tcgen05_cp_desc_11), "r"(_elected_10)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(sm_tsf_addr + stage * 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(taddr + 512 + 8)), "l"(_tcgen05_cp_desc_12), "r"(_elected_10)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(sm_tsf_addr + stage * 1024 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(taddr + 512 + 8 + 4)), "l"(_tcgen05_cp_desc_13), "r"(_elected_10)
                                : "memory");
                        }
                        int t_lo = make_warp_uniform((((sm_t_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                        int w_lo_0 = make_warp_uniform((((sm_w_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)w_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)t_lo) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8f6f4_bs_k64_cta2((accum + (acc_col)), a_desc + 0, b_desc + 0,
                                    ((0x90c00000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(p_ntok_1) >> 3) << 17)), (int)((tm_wsf + 0 + 0)), (int)((tm_tsf + 0)), ((kt == 0) ? 0 : 1));
                                tcgen05_mma_mxf8f6f4_bs_k64_cta2((accum + (acc_col)), a_desc + 4, b_desc + 4,
                                    ((0xd0c00020U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(p_ntok_1) >> 3) << 17)), (int)((tm_wsf + 0 + 0) | 0x80000000), (int)((tm_tsf + 0) | 0x80000000), 1);
                            }
                        }
                        elect_commit_cg2_multicast(ae_addr + (stage) * 8, (uint16_t)(3));
                        stage += 1;
                        if (stage == 9) { stage = 0; _phase_af ^= 1; }
                    }
                    elect_commit_cg2_multicast(cf_addr + (acs_1) * 8, (uint16_t)(3));
                    acs_1 += 1;
                    if (acs_1 == 2) { acs_1 = 0; _phase_ce ^= 1; }
                }
                #pragma unroll
                for (int _xe = 0; _xe < 2; _xe++) {
                    mbarrier_wait_cluster_hint(ce_addr + (acs_1) * 8, _phase_ce, 10000000);
                    acs_1 += 1;
                    if (acs_1 == 2) { acs_1 = 0; _phase_ce ^= 1; }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 5) {
        { // load_main
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C))) : "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            unsigned int stage_1 = 0;
            int v = cta_rank & 1;
            int k_tiles = K / 128;
            int n_items_2 = N / 256 * plist[G];
            unsigned int _phase_ae = 1;
            #pragma unroll 1
            for (int u_2 = (int)cluster_id; u_2 < n_items_2; u_2 += (int)num_clusters) {
                uint32_t su_2 = (fd_w.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)u_2), fd_w.multiplier) >> fd_w.shift_right) : (uint32_t)((unsigned int)u_2);
                uint32_t wu_2 = (uint32_t)((unsigned int)u_2) - su_2 * (uint32_t)(fd_w.divisor);
                int grp_2 = 0;
                int cand_0 = grp_2 + 512;
                int _min_0_3 = ((cand_0) < (G) ? (cand_0) : (G));
                int cand_first_0 = plist[_min_0_3];
                grp_2 = ((cand_first_0 <= (int)su_2) ? cand_0 : grp_2);
                int cand_1 = grp_2 + 256;
                int _min_1_2 = ((cand_1) < (G) ? (cand_1) : (G));
                int cand_first_1 = plist[_min_1_2];
                grp_2 = ((cand_first_1 <= (int)su_2) ? cand_1 : grp_2);
                int cand_2 = grp_2 + 128;
                int _min_2_2 = ((cand_2) < (G) ? (cand_2) : (G));
                int cand_first_2 = plist[_min_2_2];
                grp_2 = ((cand_first_2 <= (int)su_2) ? cand_2 : grp_2);
                int cand_3 = grp_2 + 64;
                int _min_3_2 = ((cand_3) < (G) ? (cand_3) : (G));
                int cand_first_3 = plist[_min_3_2];
                grp_2 = ((cand_first_3 <= (int)su_2) ? cand_3 : grp_2);
                int cand_4 = grp_2 + 32;
                int _min_4_2 = ((cand_4) < (G) ? (cand_4) : (G));
                int cand_first_4 = plist[_min_4_2];
                grp_2 = ((cand_first_4 <= (int)su_2) ? cand_4 : grp_2);
                int cand_5 = grp_2 + 16;
                int _min_5_2 = ((cand_5) < (G) ? (cand_5) : (G));
                int cand_first_5 = plist[_min_5_2];
                grp_2 = ((cand_first_5 <= (int)su_2) ? cand_5 : grp_2);
                int cand_6 = grp_2 + 8;
                int _min_6_2 = ((cand_6) < (G) ? (cand_6) : (G));
                int cand_first_6 = plist[_min_6_2];
                grp_2 = ((cand_first_6 <= (int)su_2) ? cand_6 : grp_2);
                int cand_7 = grp_2 + 4;
                int _min_7_2 = ((cand_7) < (G) ? (cand_7) : (G));
                int cand_first_7 = plist[_min_7_2];
                grp_2 = ((cand_first_7 <= (int)su_2) ? cand_7 : grp_2);
                int cand_8 = grp_2 + 2;
                int _min_8_2 = ((cand_8) < (G) ? (cand_8) : (G));
                int cand_first_8 = plist[_min_8_2];
                grp_2 = ((cand_first_8 <= (int)su_2) ? cand_8 : grp_2);
                int cand_9 = grp_2 + 1;
                int _min_9_2 = ((cand_9) < (G) ? (cand_9) : (G));
                int cand_first_9 = plist[_min_9_2];
                grp_2 = ((cand_first_9 <= (int)su_2) ? cand_9 : grp_2);
                int pj_2 = (int)su_2 - plist[grp_2];
                int p_left_2 = plist[1025 + grp_2] - pj_2 * 256;
                int p_ntok_2 = ((p_left_2 > 128) ? 256 : 128);
                int w_row = make_warp_uniform((int)wu_2 * 256 + v * 128);
                int t_row = make_warp_uniform(pj_2 * 2 * 128 + v * (p_ntok_2 / 2));
                int g = make_warp_uniform(grp_2);
                int boxes = p_ntok_2 / 128;
                int tx = 32768 + p_ntok_2 * 128;
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles; kt_1++) {
                    mbarrier_wait_cluster_hint(ae_addr + (stage_1) * 8, _phase_ae, 10000000);
                    if (elect_sync()) {
                        if (v == 0) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((af_addr + (stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(tx)) : "memory");
                        }
                        tma_3d_gmem2smem_cta2(sm_w_addr + stage_1 * 16384, (&B), kt_1 * 128, w_row, g, ((af_addr + (stage_1) * 8) & 0xFEFFFFFF));
                        #pragma unroll 1
                        for (int bx = 0; bx < boxes; bx++) {
                            tma_3d_gmem2smem_cta2(sm_t_addr + stage_1 * 16384 + (unsigned int)(bx * 8192), (&A), kt_1 * 128, t_row + bx * 64, g, ((af_addr + (stage_1) * 8) & 0xFEFFFFFF));
                        }
                    }
                    stage_1 += 1;
                    if (stage_1 == 9) { stage_1 = 0; _phase_ae ^= 1; }
                }
            }
            #pragma unroll 1
            for (int _s = 0; _s < 9; _s++) {
                mbarrier_wait_cluster_hint(ae_addr + (stage_1) * 8, _phase_ae, 10000000);
                stage_1 += 1;
                if (stage_1 == 9) { stage_1 = 0; _phase_ae ^= 1; }
            }
        }
    }
    // ---- Role: scales ----
    if (warp == 6) {
        { // scales_main
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            unsigned int stage_2 = 0;
            int k_tiles_s = K / 128;
            int cols = (K + 511) / 512;
            int lane_1 = lane;
            unsigned int words[12];
            int v_s = cta_rank & 1;
            int leader_s = cta_rank - v_s;
            int n_items_3 = N / 256 * plist[G];
            unsigned int _phase_ae_1 = 1;
            #pragma unroll 1
            for (int u_3 = (int)cluster_id; u_3 < n_items_3; u_3 += (int)num_clusters) {
                uint32_t su_3 = (fd_w.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)u_3), fd_w.multiplier) >> fd_w.shift_right) : (uint32_t)((unsigned int)u_3);
                uint32_t wu_3 = (uint32_t)((unsigned int)u_3) - su_3 * (uint32_t)(fd_w.divisor);
                int grp_3 = 0;
                int cand_10 = grp_3 + 512;
                int _min_0_4 = ((cand_10) < (G) ? (cand_10) : (G));
                int cand_first_10 = plist[_min_0_4];
                grp_3 = ((cand_first_10 <= (int)su_3) ? cand_10 : grp_3);
                int cand_11 = grp_3 + 256;
                int _min_1_3 = ((cand_11) < (G) ? (cand_11) : (G));
                int cand_first_11 = plist[_min_1_3];
                grp_3 = ((cand_first_11 <= (int)su_3) ? cand_11 : grp_3);
                int cand_12 = grp_3 + 128;
                int _min_2_3 = ((cand_12) < (G) ? (cand_12) : (G));
                int cand_first_12 = plist[_min_2_3];
                grp_3 = ((cand_first_12 <= (int)su_3) ? cand_12 : grp_3);
                int cand_13 = grp_3 + 64;
                int _min_3_3 = ((cand_13) < (G) ? (cand_13) : (G));
                int cand_first_13 = plist[_min_3_3];
                grp_3 = ((cand_first_13 <= (int)su_3) ? cand_13 : grp_3);
                int cand_14 = grp_3 + 32;
                int _min_4_3 = ((cand_14) < (G) ? (cand_14) : (G));
                int cand_first_14 = plist[_min_4_3];
                grp_3 = ((cand_first_14 <= (int)su_3) ? cand_14 : grp_3);
                int cand_15 = grp_3 + 16;
                int _min_5_3 = ((cand_15) < (G) ? (cand_15) : (G));
                int cand_first_15 = plist[_min_5_3];
                grp_3 = ((cand_first_15 <= (int)su_3) ? cand_15 : grp_3);
                int cand_16 = grp_3 + 8;
                int _min_6_3 = ((cand_16) < (G) ? (cand_16) : (G));
                int cand_first_16 = plist[_min_6_3];
                grp_3 = ((cand_first_16 <= (int)su_3) ? cand_16 : grp_3);
                int cand_17 = grp_3 + 4;
                int _min_7_3 = ((cand_17) < (G) ? (cand_17) : (G));
                int cand_first_17 = plist[_min_7_3];
                grp_3 = ((cand_first_17 <= (int)su_3) ? cand_17 : grp_3);
                int cand_18 = grp_3 + 2;
                int _min_8_3 = ((cand_18) < (G) ? (cand_18) : (G));
                int cand_first_18 = plist[_min_8_3];
                grp_3 = ((cand_first_18 <= (int)su_3) ? cand_18 : grp_3);
                int cand_19 = grp_3 + 1;
                int _min_9_3 = ((cand_19) < (G) ? (cand_19) : (G));
                int cand_first_19 = plist[_min_9_3];
                grp_3 = ((cand_first_19 <= (int)su_3) ? cand_19 : grp_3);
                int pj_3 = (int)su_3 - plist[grp_3];
                int p_left_3 = plist[1025 + grp_3] - pj_3 * 256;
                int p_ntok_3 = ((p_left_3 > 128) ? 256 : 128);
                int atoms = p_ntok_3 / 128;
                int w_blk0 = ((int)wu_3 * 256 + v_s * 128) / 128;
                int sfa_g = grp_3 * cols * M;
                int sfb_g = grp_3 * cols * N;
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < k_tiles_s; kt_2++) {
                    if (kt_2 % 4 == 0) {
                        int col = kt_2 / 4;
                        int a_blk_0 = pj_3 * 2 + ((atoms == 2) ? 0 : 0);
                        words[0] = (unsigned int)SFA[sfa_g + col * M + a_blk_0 * 128 + lane_1];
                        words[1] = (unsigned int)SFA[sfa_g + col * M + a_blk_0 * 128 + 32 + lane_1];
                        words[2] = (unsigned int)SFA[sfa_g + col * M + a_blk_0 * 128 + 64 + lane_1];
                        words[3] = (unsigned int)SFA[sfa_g + col * M + a_blk_0 * 128 + 96 + lane_1];
                        int a_blk_1 = pj_3 * 2 + ((atoms == 2) ? 1 : 0);
                        words[4] = (unsigned int)SFA[sfa_g + col * M + a_blk_1 * 128 + lane_1];
                        words[5] = (unsigned int)SFA[sfa_g + col * M + a_blk_1 * 128 + 32 + lane_1];
                        words[6] = (unsigned int)SFA[sfa_g + col * M + a_blk_1 * 128 + 64 + lane_1];
                        words[7] = (unsigned int)SFA[sfa_g + col * M + a_blk_1 * 128 + 96 + lane_1];
                        words[8] = (unsigned int)SFB[sfb_g + col * N + w_blk0 * 128 + lane_1];
                        words[9] = (unsigned int)SFB[sfb_g + col * N + w_blk0 * 128 + 32 + lane_1];
                        words[10] = (unsigned int)SFB[sfb_g + col * N + w_blk0 * 128 + 64 + lane_1];
                        words[11] = (unsigned int)SFB[sfb_g + col * N + w_blk0 * 128 + 96 + lane_1];
                    }
                    unsigned int shift = (unsigned int)(kt_2 % 4 * 8);
                    mbarrier_wait_cluster_hint(ae_addr + (stage_2) * 8, _phase_ae_1, 10000000);
                    unsigned int rep_0[4];
                    rep_0[0] = (words[0] >> shift & 255) * 16843009;
                    rep_0[1] = (words[1] >> shift & 255) * 16843009;
                    rep_0[2] = (words[2] >> shift & 255) * 16843009;
                    rep_0[3] = (words[3] >> shift & 255) * 16843009;
                    unsigned int atom_addr_0 = sm_tsf_addr + stage_2 * 1024 + 0;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(atom_addr_0 + (unsigned int)(lane_1 * 16)), "r"(rep_0[0]), "r"(rep_0[1]), "r"(rep_0[2]), "r"(rep_0[3]) : "memory");
                    unsigned int rep_1[4];
                    rep_1[0] = (words[4] >> shift & 255) * 16843009;
                    rep_1[1] = (words[5] >> shift & 255) * 16843009;
                    rep_1[2] = (words[6] >> shift & 255) * 16843009;
                    rep_1[3] = (words[7] >> shift & 255) * 16843009;
                    unsigned int atom_addr_1 = sm_tsf_addr + stage_2 * 1024 + 512;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(atom_addr_1 + (unsigned int)(lane_1 * 16)), "r"(rep_1[0]), "r"(rep_1[1]), "r"(rep_1[2]), "r"(rep_1[3]) : "memory");
                    unsigned int rep_2[4];
                    rep_2[0] = (words[8] >> shift & 255) * 16843009;
                    rep_2[1] = (words[9] >> shift & 255) * 16843009;
                    rep_2[2] = (words[10] >> shift & 255) * 16843009;
                    rep_2[3] = (words[11] >> shift & 255) * 16843009;
                    unsigned int atom_addr_2 = sm_wsf_addr + stage_2 * 512 + 0;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(atom_addr_2 + (unsigned int)(lane_1 * 16)), "r"(rep_2[0]), "r"(rep_2[1]), "r"(rep_2[2]), "r"(rep_2[3]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(sf_addr + (stage_2) * 8), "r"(leader_s) : "memory");
                    }
                    stage_2 += 1;
                    if (stage_2 == 9) { stage_2 = 0; _phase_ae_1 ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
