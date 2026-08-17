typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 272
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 264
#define NUM_TMA_PIPE_STAGES 2
#define NUM_MAINLOOP_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 51200
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 51200
#define SMEM_SMEM_SFA_OFF 50176
#define SMEM_SMEM_SFA_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_STRIDE 51200
#define SMEM_SMEM_SFB_OFF 51200
#define SMEM_SMEM_SFB_STAGE_BYTES 1024
#define SMEM_SMEM_SFB_STRIDE 51200
#define SMEM_TOTAL 103424

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
        :: "r"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}


__device__ __forceinline__ void tcgen05_mma_mxf8_bs_cta2(
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


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ void mma_ss_step_cg2(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::mxf8f6f4 [%2], da, db, %3, "
        "{m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
}


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


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm("{\n\t"
        ".reg .u64 u64addr;\n\t"
        "cvta.to.shared.u64 u64addr, %1;\n\t"
        "cvt.u32.u64 %0, u64addr;\n\t"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}


__device__ __forceinline__ uint32_t mapa_to_rank(uint32_t local_addr, uint32_t rank) {
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote) : "r"(local_addr), "r"(rank));
    return remote;
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo256(int addr) {
    const int SBO = 256;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
}


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_ld_x8_wait(float* dst, int addr) {
    tmem_ld_x8(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

extern "C" {

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_n128_k512(CakeTensorMap const* A, CakeTensorMap const* B, float* __restrict__ A_scale, float* __restrict__ B_scale, int* __restrict__ masked_m, __nv_bfloat16* __restrict__ C, int batch_size, int shape_m)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int smem_sfa_addr = smem + 50176;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 51200);
    const int smem_sfb_addr = smem + 51200;

    // Mbarrier init (4 groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 2 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            // mma_done: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // epilogue_done: 2 barriers, init_count=8
            mbarrier_init(smem + 48, 8);
            mbarrier_init(smem + 56, 8);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 272 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 64);
    if (warp == 0) {
        int _tmem_hold = smem + 64;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 16)
    #define mainloop_done_addr (mbar_base + 32)
    #define epilogue_done_addr (mbar_base + 48)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 264;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            const int epi_warp = warp % 4;
            const int epi_tid = epi_warp * 32 + lane;
            int chunks_per_group = (shape_m + 256 - 1) / 256;
            int total_chunks = batch_size * chunks_per_group;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int chunk_idx = cluster_id; chunk_idx < total_chunks; chunk_idx += num_clusters) {
                int group = chunk_idx / (unsigned int)chunks_per_group;
                int chunk_in_group = chunk_idx - (unsigned int)(group * chunks_per_group);
                int cluster_row = chunk_in_group * 256;
                int group_m = masked_m[group];
                if (cluster_row < group_m) {
                    mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int off_m = cluster_row + cta_rank * 128;
                    int output_row = off_m + epi_tid;
                    #pragma unroll 1
                    for (int n_chunk = 0; n_chunk < 16; n_chunk++) {
                        int tmem_row = cta_rank * 128 + epi_warp * 32;
                        int tmem_col = epi_stage * 128 + (unsigned int)(n_chunk * 8);
                        int tmem_addr = taddr + (unsigned int)(tmem_row << 16) + (unsigned int)tmem_col;
                        float _tmem_load_0[8];
                        tmem_ld_x8(&_tmem_load_0[0], tmem_addr);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        uint32_t _tmem_load_0_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        if (output_row < group_m) {
                            int output_index = (group * shape_m + output_row) * 128 + n_chunk * 8;
                            reinterpret_cast<int4*>(C + output_index)[0] = reinterpret_cast<int4*>(_tmem_load_0_bf16)[0];
                        }
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    epi_stage += 1;
                    if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_epi_stage = 0;
            int chunks_per_group_1 = (shape_m + 256 - 1) / 256;
            int total_chunks_1 = batch_size * chunks_per_group_1;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int chunk_idx_1 = cluster_id; chunk_idx_1 < total_chunks_1; chunk_idx_1 += num_clusters) {
                    int group_1 = chunk_idx_1 / (unsigned int)chunks_per_group_1;
                    int chunk_in_group_1 = chunk_idx_1 - (unsigned int)(group_1 * chunks_per_group_1);
                    int cluster_row_1 = chunk_in_group_1 * 256;
                    int group_m_1 = masked_m[group_1];
                    if (cluster_row_1 < group_m_1) {
                        mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < 2; iter_k++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init_flag = ((iter_k == 0) ? 1 : 0);
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_sbo256(smem_sfa_addr + mma_tma_stage * 51200));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 4), make_sf_cp_desc_sbo256((smem_sfa_addr + mma_tma_stage * 51200 + 128)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_sbo256(smem_sfb_addr + mma_tma_stage * 51200));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_sbo256((smem_sfb_addr + mma_tma_stage * 51200 + 128)));
                                int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3200;
                                int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3200;
                                {
                                    uint64_t a_desc = ((uint64_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf8_bs_cta2((tmem_accum + (mma_epi_stage * 128)), a_desc + 0, b_desc + 0,
                                        0x10a00000U, tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag) ? 0 : 1));
                                    tcgen05_mma_mxf8_bs_cta2((tmem_accum + (mma_epi_stage * 128)), a_desc + 2, b_desc + 2,
                                        0x30a00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                    tcgen05_mma_mxf8_bs_cta2((tmem_accum + (mma_epi_stage * 128)), a_desc + 4, b_desc + 4,
                                        0x50a00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                    tcgen05_mma_mxf8_bs_cta2((tmem_accum + (mma_epi_stage * 128)), a_desc + 6, b_desc + 6,
                                        0x70a00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                }
                                int _mma_a_lo_1 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 3200;
                                int _mma_b_lo_1 = (((smem_b_addr + 8192) >> 4) & 0x3FFF) + (mma_tma_stage) * 3200;
                                {
                                    uint64_t a_desc = ((uint64_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf8_bs_cta2((tmem_accum + (mma_epi_stage * 128)), a_desc + 0, b_desc + 0,
                                        0x10a00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                                    tcgen05_mma_mxf8_bs_cta2((tmem_accum + (mma_epi_stage * 128)), a_desc + 2, b_desc + 2,
                                        0x30a00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                                    tcgen05_mma_mxf8_bs_cta2((tmem_accum + (mma_epi_stage * 128)), a_desc + 4, b_desc + 4,
                                        0x50a00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                                    tcgen05_mma_mxf8_bs_cta2((tmem_accum + (mma_epi_stage * 128)), a_desc + 6, b_desc + 6,
                                        0x70a00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                                }
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 2) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                        mma_epi_stage += 1;
                        if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 5) {
        { // load_main
            unsigned int load_stage = 0;
            int chunks_per_group_2 = (shape_m + 256 - 1) / 256;
            int total_chunks_2 = batch_size * chunks_per_group_2;
            unsigned int _phase_mma_done = 1;
            #pragma unroll 1
            for (unsigned int chunk_idx_2 = cluster_id; chunk_idx_2 < total_chunks_2; chunk_idx_2 += num_clusters) {
                int group_2 = chunk_idx_2 / (unsigned int)chunks_per_group_2;
                int chunk_in_group_2 = chunk_idx_2 - (unsigned int)(group_2 * chunks_per_group_2);
                int cluster_row_2 = chunk_in_group_2 * 256;
                int group_m_2 = masked_m[group_2];
                if (cluster_row_2 < group_m_2) {
                    int off_m_1 = cluster_row_2 + cta_rank * 128;
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < 2; iter_k_1++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int sfa_base = smem_sfa_addr + load_stage * 51200;
                        int sfb_base = smem_sfb_addr + load_stage * 51200;
                        int b_scale_base = group_2 * 4 + iter_k_1 * 2;
                        unsigned int b0_bits = 0;
                        unsigned int b1_bits = 0;
                        b0_bits = reinterpret_cast<unsigned int*>(&B_scale[b_scale_base])[0];
                        b1_bits = reinterpret_cast<unsigned int*>(&B_scale[b_scale_base + 1])[0];
                        unsigned int b0 = b0_bits >> 23;
                        unsigned int b1 = b1_bits >> 23;
                        unsigned int b0_word = b0 | b0 << 8 | b0 << 16 | b0 << 24;
                        unsigned int b1_word = b1 | b1 << 8 | b1 << 16 | b1 << 24;
                        int sf_row = lane;
                        int source_row = off_m_1 + sf_row;
                        int safe_row = ((source_row < shape_m) ? source_row : 0);
                        int sf_c = lane / 8;
                        int sf_d = lane % 8;
                        int dst0 = (sf_c * 2 * 8 + sf_d) * 16;
                        int dst1 = ((sf_c * 2 + 1) * 8 + sf_d) * 16;
                        int a_scale_base = (group_2 * shape_m + safe_row) * 4 + iter_k_1 * 2;
                        unsigned int a0_bits = 0;
                        unsigned int a1_bits = 0;
                        a0_bits = reinterpret_cast<unsigned int*>(&A_scale[a_scale_base])[0];
                        a1_bits = reinterpret_cast<unsigned int*>(&A_scale[a_scale_base + 1])[0];
                        unsigned int a0 = a0_bits >> 23;
                        unsigned int a1 = a1_bits >> 23;
                        unsigned int a0_word = a0 | a0 << 8 | a0 << 16 | a0 << 24;
                        unsigned int a1_word = a1 | a1 << 8 | a1 << 16 | a1 << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base + dst0), "r"(a0_word));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base + dst1), "r"(a1_word));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base + dst0), "r"(b0_word));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base + dst1), "r"(b1_word));
                        int sf_row_0 = 32 + lane;
                        int source_row_1 = off_m_1 + sf_row_0;
                        int safe_row_2 = ((source_row_1 < shape_m) ? source_row_1 : 0);
                        int sf_c_3 = lane / 8;
                        int sf_d_4 = lane % 8;
                        int dst0_5 = (sf_c_3 * 2 * 8 + sf_d_4) * 16 + 4;
                        int dst1_6 = ((sf_c_3 * 2 + 1) * 8 + sf_d_4) * 16 + 4;
                        int a_scale_base_7 = (group_2 * shape_m + safe_row_2) * 4 + iter_k_1 * 2;
                        unsigned int a0_bits_8 = 0;
                        unsigned int a1_bits_9 = 0;
                        a0_bits_8 = reinterpret_cast<unsigned int*>(&A_scale[a_scale_base_7])[0];
                        a1_bits_9 = reinterpret_cast<unsigned int*>(&A_scale[a_scale_base_7 + 1])[0];
                        unsigned int a0_10 = a0_bits_8 >> 23;
                        unsigned int a1_11 = a1_bits_9 >> 23;
                        unsigned int a0_word_12 = a0_10 | a0_10 << 8 | a0_10 << 16 | a0_10 << 24;
                        unsigned int a1_word_13 = a1_11 | a1_11 << 8 | a1_11 << 16 | a1_11 << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base + dst0_5), "r"(a0_word_12));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base + dst1_6), "r"(a1_word_13));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base + dst0_5), "r"(b0_word));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base + dst1_6), "r"(b1_word));
                        int sf_row_14 = 64 + lane;
                        int source_row_15 = off_m_1 + sf_row_14;
                        int safe_row_16 = ((source_row_15 < shape_m) ? source_row_15 : 0);
                        int sf_c_17 = lane / 8;
                        int sf_d_18 = lane % 8;
                        int dst0_19 = (sf_c_17 * 2 * 8 + sf_d_18) * 16 + 8;
                        int dst1_20 = ((sf_c_17 * 2 + 1) * 8 + sf_d_18) * 16 + 8;
                        int a_scale_base_21 = (group_2 * shape_m + safe_row_16) * 4 + iter_k_1 * 2;
                        unsigned int a0_bits_22 = 0;
                        unsigned int a1_bits_23 = 0;
                        a0_bits_22 = reinterpret_cast<unsigned int*>(&A_scale[a_scale_base_21])[0];
                        a1_bits_23 = reinterpret_cast<unsigned int*>(&A_scale[a_scale_base_21 + 1])[0];
                        unsigned int a0_24 = a0_bits_22 >> 23;
                        unsigned int a1_25 = a1_bits_23 >> 23;
                        unsigned int a0_word_26 = a0_24 | a0_24 << 8 | a0_24 << 16 | a0_24 << 24;
                        unsigned int a1_word_27 = a1_25 | a1_25 << 8 | a1_25 << 16 | a1_25 << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base + dst0_19), "r"(a0_word_26));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base + dst1_20), "r"(a1_word_27));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base + dst0_19), "r"(b0_word));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base + dst1_20), "r"(b1_word));
                        int sf_row_28 = 96 + lane;
                        int source_row_29 = off_m_1 + sf_row_28;
                        int safe_row_30 = ((source_row_29 < shape_m) ? source_row_29 : 0);
                        int sf_c_31 = lane / 8;
                        int sf_d_32 = lane % 8;
                        int dst0_33 = (sf_c_31 * 2 * 8 + sf_d_32) * 16 + 12;
                        int dst1_34 = ((sf_c_31 * 2 + 1) * 8 + sf_d_32) * 16 + 12;
                        int a_scale_base_35 = (group_2 * shape_m + safe_row_30) * 4 + iter_k_1 * 2;
                        unsigned int a0_bits_36 = 0;
                        unsigned int a1_bits_37 = 0;
                        a0_bits_36 = reinterpret_cast<unsigned int*>(&A_scale[a_scale_base_35])[0];
                        a1_bits_37 = reinterpret_cast<unsigned int*>(&A_scale[a_scale_base_35 + 1])[0];
                        unsigned int a0_38 = a0_bits_36 >> 23;
                        unsigned int a1_39 = a1_bits_37 >> 23;
                        unsigned int a0_word_40 = a0_38 | a0_38 << 8 | a0_38 << 16 | a0_38 << 24;
                        unsigned int a1_word_41 = a1_39 | a1_39 << 8 | a1_39 << 16 | a1_39 << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base + dst0_33), "r"(a0_word_40));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base + dst1_34), "r"(a1_word_41));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base + dst0_33), "r"(b0_word));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base + dst1_34), "r"(b1_word));
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            tma_4d_gmem2smem_cta2(smem_a_addr + load_stage * 51200, A, 0, off_m_1, iter_k_1 * 2, group_2, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_4d_gmem2smem_cta2(smem_b_addr + load_stage * 51200, B, 0, cta_rank * 64, iter_k_1 * 2, group_2, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(49152)) : "memory");
                        }
                        load_stage += 1;
                        if (load_stage == 2) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
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

