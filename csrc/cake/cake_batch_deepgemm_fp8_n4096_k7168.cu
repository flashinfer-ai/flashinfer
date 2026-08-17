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
#define TMEM_TMEM_ACC_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 264
#define NUM_TMA_PIPE_STAGES 3
#define NUM_MAINLOOP_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 67584
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 67584
#define SMEM_SMEM_SFA_OFF 66560
#define SMEM_SMEM_SFA_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_STRIDE 67584
#define SMEM_SMEM_SFB_OFF 67584
#define SMEM_SMEM_SFB_STAGE_BYTES 1024
#define SMEM_SMEM_SFB_STRIDE 67584
#define SMEM_TOTAL 203776

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


__device__ __forceinline__ void tcgen05_mma_mxf8_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ void mma_ss_step(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::mxf8f6f4 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
}


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
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


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo256(int addr) {
    const int SBO = 256;
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


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(192, 1) void
kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_n4096_k7168(CakeTensorMap const* A, CakeTensorMap const* B, int* __restrict__ SFA_bits, int* __restrict__ SFB_bits, __nv_bfloat16* __restrict__ C, int* __restrict__ masked_m, unsigned int batch_size, unsigned int shape_m)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
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
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_sfa_addr = smem + 66560;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 67584);
    const int smem_sfb_addr = smem + 67584;

    // Mbarrier init (4 groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // mma_done: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // epilogue_done: 2 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 272 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 0) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 24)
    #define mainloop_done_addr (mbar_base + 48)
    #define epilogue_done_addr (mbar_base + 64)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 264;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            const int epi_warp = warp % 4;
            const int epi_tid = epi_warp * 32 + lane;
            unsigned int num_bids_u = (unsigned int)num_bids;
            unsigned int bid_u = (unsigned int)bid;
            unsigned int current_group_idx = 0;
            unsigned int current_m_cumsum = 0;
            unsigned int max_m_blocks = (shape_m + 128 - 1) / 128;
            unsigned int max_total_tiles = batch_size * max_m_blocks * 32;
            unsigned int max_scheduler_iters = (max_total_tiles + num_bids_u - 1) / num_bids_u;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (int current_iter = 0; current_iter < max_scheduler_iters; current_iter++) {
                unsigned int next_block_idx = (unsigned int)current_iter * num_bids_u + bid_u;
                int has_block = 0;
                unsigned int m_blocks_g = 0;
                #pragma unroll 1
                for (int scan_g = current_group_idx; scan_g < batch_size; scan_g++) {
                    unsigned int group_m = (unsigned int)masked_m[scan_g];
                    m_blocks_g = (group_m + 128 - 1) / 128;
                    unsigned int next_m_cumsum = current_m_cumsum + m_blocks_g;
                    if (next_block_idx < next_m_cumsum * 32) {
                        current_group_idx = scan_g;
                        has_block = 1;
                        break;
                    }
                    current_m_cumsum = next_m_cumsum;
                }
                if (has_block == 0) {
                    break;
                }
                unsigned int block_idx = next_block_idx - current_m_cumsum * 32;
                unsigned int blocks_per_l2_group = m_blocks_g * 16;
                unsigned int l2_group = block_idx / blocks_per_l2_group;
                unsigned int first_n_block = l2_group * 16;
                unsigned int in_l2_group = block_idx % blocks_per_l2_group;
                unsigned int n_blocks_in_group = (unsigned int)16;
                unsigned int m_block = in_l2_group / n_blocks_in_group;
                unsigned int n_block = first_n_block + in_l2_group % n_blocks_in_group;
                unsigned int off_m = m_block * 128;
                unsigned int off_n = n_block * 128;
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll 1
                for (int n_chunk = 0; n_chunk < 16; n_chunk++) {
                    int row = epi_warp * 32;
                    int col = epi_stage * 128 + (unsigned int)(n_chunk * 8);
                    int tmem_addr = taddr + (unsigned int)(row << 16) + (unsigned int)col;
                    float _tmem_load_0[8];
                    tmem_ld_x8(&_tmem_load_0[0], tmem_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    uint32_t _tmem_load_0_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int flat_row = current_group_idx * shape_m + off_m + (unsigned int)epi_tid;
                    reinterpret_cast<int4*>(C + (flat_row * 4096 + off_n + (unsigned int)(n_chunk * 8)))[0] = reinterpret_cast<int4*>(_tmem_load_0_bf16)[0];
                }
                if (elect_sync()) {
                    mbarrier_arrive(epilogue_done_addr + (epi_stage) * 8);
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_epi_stage = 0;
            unsigned int num_bids_u_1 = (unsigned int)num_bids;
            unsigned int bid_u_1 = (unsigned int)bid;
            unsigned int current_group_idx_1 = 0;
            unsigned int current_m_cumsum_1 = 0;
            unsigned int max_m_blocks_1 = (shape_m + 128 - 1) / 128;
            unsigned int max_total_tiles_1 = batch_size * max_m_blocks_1 * 32;
            unsigned int max_scheduler_iters_1 = (max_total_tiles_1 + num_bids_u_1 - 1) / num_bids_u_1;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (int current_iter_1 = 0; current_iter_1 < max_scheduler_iters_1; current_iter_1++) {
                unsigned int next_block_idx_1 = (unsigned int)current_iter_1 * num_bids_u_1 + bid_u_1;
                int has_block_1 = 0;
                #pragma unroll 1
                for (int scan_g_1 = current_group_idx_1; scan_g_1 < batch_size; scan_g_1++) {
                    unsigned int group_m_1 = (unsigned int)masked_m[scan_g_1];
                    unsigned int m_blocks_g_1 = (group_m_1 + 128 - 1) / 128;
                    unsigned int next_m_cumsum_1 = current_m_cumsum_1 + m_blocks_g_1;
                    if (next_block_idx_1 < next_m_cumsum_1 * 32) {
                        current_group_idx_1 = scan_g_1;
                        has_block_1 = 1;
                        break;
                    }
                    current_m_cumsum_1 = next_m_cumsum_1;
                }
                if (has_block_1 == 0) {
                    break;
                }
                mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                #pragma unroll 1
                for (int iter_k = 0; iter_k < 28; iter_k++) {
                    mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((iter_k == 0) ? 1 : 0);
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfa, make_sf_cp_desc_sbo256(smem_sfa_addr + mma_tma_stage * 67584));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa + 4), make_sf_cp_desc_sbo256((smem_sfa_addr + mma_tma_stage * 67584 + 128)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb, make_sf_cp_desc_sbo256(smem_sfb_addr + mma_tma_stage * 67584));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb + 4), make_sf_cp_desc_sbo256((smem_sfb_addr + mma_tma_stage * 67584 + 128)));
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 4224;
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 4224;
                        {
                            uint64_t a_desc = ((uint64_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 0, b_desc + 0,
                                0x8a00000U, tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag) ? 0 : 1));
                            tcgen05_mma_mxf8_bs((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 2, b_desc + 2,
                                0x28a00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            tcgen05_mma_mxf8_bs((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 4, b_desc + 4,
                                0x48a00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            tcgen05_mma_mxf8_bs((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 6, b_desc + 6,
                                0x68a00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                        }
                        int _mma_a_lo_1 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 4224;
                        int _mma_b_lo_1 = (((smem_b_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 4224;
                        {
                            uint64_t a_desc = ((uint64_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 0, b_desc + 0,
                                0x8a00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 2, b_desc + 2,
                                0x28a00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 4, b_desc + 4,
                                0x48a00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 6, b_desc + 6,
                                0x68a00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                        }
                    }
                    elect_commit(mma_done_addr + (mma_tma_stage) * 8);
                    mma_tma_stage += 1;
                    if (mma_tma_stage == 3) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                }
                elect_commit(mainloop_done_addr + (mma_epi_stage) * 8);
                mma_epi_stage += 1;
                if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 5) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int num_bids_u_2 = (unsigned int)num_bids;
            unsigned int bid_u_2 = (unsigned int)bid;
            unsigned int current_group_idx_2 = 0;
            unsigned int current_m_cumsum_2 = 0;
            unsigned int max_m_blocks_2 = (shape_m + 128 - 1) / 128;
            unsigned int max_total_tiles_2 = batch_size * max_m_blocks_2 * 32;
            unsigned int max_scheduler_iters_2 = (max_total_tiles_2 + num_bids_u_2 - 1) / num_bids_u_2;
            unsigned int _phase_mma_done = 1;
            #pragma unroll 1
            for (int current_iter_2 = 0; current_iter_2 < max_scheduler_iters_2; current_iter_2++) {
                unsigned int next_block_idx_2 = (unsigned int)current_iter_2 * num_bids_u_2 + bid_u_2;
                int has_block_2 = 0;
                unsigned int m_blocks_g_2 = 0;
                #pragma unroll 1
                for (int scan_g_2 = current_group_idx_2; scan_g_2 < batch_size; scan_g_2++) {
                    unsigned int group_m_2 = (unsigned int)masked_m[scan_g_2];
                    m_blocks_g_2 = (group_m_2 + 128 - 1) / 128;
                    unsigned int next_m_cumsum_2 = current_m_cumsum_2 + m_blocks_g_2;
                    if (next_block_idx_2 < next_m_cumsum_2 * 32) {
                        current_group_idx_2 = scan_g_2;
                        has_block_2 = 1;
                        break;
                    }
                    current_m_cumsum_2 = next_m_cumsum_2;
                }
                if (has_block_2 == 0) {
                    break;
                }
                unsigned int block_idx_1 = next_block_idx_2 - current_m_cumsum_2 * 32;
                unsigned int blocks_per_l2_group_1 = m_blocks_g_2 * 16;
                unsigned int l2_group_1 = block_idx_1 / blocks_per_l2_group_1;
                unsigned int first_n_block_1 = l2_group_1 * 16;
                unsigned int in_l2_group_1 = block_idx_1 % blocks_per_l2_group_1;
                unsigned int n_blocks_in_group_1 = (unsigned int)16;
                unsigned int m_block_1 = in_l2_group_1 / n_blocks_in_group_1;
                unsigned int n_block_1 = first_n_block_1 + in_l2_group_1 % n_blocks_in_group_1;
                unsigned int off_m_1 = m_block_1 * 128;
                unsigned int off_n_1 = n_block_1 * 128;
                unsigned int sf_cols = 56;
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < 28; iter_k_1++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    int sfa_base_smem = smem_sfa_addr + load_stage * 67584;
                    int sfb_base_smem = smem_sfb_addr + load_stage * 67584;
                    unsigned int sfb_idx0 = (current_group_idx_2 * 32 + n_block_1) * sf_cols + (unsigned int)iter_k_1 * 2;
                    unsigned int sfb_idx1 = sfb_idx0 + 1;
                    unsigned int sfb_bits0 = (unsigned int)SFB_bits[sfb_idx0];
                    unsigned int sfb_bits1 = (unsigned int)SFB_bits[sfb_idx1];
                    unsigned int sfb0 = sfb_bits0 >> 23 & 255;
                    unsigned int sfb1 = sfb_bits1 >> 23 & 255;
                    unsigned int sfb0_word = sfb0 | sfb0 << 8 | sfb0 << 16 | sfb0 << 24;
                    unsigned int sfb1_word = sfb1 | sfb1 << 8 | sfb1 << 16 | sfb1 << 24;
                    unsigned int sf_row = (unsigned int)lane;
                    int sf_c = lane / 8;
                    int sf_d = lane % 8;
                    int sf_dst0 = (sf_c * 2 * 8 + sf_d) * 16;
                    int sf_dst1 = ((sf_c * 2 + 1) * 8 + sf_d) * 16;
                    unsigned int sfa_idx0 = (current_group_idx_2 * shape_m + off_m_1 + sf_row) * sf_cols + (unsigned int)iter_k_1 * 2;
                    unsigned int sfa_idx1 = sfa_idx0 + 1;
                    unsigned int sfa_bits0 = (unsigned int)SFA_bits[sfa_idx0];
                    unsigned int sfa_bits1 = (unsigned int)SFA_bits[sfa_idx1];
                    unsigned int sfa0 = sfa_bits0 >> 23 & 255;
                    unsigned int sfa1 = sfa_bits1 >> 23 & 255;
                    unsigned int sfa0_word = sfa0 | sfa0 << 8 | sfa0 << 16 | sfa0 << 24;
                    unsigned int sfa1_word = sfa1 | sfa1 << 8 | sfa1 << 16 | sfa1 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst0), "r"(sfa0_word));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst1), "r"(sfa1_word));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst0), "r"(sfb0_word));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst1), "r"(sfb1_word));
                    unsigned int sf_row_0 = (unsigned int)(32 + lane);
                    int sf_c_1 = lane / 8;
                    int sf_d_2 = lane % 8;
                    int sf_dst0_3 = (sf_c_1 * 2 * 8 + sf_d_2) * 16 + 4;
                    int sf_dst1_4 = ((sf_c_1 * 2 + 1) * 8 + sf_d_2) * 16 + 4;
                    unsigned int sfa_idx0_5 = (current_group_idx_2 * shape_m + off_m_1 + sf_row_0) * sf_cols + (unsigned int)iter_k_1 * 2;
                    unsigned int sfa_idx1_6 = sfa_idx0_5 + 1;
                    unsigned int sfa_bits0_7 = (unsigned int)SFA_bits[sfa_idx0_5];
                    unsigned int sfa_bits1_8 = (unsigned int)SFA_bits[sfa_idx1_6];
                    unsigned int sfa0_9 = sfa_bits0_7 >> 23 & 255;
                    unsigned int sfa1_10 = sfa_bits1_8 >> 23 & 255;
                    unsigned int sfa0_word_11 = sfa0_9 | sfa0_9 << 8 | sfa0_9 << 16 | sfa0_9 << 24;
                    unsigned int sfa1_word_12 = sfa1_10 | sfa1_10 << 8 | sfa1_10 << 16 | sfa1_10 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst0_3), "r"(sfa0_word_11));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst1_4), "r"(sfa1_word_12));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst0_3), "r"(sfb0_word));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst1_4), "r"(sfb1_word));
                    unsigned int sf_row_13 = (unsigned int)(64 + lane);
                    int sf_c_14 = lane / 8;
                    int sf_d_15 = lane % 8;
                    int sf_dst0_16 = (sf_c_14 * 2 * 8 + sf_d_15) * 16 + 8;
                    int sf_dst1_17 = ((sf_c_14 * 2 + 1) * 8 + sf_d_15) * 16 + 8;
                    unsigned int sfa_idx0_18 = (current_group_idx_2 * shape_m + off_m_1 + sf_row_13) * sf_cols + (unsigned int)iter_k_1 * 2;
                    unsigned int sfa_idx1_19 = sfa_idx0_18 + 1;
                    unsigned int sfa_bits0_20 = (unsigned int)SFA_bits[sfa_idx0_18];
                    unsigned int sfa_bits1_21 = (unsigned int)SFA_bits[sfa_idx1_19];
                    unsigned int sfa0_22 = sfa_bits0_20 >> 23 & 255;
                    unsigned int sfa1_23 = sfa_bits1_21 >> 23 & 255;
                    unsigned int sfa0_word_24 = sfa0_22 | sfa0_22 << 8 | sfa0_22 << 16 | sfa0_22 << 24;
                    unsigned int sfa1_word_25 = sfa1_23 | sfa1_23 << 8 | sfa1_23 << 16 | sfa1_23 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst0_16), "r"(sfa0_word_24));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst1_17), "r"(sfa1_word_25));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst0_16), "r"(sfb0_word));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst1_17), "r"(sfb1_word));
                    unsigned int sf_row_26 = (unsigned int)(96 + lane);
                    int sf_c_27 = lane / 8;
                    int sf_d_28 = lane % 8;
                    int sf_dst0_29 = (sf_c_27 * 2 * 8 + sf_d_28) * 16 + 12;
                    int sf_dst1_30 = ((sf_c_27 * 2 + 1) * 8 + sf_d_28) * 16 + 12;
                    unsigned int sfa_idx0_31 = (current_group_idx_2 * shape_m + off_m_1 + sf_row_26) * sf_cols + (unsigned int)iter_k_1 * 2;
                    unsigned int sfa_idx1_32 = sfa_idx0_31 + 1;
                    unsigned int sfa_bits0_33 = (unsigned int)SFA_bits[sfa_idx0_31];
                    unsigned int sfa_bits1_34 = (unsigned int)SFA_bits[sfa_idx1_32];
                    unsigned int sfa0_35 = sfa_bits0_33 >> 23 & 255;
                    unsigned int sfa1_36 = sfa_bits1_34 >> 23 & 255;
                    unsigned int sfa0_word_37 = sfa0_35 | sfa0_35 << 8 | sfa0_35 << 16 | sfa0_35 << 24;
                    unsigned int sfa1_word_38 = sfa1_36 | sfa1_36 << 8 | sfa1_36 << 16 | sfa1_36 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst0_29), "r"(sfa0_word_37));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst1_30), "r"(sfa1_word_38));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst0_29), "r"(sfb0_word));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst1_30), "r"(sfb1_word));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_a_addr + load_stage * 67584), "l"(A), "r"(0), "r"(current_group_idx_2 * shape_m + off_m_1), "r"(iter_k_1 * 2),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x1000000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_b_addr + load_stage * 67584), "l"(B), "r"(0), "r"(current_group_idx_2 * 4096 + off_n_1), "r"(iter_k_1 * 2),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x1000000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 65536);
                    }
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"

