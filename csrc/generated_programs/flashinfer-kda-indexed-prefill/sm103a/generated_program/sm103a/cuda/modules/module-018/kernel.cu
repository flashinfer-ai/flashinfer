typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CudaTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CudaTensorMapPack { CudaTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CUDA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_M_OFF 1024
#define SMEM_SMEM_M_STAGE_BYTES 65536
#define SMEM_SMEM_M_STRIDE 65536
#define SMEM_TOTAL 66560
#define THREADS 128

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


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}

extern "C" {

__global__ __launch_bounds__(128, 3) void
kernel_flashkda_split_scan_bf16_m128(float* __restrict__ split_state, __nv_bfloat16* __restrict__ map_state_bf16, float* __restrict__ carry, int num_heads, int num_parts)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int mbar_base = smem;
    #define map_full0_addr (mbar_base + 0)
    #define map_full1_addr (mbar_base + 8)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_m = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_m_addr = smem + 1024;

    // Mbarrier init (2 groups, 2 barriers)
    // Mbarriers at smem_raw[0..16)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // map_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // map_full1: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // === Task calls (dependency order) ===
    int task = bid / 32;
    int row_band = bid % 32;
    int row = row_band * 4 + warp;
    int col = lane * 4;
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(map_full0_addr, 32768);
            cp_async_bulk_gmem2smem(smem_m_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(map_state_bf16) + ((unsigned long long)(task * 16384) * (unsigned long long)2)), 32768, map_full0_addr);
        }
    }
    float carry_reg[4];
    float next_state[4];
    int state_offset = task * 16384 + row * 128 + col;
    {
        float4 _v4 = *reinterpret_cast<const float4*>(split_state + state_offset);
        carry_reg[0 + 0] = _v4.x;
        carry_reg[0 + 1] = _v4.y;
        carry_reg[0 + 2] = _v4.z;
        carry_reg[0 + 3] = _v4.w;
    }
    {
        float4 _v4 = *reinterpret_cast<const float4*>(split_state + (num_heads + task) * 16384 + row * 128 + col);
        next_state[0 + 0] = _v4.x;
        next_state[0 + 1] = _v4.y;
        next_state[0 + 2] = _v4.z;
        next_state[0 + 3] = _v4.w;
    }
    __syncthreads();
    unsigned int _phase_map_full0_0 = 0;
    unsigned int _phase_map_full1_0 = 0;
    #pragma unroll 1
    for (int part = 1; part < num_parts; part++) {
        if (part + 1 < num_parts) {
            if (warp == 0) {
                if (elect_sync()) {
                    int map_offset = (part * num_heads + task) * 16384;
                    if (part % 2 == 1) {
                        mbarrier_arrive_expect_tx(map_full1_addr, 32768);
                        cp_async_bulk_gmem2smem(smem_m_addr + 32768, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(map_state_bf16) + ((unsigned long long)map_offset * (unsigned long long)2)), 32768, map_full1_addr);
                    } else {
                        mbarrier_arrive_expect_tx(map_full0_addr, 32768);
                        cp_async_bulk_gmem2smem(smem_m_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(map_state_bf16) + ((unsigned long long)map_offset * (unsigned long long)2)), 32768, map_full0_addr);
                    }
                }
            }
        }
        int carry_offset = ((part - 1) * num_heads + task) * 16384 + row * 128 + col;
        {
            float4 _v4 = make_float4(carry_reg[0 + 0], carry_reg[0 + 1], carry_reg[0 + 2], carry_reg[0 + 3]);
            *reinterpret_cast<float4*>(carry + carry_offset) = _v4;
        }
        float accum[4];
        for (int out_col = 0; out_col < 4; out_col++) {
            accum[out_col] = next_state[out_col];
        }
        if (part + 1 < num_parts) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(split_state + ((part + 1) * num_heads + task) * 16384 + row * 128 + col);
                next_state[0 + 0] = _v4.x;
                next_state[0 + 1] = _v4.y;
                next_state[0 + 2] = _v4.z;
                next_state[0 + 3] = _v4.w;
            }
        }
        int panel_offset = 0;
        if (part % 2 == 1) {
            mbarrier_wait(map_full0_addr, _phase_map_full0_0);
            _phase_map_full0_0 ^= 1;
        } else {
            mbarrier_wait(map_full1_addr, _phase_map_full1_0);
            _phase_map_full1_0 ^= 1;
            panel_offset = 16384;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        #pragma unroll
        for (int inner = 0; inner < 128; inner++) {
            float _shfl_0 = __shfl_sync(0xFFFFFFFF, carry_reg[inner % 4], inner / 4);
            float a = _shfl_0;
            unsigned int packed_m[2];
            int m_addr = smem_m_addr + (unsigned int)((panel_offset + inner * 128 + col) * 2);
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&packed_m[0])) : "r"(m_addr));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&packed_m[1])) : "r"(m_addr + 4));
            float _bf16x2_add_f32_0[2];
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(_bf16x2_add_f32_0[0]), "=&f"(_bf16x2_add_f32_0[1]) : "r"(packed_m[0]), "f"(0.0f), "f"(0.0f));
            float _bf16x2_add_f32_1[2];
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                "}\n"
                : "=&f"(_bf16x2_add_f32_1[0]), "=&f"(_bf16x2_add_f32_1[1]) : "r"(packed_m[1]), "f"(0.0f), "f"(0.0f));
            float _fma_0 = __fmaf_rn(a, _bf16x2_add_f32_0[0], accum[0]);
            accum[0] = _fma_0;
            float _fma_1 = __fmaf_rn(a, _bf16x2_add_f32_0[1], accum[1]);
            accum[1] = _fma_1;
            float _fma_2 = __fmaf_rn(a, _bf16x2_add_f32_1[0], accum[2]);
            accum[2] = _fma_2;
            float _fma_3 = __fmaf_rn(a, _bf16x2_add_f32_1[1], accum[3]);
            accum[3] = _fma_3;
        }
        for (int out_col_1 = 0; out_col_1 < 4; out_col_1++) {
            carry_reg[out_col_1] = accum[out_col_1];
        }
        __syncthreads();
    }
    __threadfence();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // Cleanup
    __syncthreads();
}

} // extern "C"

