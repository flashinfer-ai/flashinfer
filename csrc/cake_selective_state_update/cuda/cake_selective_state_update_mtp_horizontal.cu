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
#define NUM_INPUT_PIPE_STAGES 2
#define SMEM_S_STATE_OFF 1024
#define SMEM_S_STATE_STAGE_BYTES 8192
#define SMEM_S_STATE_STRIDE 8192
#define SMEM_S_B_OFF 33792
#define SMEM_S_B_STAGE_BYTES 1536
#define SMEM_S_B_STRIDE 1536
#define SMEM_S_C_OFF 39936
#define SMEM_S_C_STAGE_BYTES 1536
#define SMEM_S_C_STRIDE 1536
#define SMEM_S_X_OFF 46080
#define SMEM_S_X_STAGE_BYTES 768
#define SMEM_S_X_STRIDE 768
#define SMEM_S_DT_OFF 49152
#define SMEM_S_DT_STAGE_BYTES 96
#define SMEM_S_DT_STRIDE 96
#define SMEM_S_DECAY_OFF 49248
#define SMEM_S_DECAY_STAGE_BYTES 96
#define SMEM_S_DECAY_STRIDE 96
#define SMEM_TOTAL 49408
#define THREADS 288

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


__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ void fma_scale_x32(
    float* sv, const float2* scale2, const float2* neg_max2)
{
    float2* sv_2 = reinterpret_cast<float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++)
        fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(288, 4) void
kernel_cake_selective_state_update_mtp_horizontal(CakeTensorMap const* state_tma, CakeTensorMap const* x_tma, CakeTensorMap const* b_tma, CakeTensorMap const* c_tma, float* __restrict__ dt, float* __restrict__ A, float* __restrict__ D, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ output, long long* __restrict__ state_batch_indices, __nv_bfloat16* __restrict__ intermediate_state, long long* __restrict__ intermediate_state_indices, int nheads, int ngroups, int total_tiles, unsigned long long intermediate_stride_slot, int dt_softplus, int cache_intermediate)
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
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(state_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(x_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(b_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(c_tma)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* s_state = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int s_state_addr = smem + 1024;
    __nv_bfloat16* s_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int s_b_addr = smem + 33792;
    __nv_bfloat16* s_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 39936);
    const int s_c_addr = smem + 39936;
    __nv_bfloat16* s_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 46080);
    const int s_x_addr = smem + 46080;
    float* s_dt = reinterpret_cast<float*>(smem_raw + 49152);
    const int s_dt_addr = smem + 49152;
    float* s_decay = reinterpret_cast<float*>(smem_raw + 49248);
    const int s_decay_addr = smem + 49248;

    // Mbarrier init (7 groups, 14 barriers)
    // Mbarriers at smem_raw[0..112)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // state_full_0: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // state_empty_0: 2 barriers, init_count=4
            mbarrier_init(smem + 16, 4);
            mbarrier_init(smem + 24, 4);
            // --- pipeline 'input_pipe' ---
            // inputs_empty_0: 2 barriers, init_count=4
            mbarrier_init(smem + 32, 4);
            mbarrier_init(smem + 40, 4);
            // state_full_1: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // state_empty_1: 2 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            // inputs_empty_1: 2 barriers, init_count=4
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            // dt_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncthreads();

    const int mbar_base = smem;
    #define state_full_0_addr (mbar_base + 0)
    #define state_empty_0_addr (mbar_base + 16)
    #define inputs_empty_0_addr (mbar_base + 32)
    #define state_full_1_addr (mbar_base + 48)
    #define state_empty_1_addr (mbar_base + 64)
    #define inputs_empty_1_addr (mbar_base + 80)
    #define dt_ready_addr (mbar_base + 96)

    // ---- Role: consumers ----
    if (warp <= 7) {
        { // consumers_main
            int warp_id_in_role = (warp - 0);
            int local_warp = warp_id_in_role;
            int cohort = local_warp / 4;
            int cohort_warp = local_warp - cohort * 4;
            int subgroup = lane / 8;
            int member = lane % 8;
            float state_values[16];
            unsigned int state_carriers[8];
            unsigned int input_phase = 0;
            unsigned int state_phase = 0;
            #pragma unroll 1
            for (int work_group = bid; work_group < (total_tiles + 2 - 1) / 2; work_group += num_bids) {
                int raw_work_tile = work_group * 2 + cohort;
                bool tile_valid = raw_work_tile < total_tiles;
                int work_tile = raw_work_tile;
                if (!tile_valid) {
                    work_tile = total_tiles - 1;
                }
                int batch = work_tile / nheads;
                int head = work_tile % nheads;
                long long source_slot = state_batch_indices[batch];
                float a_value = A[head];
                float d_value = D[head];
                unsigned int input_stage = (unsigned int)(cohort * 2) + state_phase;
                unsigned int state_base = (unsigned int)(cohort * 2);
                int input_b_addr = s_b_addr + input_stage * 1536;
                int input_c_addr = s_c_addr + input_stage * 1536;
                if (cohort_warp == 0) {
                    if (lane < 6) {
                        int token = lane;
                        float dt_value = dt[(batch * 6 + token) * nheads + head];
                        dt_value += dt_bias[head];
                        if (dt_softplus != 0) {
                            if (dt_value <= 20.0f) {
                                float _exp_0 = expf(dt_value);
                                float _log1p_0 = log1pf(_exp_0);
                                dt_value = _log1p_0;
                            }
                        }
                        int token_scalar_index = input_stage * 6 + (unsigned int)token;
                        s_dt[token_scalar_index] = dt_value;
                        float _exp_1 = expf(a_value * dt_value);
                        s_decay[token_scalar_index] = _exp_1;
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(dt_ready_addr + (cohort) * 8);
                    }
                }
                mbarrier_wait(dt_ready_addr + (cohort) * 8, state_phase);
                for (int stage = 0; stage < 2; stage++) {
                    unsigned int barrier_state_stage = (unsigned int)stage;
                    unsigned int physical_state_stage = state_base + barrier_state_stage;
                    if (cohort == 0) {
                        mbarrier_wait(state_full_0_addr + (barrier_state_stage) * 8, state_phase);
                    } else {
                        mbarrier_wait(state_full_1_addr + (barrier_state_stage) * 8, state_phase);
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    for (int subpass = 0; subpass < 2; subpass++) {
                        int row_in_stage = subpass * 16 + cohort_warp * 4 + subgroup;
                        int dim_index = stage * 32 + row_in_stage;
                        int state_col_low = member * 8;
                        int state_col_high = 64 + state_col_low;
                        int state_row_base = (int)physical_state_stage * 4096 + row_in_stage * 128;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(0) + 3]))
                            : "r"(s_state_addr + (unsigned int)((state_row_base + state_col_low) * 2)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[4])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&state_carriers[(4) + 3]))
                            : "r"(s_state_addr + (unsigned int)((state_row_base + state_col_high) * 2)));
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&state_values[_pair * 2])[0]), "=f"((&state_values[_pair * 2])[1])
                                : "r"(state_carriers[_pair]));
                        }
                        unsigned long long intermediate_index_low = 0;
                        unsigned long long intermediate_index_high = 0;
                        unsigned long long intermediate_step_stride = (unsigned long long)(nheads * 64 * 128);
                        if (cache_intermediate != 0) {
                            if (tile_valid) {
                                if (source_slot >= 0) {
                                    long long cache_slot = intermediate_state_indices[batch];
                                    unsigned long long row_offset = (unsigned long long)((head * 64 + dim_index) * 128);
                                    intermediate_index_low = (unsigned long long)cache_slot * intermediate_stride_slot + row_offset + (unsigned long long)state_col_low;
                                    intermediate_index_high = (unsigned long long)cache_slot * intermediate_stride_slot + row_offset + (unsigned long long)state_col_high;
                                }
                            }
                        }
                        for (int token_1 = 0; token_1 < 6; token_1++) {
                            float dt_value_1 = s_dt[input_stage * 6 + (unsigned int)token_1];
                            float decay = s_decay[input_stage * 6 + (unsigned int)token_1];
                            float x_value = s_x[input_stage * 6 * 64 + (unsigned int)(token_1 * 64) + (unsigned int)dim_index];
                            float2 _f2_0 = make_float2(decay, decay);
                            float2 decay_pair = _f2_0;
                            float2 _f2_1 = make_float2(dt_value_1, dt_value_1);
                            float2 dt_pair = _f2_1;
                            float2 _f2_2 = make_float2(x_value, x_value);
                            float2 x_pair = _f2_2;
                            float2 _f2_3 = make_float2(0.0f, 0.0f);
                            float2 partial_pair = _f2_3;
                            #pragma unroll
                            for (int fragment = 0; fragment < 2; fragment++) {
                                int fragment_col = state_col_low + fragment * 64;
                                int operand_index = token_1 * 128 + fragment_col;
                                unsigned int b_carriers[4];
                                unsigned int c_carriers[4];
                                float b_values[8];
                                float c_values[8];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 3]))
                                    : "r"(input_b_addr + operand_index * 2));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 3]))
                                    : "r"(input_c_addr + operand_index * 2));
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                                        : "r"(b_carriers[_pair]));
                                }
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                                        : "r"(c_carriers[_pair]));
                                }
                                #pragma unroll
                                for (int pair = 0; pair < 4; pair++) {
                                    int item = pair * 2;
                                    int state_item = fragment * 8 + item;
                                    float2 _f2_4 = make_float2(state_values[state_item], state_values[state_item + 1]);
                                    float2 state_pair = _f2_4;
                                    float2 _f2_5 = make_float2(b_values[item], b_values[item + 1]);
                                    float2 b_pair = _f2_5;
                                    float2 _f2_6 = make_float2(c_values[item], c_values[item + 1]);
                                    float2 c_pair = _f2_6;
                                    float2 db_pair = mul_f32x2(b_pair, dt_pair);
                                    float2 dbx_pair = mul_f32x2(db_pair, x_pair);
                                    state_pair = fma_f32x2(state_pair, decay_pair, dbx_pair);
                                    partial_pair = fma_f32x2(state_pair, c_pair, partial_pair);
                                    state_values[state_item] = state_pair.x;
                                    state_values[state_item + 1] = state_pair.y;
                                }
                            }
                            if (cache_intermediate != 0) {
                                if (tile_valid) {
                                    if (source_slot >= 0) {
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(state_values[0 + 0], state_values[0 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(state_values[0 + 2], state_values[0 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(state_values[0 + 4], state_values[0 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(state_values[0 + 6], state_values[0 + 7]);
                                            uint4 _st_v4_0 = *reinterpret_cast<uint4*>(&_pk[0]);
                                            asm volatile(
                                                "st.global.L1::no_allocate.v4.b32 [%0], {%1, %2, %3, %4};"
                                                :: "l"(&((__nv_bfloat16*)(intermediate_state))[intermediate_index_low + 0]), "r"(_st_v4_0.x), "r"(_st_v4_0.y), "r"(_st_v4_0.z), "r"(_st_v4_0.w) : "memory");
                                        }
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(state_values[8 + 0], state_values[8 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(state_values[8 + 2], state_values[8 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(state_values[8 + 4], state_values[8 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(state_values[8 + 6], state_values[8 + 7]);
                                            uint4 _st_v4_0 = *reinterpret_cast<uint4*>(&_pk[0]);
                                            asm volatile(
                                                "st.global.L1::no_allocate.v4.b32 [%0], {%1, %2, %3, %4};"
                                                :: "l"(&((__nv_bfloat16*)(intermediate_state))[intermediate_index_high + 0]), "r"(_st_v4_0.x), "r"(_st_v4_0.y), "r"(_st_v4_0.z), "r"(_st_v4_0.w) : "memory");
                                        }
                                    }
                                }
                                intermediate_index_low += intermediate_step_stride;
                                intermediate_index_high += intermediate_step_stride;
                            }
                            float row_sum = partial_pair.x + partial_pair.y;
                            float _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, row_sum, 4, 8);
                            row_sum += _shfl_down_0;
                            float _shfl_down_1 = __shfl_down_sync(0xFFFFFFFF, row_sum, 2, 8);
                            row_sum += _shfl_down_1;
                            float _shfl_down_2 = __shfl_down_sync(0xFFFFFFFF, row_sum, 1, 8);
                            row_sum += _shfl_down_2;
                            if (member == 0) {
                                if (tile_valid) {
                                    int output_index = ((batch * 6 + token_1) * nheads + head) * 64 + dim_index;
                                    output[output_index] = row_sum + d_value * x_value;
                                }
                            }
                        }
                    }
                    if (elect_sync()) {
                        if (cohort == 0) {
                            mbarrier_arrive(state_empty_0_addr + (barrier_state_stage) * 8);
                        } else {
                            mbarrier_arrive(state_empty_1_addr + (barrier_state_stage) * 8);
                        }
                    }
                }
                if (elect_sync()) {
                    if (cohort == 0) {
                        mbarrier_arrive(inputs_empty_0_addr + (state_phase) * 8);
                    } else {
                        mbarrier_arrive(inputs_empty_1_addr + (state_phase) * 8);
                    }
                }
                state_phase += 1;
                if (state_phase == 2) { state_phase = 0; input_phase ^= 1; }
            }
        }
    }
    // ---- Role: producer ----
    if (warp == 8) {
        { // producer_main
            if (elect_sync()) {
                unsigned int input_phase_1 = 0;
                unsigned int state_phase_1 = 0;
                #pragma unroll 1
                for (int work_group_1 = bid; work_group_1 < (total_tiles + 2 - 1) / 2; work_group_1 += num_bids) {
                    int work_tile_0 = work_group_1 * 2;
                    int batch_0 = work_tile_0 / nheads;
                    int head_0 = work_tile_0 % nheads;
                    int heads_per_group = nheads / ngroups;
                    int group_0 = head_0 / heads_per_group;
                    long long source_slot_0 = state_batch_indices[batch_0];
                    mbarrier_wait(inputs_empty_0_addr + (state_phase_1) * 8, input_phase_1 ^ 1);
                    mbarrier_wait(state_empty_0_addr, state_phase_1 ^ 1);
                    tma_4d_gmem2smem(s_b_addr + state_phase_1 * 1536, b_tma, 0, group_0, 0, batch_0, state_full_0_addr);
                    tma_4d_gmem2smem(s_c_addr + state_phase_1 * 1536, c_tma, 0, group_0, 0, batch_0, state_full_0_addr);
                    tma_4d_gmem2smem(s_x_addr + state_phase_1 * 768, x_tma, 0, head_0, 0, batch_0, state_full_0_addr);
                    tma_4d_gmem2smem(s_state_addr, state_tma, 0, 0, head_0, (int)source_slot_0, state_full_0_addr);
                    mbarrier_arrive_expect_tx(state_full_0_addr, 12032);
                    mbarrier_wait(state_empty_0_addr + 8, state_phase_1 ^ 1);
                    tma_4d_gmem2smem(s_state_addr + 8192, state_tma, 0, 32, head_0, (int)source_slot_0, state_full_0_addr + 8);
                    mbarrier_arrive_expect_tx(state_full_0_addr + 8, 8192);
                    int raw_work_tile_1 = work_tile_0 + 1;
                    int work_tile_1 = raw_work_tile_1;
                    if (raw_work_tile_1 >= total_tiles) {
                        work_tile_1 = total_tiles - 1;
                    }
                    int batch_1 = work_tile_1 / nheads;
                    int head_1 = work_tile_1 % nheads;
                    int group_1 = head_1 / heads_per_group;
                    long long source_slot_1 = state_batch_indices[batch_1];
                    unsigned int input_stage_1 = 2 + state_phase_1;
                    mbarrier_wait(inputs_empty_1_addr + (state_phase_1) * 8, input_phase_1 ^ 1);
                    mbarrier_wait(state_empty_1_addr, state_phase_1 ^ 1);
                    tma_4d_gmem2smem(s_b_addr + input_stage_1 * 1536, b_tma, 0, group_1, 0, batch_1, state_full_1_addr);
                    tma_4d_gmem2smem(s_c_addr + input_stage_1 * 1536, c_tma, 0, group_1, 0, batch_1, state_full_1_addr);
                    tma_4d_gmem2smem(s_x_addr + input_stage_1 * 768, x_tma, 0, head_1, 0, batch_1, state_full_1_addr);
                    tma_4d_gmem2smem(s_state_addr + 16384, state_tma, 0, 0, head_1, (int)source_slot_1, state_full_1_addr);
                    mbarrier_arrive_expect_tx(state_full_1_addr, 12032);
                    mbarrier_wait(state_empty_1_addr + 8, state_phase_1 ^ 1);
                    tma_4d_gmem2smem(s_state_addr + 24576, state_tma, 0, 32, head_1, (int)source_slot_1, state_full_1_addr + 8);
                    mbarrier_arrive_expect_tx(state_full_1_addr + 8, 8192);
                    state_phase_1 += 1;
                    if (state_phase_1 == 2) { state_phase_1 = 0; input_phase_1 ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"

