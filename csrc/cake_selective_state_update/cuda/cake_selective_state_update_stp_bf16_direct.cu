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
#define NUM_STATE_PIPE_STAGES 4
#define SMEM_S_STATE_OFF 1024
#define SMEM_S_STATE_STAGE_BYTES 8192
#define SMEM_S_STATE_STRIDE 8192
#define SMEM_S_B_OFF 33792
#define SMEM_S_B_STAGE_BYTES 256
#define SMEM_S_B_STRIDE 256
#define SMEM_S_C_OFF 34048
#define SMEM_S_C_STAGE_BYTES 256
#define SMEM_S_C_STRIDE 256
#define SMEM_S_HEAD_SCALARS_OFF 34304
#define SMEM_S_HEAD_SCALARS_STAGE_BYTES 48
#define SMEM_S_HEAD_SCALARS_STRIDE 48
#define SMEM_TOTAL 34432
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


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(288, 4) void
kernel_cake_selective_state_update_stp_bf16_direct(CakeTensorMap const* state_tma, __nv_bfloat16* __restrict__ x, float* __restrict__ dt, float* __restrict__ A, __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C, float* __restrict__ D, __nv_bfloat16* __restrict__ z, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ output, long long* __restrict__ state_batch_indices, long long* __restrict__ dst_state_batch_indices, int nheads, int ngroups, int head_tiles, int dt_softplus, int has_z, int disable_state_update)
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
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* s_state = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int s_state_addr = smem + 1024;
    __nv_bfloat16* s_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int s_b_addr = smem + 33792;
    __nv_bfloat16* s_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 34048);
    const int s_c_addr = smem + 34048;
    float* s_head_scalars = reinterpret_cast<float*>(smem_raw + 34304);
    const int s_head_scalars_addr = smem + 34304;

    // Mbarrier init (2 groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'state_pipe' ---
            // state_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // state_updated: 4 barriers, init_count=8
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            mbarrier_init(smem + 48, 8);
            mbarrier_init(smem + 56, 8);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncthreads();

    const int mbar_base = smem;
    #define state_full_addr (mbar_base + 0)
    #define state_updated_addr (mbar_base + 32)

    // ---- Role: consumers ----
    if (warp <= 7) {
        { // consumers_main
            int head_tile = 0;
            int batch_group = 0;
            {
                {
                    head_tile = bid % head_tiles;
                    batch_group = bid / head_tiles;
                }
            }
            int batch = batch_group / ngroups;
            int group = batch_group % ngroups;
            int heads_per_group = 0;
            {
                {
                    heads_per_group = nheads / ngroups;
                }
            }
            int first_local_head = head_tile * 4;
            int work_heads = heads_per_group - first_local_head;
            if (work_heads > 4) {
                work_heads = 4;
            }
            int warp_id_in_role = (warp - 0);
            int local_warp = warp_id_in_role;
            int row_in_warp = lane & 15;
            int member = lane >> 4;
            int dim_index = local_warp * 16 + row_in_warp;
            if (local_warp == 0) {
                if (lane < 16) {
                    int col = lane * 8;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_b_addr + (unsigned int)(col * 2)), "l"(B + ((batch * ngroups + group) * 128 + col)));
                }
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 0;");
            }
            if (local_warp == 1) {
                if (lane < 16) {
                    int col_1 = lane * 8;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_c_addr + (unsigned int)(col_1 * 2)), "l"(C + ((batch * ngroups + group) * 128 + col_1)));
                }
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 0;");
            }
            if (local_warp == 2) {
                if (work_heads > lane) {
                    int scalar_head_offset = lane;
                    int scalar_local_head = first_local_head + scalar_head_offset;
                    int scalar_head = group * heads_per_group + scalar_local_head;
                    float scalar_dt = dt[batch * nheads + scalar_head];
                    scalar_dt += dt_bias[scalar_head];
                    if (dt_softplus != 0) {
                        if (scalar_dt <= 20.0f) {
                            float _expf_0 = __expf(scalar_dt);
                            float _log2_0;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(1.0f + _expf_0));
                            scalar_dt = _log2_0 * 0.6931471805599453f;
                        }
                    }
                    int published_scalar_base = scalar_head_offset * 3;
                    s_head_scalars[published_scalar_base] = scalar_dt;
                    float _exp_0 = expf(A[scalar_head] * scalar_dt);
                    s_head_scalars[published_scalar_base + 1] = _exp_0;
                    s_head_scalars[published_scalar_base + 2] = D[scalar_head];
                }
            }
            asm volatile("barrier.sync 8, 256;" ::: "memory");
            #pragma unroll 4
            for (int head_offset = 0; head_offset < work_heads; head_offset++) {
                int local_head = first_local_head + head_offset;
                int head = group * heads_per_group + local_head;
                unsigned int work_phase = (unsigned int)(head_offset & 1);
                int head_scalar_base = head_offset * 3;
                float dt_value = s_head_scalars[head_scalar_base];
                float decay = s_head_scalars[head_scalar_base + 1];
                float d_value = s_head_scalars[head_scalar_base + 2];
                int x_index = (batch * nheads + head) * 128 + dim_index;
                float x_value = (float)x[x_index];
                float z_value = 0.0f;
                if (has_z != 0) {
                    z_value = (float)z[x_index];
                }
                float2 _f2_0 = make_float2(decay, decay);
                float2 decay_pair = _f2_0;
                float2 _f2_1 = make_float2(dt_value, dt_value);
                float2 dt_pair = _f2_1;
                float2 _f2_2 = make_float2(x_value, x_value);
                float2 x_pair = _f2_2;
                float2 _f2_3 = make_float2(0.0f, 0.0f);
                float2 partial_pair = _f2_3;
                unsigned int state_carrier[1];
                unsigned int b_carrier[1];
                unsigned int c_carrier[1];
                float state_values[2];
                float b_values[2];
                float c_values[2];
                #pragma unroll
                for (int stage = 0; stage < 4; stage++) {
                    mbarrier_wait(state_full_addr + (stage) * 8, work_phase);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    #pragma unroll
                    for (int pair = 0; pair < 8; pair++) {
                        int base_col = member * 16 + pair * 2;
                        int sequence_index = row_in_warp * 32 + base_col;
                        int bank_cycle = sequence_index / 64;
                        int local_col = (base_col + 2 * bank_cycle) % 32;
                        int global_col = stage * 32 + local_col;
                        int state_index = stage * 4096 + dim_index * 32 + local_col;
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&state_carrier[0])) : "r"(s_state_addr + (unsigned int)(state_index * 2)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&b_carrier[0])) : "r"(s_b_addr + (unsigned int)(global_col * 2)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&c_carrier[0])) : "r"(s_c_addr + (unsigned int)(global_col * 2)));
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&state_values[_pair * 2])[0]), "=f"((&state_values[_pair * 2])[1])
                                : "r"(state_carrier[_pair]));
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                                : "r"(b_carrier[_pair]));
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                                : "r"(c_carrier[_pair]));
                        }
                        float2 _f2_4 = make_float2(state_values[0], state_values[1]);
                        float2 state_pair = _f2_4;
                        float2 _f2_5 = make_float2(b_values[0], b_values[1]);
                        float2 b_pair = _f2_5;
                        float2 _f2_6 = make_float2(c_values[0], c_values[1]);
                        float2 c_pair = _f2_6;
                        float2 db_pair = mul_f32x2(b_pair, dt_pair);
                        float2 dbx_pair = mul_f32x2(db_pair, x_pair);
                        state_pair = fma_f32x2(state_pair, decay_pair, dbx_pair);
                        partial_pair = fma_f32x2(state_pair, c_pair, partial_pair);
                        state_values[0] = state_pair.x;
                        state_values[1] = state_pair.y;
                        uint32_t state_values_bf16[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_values[_lp*2 + 0], state_values[_lp*2+1 + 0]));
                            state_values_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(s_state_addr + (unsigned int)(state_index * 2)), "r"((state_values_bf16[0])));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(state_updated_addr + (stage) * 8);
                    }
                }
                float partial = partial_pair.x + partial_pair.y;
                float _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, partial, 16, 32);
                partial += _shfl_down_0;
                if (member == 0) {
                    float _fma_0 = __fmaf_rn(d_value, x_value, partial);
                    float result = _fma_0;
                    if (has_z != 0) {
                        float _exp_1 = expf(-z_value);
                        float sigmoid_z = 1.0f / (1.0f + _exp_1);
                        result *= z_value * sigmoid_z;
                    }
                    output[x_index] = result;
                }
            }
        }
    }
    // ---- Role: producer ----
    if (warp == 8) {
        { // producer_main
            int head_tile_1 = 0;
            int batch_group_1 = 0;
            {
                {
                    head_tile_1 = bid % head_tiles;
                    batch_group_1 = bid / head_tiles;
                }
            }
            int batch_1 = batch_group_1 / ngroups;
            int group_1 = batch_group_1 % ngroups;
            int heads_per_group_1 = 0;
            {
                {
                    heads_per_group_1 = nheads / ngroups;
                }
            }
            int first_local_head_1 = head_tile_1 * 4;
            int work_heads_1 = heads_per_group_1 - first_local_head_1;
            if (work_heads_1 > 4) {
                work_heads_1 = 4;
            }
            long long source_slot = state_batch_indices[batch_1];
            long long destination_slot = dst_state_batch_indices[batch_1];
            if (elect_sync()) {
                int first_head = group_1 * heads_per_group_1 + first_local_head_1;
                #pragma unroll
                for (int stage_1 = 0; stage_1 < 4; stage_1++) {
                    tma_4d_gmem2smem(s_state_addr + (unsigned int)(stage_1 * 8192), state_tma, stage_1 * 32, 0, first_head, (int)source_slot, state_full_addr + (stage_1) * 8);
                    mbarrier_arrive_expect_tx(state_full_addr + (stage_1) * 8, 8192);
                }
                #pragma unroll 4
                for (int head_offset_1 = 0; head_offset_1 < work_heads_1; head_offset_1++) {
                    int local_head_1 = first_local_head_1 + head_offset_1;
                    int head_1 = group_1 * heads_per_group_1 + local_head_1;
                    unsigned int work_phase_1 = (unsigned int)(head_offset_1 & 1);
                    int next_head_valid = ((work_heads_1 > head_offset_1 + 1) ? 1 : 0);
                    #pragma unroll
                    for (int stage_2 = 0; stage_2 < 4; stage_2++) {
                        mbarrier_wait(state_updated_addr + (stage_2) * 8, work_phase_1);
                        if (disable_state_update == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_4d(state_tma, stage_2 * 32, 0, head_1, (int)destination_slot, s_state_addr + (unsigned int)(stage_2 * 8192));
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        if (next_head_valid != 0) {
                            int next_head = head_1 + 1;
                            tma_4d_gmem2smem(s_state_addr + (unsigned int)(stage_2 * 8192), state_tma, stage_2 * 32, 0, next_head, (int)source_slot, state_full_addr + (stage_2) * 8);
                            mbarrier_arrive_expect_tx(state_full_addr + (stage_2) * 8, 8192);
                        }
                    }
                }
                if (disable_state_update == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"

