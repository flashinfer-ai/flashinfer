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
#define NUM_MAIN_STAGES 1
#define SMEM_S_B_OFF 0
#define SMEM_S_B_STAGE_BYTES 512
#define SMEM_S_B_STRIDE 512
#define SMEM_S_C_OFF 512
#define SMEM_S_C_STAGE_BYTES 512
#define SMEM_S_C_STRIDE 512
#define SMEM_S_X_OFF 1024
#define SMEM_S_X_STAGE_BYTES 512
#define SMEM_S_X_STRIDE 512
#define SMEM_S_DT_OFF 1536
#define SMEM_S_DT_STAGE_BYTES 8
#define SMEM_S_DT_STRIDE 8
#define SMEM_TOTAL 1664
#define THREADS 128

#include <math_constants.h>

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

extern "C" {

__global__ __launch_bounds__(128, 8) void
kernel_cake_selective_state_update_mtp_short(__nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ x, float* __restrict__ dt, float* __restrict__ A, __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C, float* __restrict__ D, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ output, long long* __restrict__ state_batch_indices, int batch_size, int nheads, int dim, int dstate, int ngroups, int token_steps, unsigned long long state_stride_slot, int dt_softplus, long long pad_slot_id)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* s_B = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_B_addr = smem + 0;
    __nv_bfloat16* s_C = reinterpret_cast<__nv_bfloat16*>(smem_raw + 512);
    const int s_C_addr = smem + 512;
    __nv_bfloat16* s_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int s_x_addr = smem + 1024;
    float* s_dt = reinterpret_cast<float*>(smem_raw + 1536);
    const int s_dt_addr = smem + 1536;

    // === Task calls (dependency order) ===
    int batch = bid / nheads;
    int head = bid % nheads;
    int heads_per_group = nheads / ngroups;
    int group = head / heads_per_group;
    int token_base = batch * token_steps;
    long long source_slot = state_batch_indices[batch];
    int pack = lane;
    int step = pack / 16;
    int col = pack % 16 * 8;
    int source_step = step;
    if (source_step >= token_steps) {
        source_step = 0;
    }
    int source_index = ((batch * token_steps + source_step) * ngroups + group) * dstate + col;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %0, 0;\n\t"
        "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
        "}"
        :: "r"((warp == 0) ? 1 : 0), "r"(s_B_addr + (unsigned int)((step * 128 + col) * 2)), "l"(B + source_index));
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %0, 0;\n\t"
        "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
        "}"
        :: "r"((warp == 1) ? 1 : 0), "r"(s_C_addr + (unsigned int)((step * 128 + col) * 2)), "l"(C + source_index));
    int step_0 = warp;
    int source_step_1 = step_0;
    if (source_step_1 >= token_steps) {
        source_step_1 = 0;
    }
    int col_2 = lane * 8;
    int source_index_3 = ((token_base + source_step_1) * nheads + head) * dim + col_2;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %0, 0;\n\t"
        "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
        "}"
        :: "r"((warp < 2 && lane < 16) ? 1 : 0), "r"(s_x_addr + (unsigned int)((source_step_1 * 128 + col_2) * 2)), "l"(x + source_index_3));
    asm volatile("cp.async.commit_group;");
    float dt_value = 0.0f;
    if (tid < token_steps) {
        int step_1 = tid;
        dt_value = dt[(token_base + step_1) * nheads + head];
        dt_value += dt_bias[head];
        if (dt_softplus != 0) {
            float _exp_0 = expf(dt_value);
            float _log1p_0 = log1pf(_exp_0);
            dt_value = _log1p_0;
        }
    }
    asm volatile("cp.async.wait_group 0;");
    asm volatile("fence.proxy.async;");
    if (tid < token_steps) {
        s_dt[tid] = dt_value;
    }
    __syncthreads();
    int member = lane % 4;
    int row_in_warp = lane / 4;
    int local_row = warp * 8 + row_in_warp;
    float a_value = A[head];
    float d_value = D[head];
    float dt_value_0 = s_dt[0];
    float _exp_1 = expf(a_value * dt_value_0);
    float decay_0 = _exp_1;
    float dt_value_1 = 0.0f;
    float decay_1 = 0.0f;
    if (token_steps > 1) {
        dt_value_1 = s_dt[1];
        float _exp_2 = expf(a_value * dt_value_1);
        decay_1 = _exp_2;
    }
    unsigned int b_carriers[8];
    unsigned int c_carriers[8];
    float b_values[2];
    float c_values[2];
    #pragma unroll 1
    for (int tile = 0; tile < 4; tile++) {
        int dim_index = tile * 32 + local_row;
        float state_values[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            state_values[i] = 0.0f;
        }
        int row_offset_i32 = (head * dim + dim_index) * dstate;
        unsigned long long row_offset = (unsigned long long)row_offset_i32;
        if (source_slot != pad_slot_id) {
            #pragma unroll
            for (int state_tile = 0; state_tile < 4; state_tile++) {
                int col_0 = state_tile * 32 + member * 8;
                unsigned long long state_index = (unsigned long long)source_slot * state_stride_slot + row_offset + (unsigned long long)col_0;
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(state + state_index);
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
                                : "=f"((&state_values[state_tile * 8 + _blk * 8 + _pair * 2])[0]), "=f"((&state_values[state_tile * 8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int pair_base = 0; pair_base < 2; pair_base += 2) {
            float2 _f2_0 = make_float2(decay_0, decay_0);
            float2 _f2_1 = make_float2(dt_value_0 * (float)s_x[pair_base * 128 + dim_index], dt_value_0 * (float)s_x[pair_base * 128 + dim_index]);
            float2 _f2_2 = make_float2(0.0f, 0.0f);
            float2 _f2_3 = make_float2(0.0f, 0.0f);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 3]))
                : "r"(s_B_addr + (unsigned int)((pair_base * 128 + member * 8) * 2)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 3]))
                : "r"(s_C_addr + (unsigned int)((pair_base * 128 + member * 8) * 2)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[4])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 3]))
                : "r"(s_B_addr + (unsigned int)((pair_base * 128 + (32 + member * 8)) * 2)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[4])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 3]))
                : "r"(s_C_addr + (unsigned int)((pair_base * 128 + (32 + member * 8)) * 2)));
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[_pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[_pair]));
            }
            float2 _f2_4 = make_float2(state_values[0], state_values[1]);
            float2 _f2_5 = make_float2(b_values[0], b_values[1]);
            float2 _f2_6 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_0_0 = fma_f32x2(_f2_4, _f2_0, mul_f32x2(_f2_5, _f2_1));
            float2 _projection_pair_0_0 = fma_f32x2(_state_pair_0_0, _f2_6, _f2_2);
            state_values[0] = _state_pair_0_0.x;
            state_values[1] = _state_pair_0_0.y;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[4 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[4 + _pair]));
            }
            float2 _f2_7 = make_float2(state_values[8], state_values[9]);
            float2 _f2_8 = make_float2(b_values[0], b_values[1]);
            float2 _f2_9 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_4_0 = fma_f32x2(_f2_7, _f2_0, mul_f32x2(_f2_8, _f2_1));
            float2 _projection_pair_4_0 = fma_f32x2(_state_pair_4_0, _f2_9, _f2_3);
            state_values[8] = _state_pair_4_0.x;
            state_values[9] = _state_pair_4_0.y;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[1 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[1 + _pair]));
            }
            float2 _f2_10 = make_float2(state_values[2], state_values[3]);
            float2 _f2_11 = make_float2(b_values[0], b_values[1]);
            float2 _f2_12 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_0_1 = fma_f32x2(_f2_10, _f2_0, mul_f32x2(_f2_11, _f2_1));
            float2 _projection_pair_0_1 = fma_f32x2(_state_pair_0_1, _f2_12, _projection_pair_0_0);
            state_values[2] = _state_pair_0_1.x;
            state_values[3] = _state_pair_0_1.y;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[5 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[5 + _pair]));
            }
            float2 _f2_13 = make_float2(state_values[10], state_values[11]);
            float2 _f2_14 = make_float2(b_values[0], b_values[1]);
            float2 _f2_15 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_4_1 = fma_f32x2(_f2_13, _f2_0, mul_f32x2(_f2_14, _f2_1));
            float2 _projection_pair_4_1 = fma_f32x2(_state_pair_4_1, _f2_15, _projection_pair_4_0);
            state_values[10] = _state_pair_4_1.x;
            state_values[11] = _state_pair_4_1.y;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[2 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[2 + _pair]));
            }
            float2 _f2_16 = make_float2(state_values[4], state_values[5]);
            float2 _f2_17 = make_float2(b_values[0], b_values[1]);
            float2 _f2_18 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_0_2 = fma_f32x2(_f2_16, _f2_0, mul_f32x2(_f2_17, _f2_1));
            float2 _projection_pair_0_2 = fma_f32x2(_state_pair_0_2, _f2_18, _projection_pair_0_1);
            state_values[4] = _state_pair_0_2.x;
            state_values[5] = _state_pair_0_2.y;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[6 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[6 + _pair]));
            }
            float2 _f2_19 = make_float2(state_values[12], state_values[13]);
            float2 _f2_20 = make_float2(b_values[0], b_values[1]);
            float2 _f2_21 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_4_2 = fma_f32x2(_f2_19, _f2_0, mul_f32x2(_f2_20, _f2_1));
            float2 _projection_pair_4_2 = fma_f32x2(_state_pair_4_2, _f2_21, _projection_pair_4_1);
            state_values[12] = _state_pair_4_2.x;
            state_values[13] = _state_pair_4_2.y;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[3 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[3 + _pair]));
            }
            float2 _f2_22 = make_float2(state_values[6], state_values[7]);
            float2 _f2_23 = make_float2(b_values[0], b_values[1]);
            float2 _f2_24 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_0_3 = fma_f32x2(_f2_22, _f2_0, mul_f32x2(_f2_23, _f2_1));
            float2 _projection_pair_0_3 = fma_f32x2(_state_pair_0_3, _f2_24, _projection_pair_0_2);
            state_values[6] = _state_pair_0_3.x;
            state_values[7] = _state_pair_0_3.y;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[7 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[7 + _pair]));
            }
            float2 _f2_25 = make_float2(state_values[14], state_values[15]);
            float2 _f2_26 = make_float2(b_values[0], b_values[1]);
            float2 _f2_27 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_4_3 = fma_f32x2(_f2_25, _f2_0, mul_f32x2(_f2_26, _f2_1));
            float2 _projection_pair_4_3 = fma_f32x2(_state_pair_4_3, _f2_27, _projection_pair_4_2);
            state_values[14] = _state_pair_4_3.x;
            state_values[15] = _state_pair_4_3.y;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 3]))
                : "r"(s_B_addr + (unsigned int)((pair_base * 128 + (64 + member * 8)) * 2)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 3]))
                : "r"(s_C_addr + (unsigned int)((pair_base * 128 + (64 + member * 8)) * 2)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[4])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 3]))
                : "r"(s_B_addr + (unsigned int)((pair_base * 128 + (96 + member * 8)) * 2)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[4])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 3]))
                : "r"(s_C_addr + (unsigned int)((pair_base * 128 + (96 + member * 8)) * 2)));
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[_pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[_pair]));
            }
            float2 _f2_28 = make_float2(state_values[16], state_values[17]);
            float2 _f2_29 = make_float2(b_values[0], b_values[1]);
            float2 _f2_30 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_0_0_0 = fma_f32x2(_f2_28, _f2_0, mul_f32x2(_f2_29, _f2_1));
            float2 _projection_pair_0_0_1 = fma_f32x2(_state_pair_0_0_0, _f2_30, _projection_pair_0_3);
            state_values[16] = _state_pair_0_0_0.x;
            state_values[17] = _state_pair_0_0_0.y;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[4 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[4 + _pair]));
            }
            float2 _f2_31 = make_float2(state_values[24], state_values[25]);
            float2 _f2_32 = make_float2(b_values[0], b_values[1]);
            float2 _f2_33 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_4_0_2 = fma_f32x2(_f2_31, _f2_0, mul_f32x2(_f2_32, _f2_1));
            float2 _projection_pair_4_0_3 = fma_f32x2(_state_pair_4_0_2, _f2_33, _projection_pair_4_3);
            state_values[24] = _state_pair_4_0_2.x;
            state_values[25] = _state_pair_4_0_2.y;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[1 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[1 + _pair]));
            }
            float2 _f2_34 = make_float2(state_values[18], state_values[19]);
            float2 _f2_35 = make_float2(b_values[0], b_values[1]);
            float2 _f2_36 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_0_1_4 = fma_f32x2(_f2_34, _f2_0, mul_f32x2(_f2_35, _f2_1));
            float2 _projection_pair_0_1_5 = fma_f32x2(_state_pair_0_1_4, _f2_36, _projection_pair_0_0_1);
            state_values[18] = _state_pair_0_1_4.x;
            state_values[19] = _state_pair_0_1_4.y;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[5 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[5 + _pair]));
            }
            float2 _f2_37 = make_float2(state_values[26], state_values[27]);
            float2 _f2_38 = make_float2(b_values[0], b_values[1]);
            float2 _f2_39 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_4_1_6 = fma_f32x2(_f2_37, _f2_0, mul_f32x2(_f2_38, _f2_1));
            float2 _projection_pair_4_1_7 = fma_f32x2(_state_pair_4_1_6, _f2_39, _projection_pair_4_0_3);
            state_values[26] = _state_pair_4_1_6.x;
            state_values[27] = _state_pair_4_1_6.y;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[2 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[2 + _pair]));
            }
            float2 _f2_40 = make_float2(state_values[20], state_values[21]);
            float2 _f2_41 = make_float2(b_values[0], b_values[1]);
            float2 _f2_42 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_0_2_8 = fma_f32x2(_f2_40, _f2_0, mul_f32x2(_f2_41, _f2_1));
            float2 _projection_pair_0_2_9 = fma_f32x2(_state_pair_0_2_8, _f2_42, _projection_pair_0_1_5);
            state_values[20] = _state_pair_0_2_8.x;
            state_values[21] = _state_pair_0_2_8.y;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[6 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[6 + _pair]));
            }
            float2 _f2_43 = make_float2(state_values[28], state_values[29]);
            float2 _f2_44 = make_float2(b_values[0], b_values[1]);
            float2 _f2_45 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_4_2_10 = fma_f32x2(_f2_43, _f2_0, mul_f32x2(_f2_44, _f2_1));
            float2 _projection_pair_4_2_11 = fma_f32x2(_state_pair_4_2_10, _f2_45, _projection_pair_4_1_7);
            state_values[28] = _state_pair_4_2_10.x;
            state_values[29] = _state_pair_4_2_10.y;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[3 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[3 + _pair]));
            }
            float2 _f2_46 = make_float2(state_values[22], state_values[23]);
            float2 _f2_47 = make_float2(b_values[0], b_values[1]);
            float2 _f2_48 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_0_3_12 = fma_f32x2(_f2_46, _f2_0, mul_f32x2(_f2_47, _f2_1));
            float2 _projection_pair_0_3_13 = fma_f32x2(_state_pair_0_3_12, _f2_48, _projection_pair_0_2_9);
            state_values[22] = _state_pair_0_3_12.x;
            state_values[23] = _state_pair_0_3_12.y;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                    : "r"(b_carriers[7 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                    : "r"(c_carriers[7 + _pair]));
            }
            float2 _f2_49 = make_float2(state_values[30], state_values[31]);
            float2 _f2_50 = make_float2(b_values[0], b_values[1]);
            float2 _f2_51 = make_float2(c_values[0], c_values[1]);
            float2 _state_pair_4_3_14 = fma_f32x2(_f2_49, _f2_0, mul_f32x2(_f2_50, _f2_1));
            float2 _projection_pair_4_3_15 = fma_f32x2(_state_pair_4_3_14, _f2_51, _projection_pair_4_2_11);
            state_values[30] = _state_pair_4_3_14.x;
            state_values[31] = _state_pair_4_3_14.y;
            float _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, _projection_pair_0_3_13.x + _projection_pair_0_3_13.y + (_projection_pair_4_3_15.x + _projection_pair_4_3_15.y), 2, 4);
            float _shfl_down_1 = __shfl_down_sync(0xFFFFFFFF, _projection_pair_0_3_13.x + _projection_pair_0_3_13.y + (_projection_pair_4_3_15.x + _projection_pair_4_3_15.y) + _shfl_down_0, 1, 4);
            if (member == 0) {
                int output_index = ((token_base + pair_base) * nheads + head) * dim + dim_index;
                output[output_index] = _projection_pair_0_3_13.x + _projection_pair_0_3_13.y + (_projection_pair_4_3_15.x + _projection_pair_4_3_15.y) + _shfl_down_0 + _shfl_down_1 + d_value * (float)s_x[pair_base * 128 + dim_index];
            }
            if (pair_base + 1 >= token_steps) {
                if (source_slot != pad_slot_id) {
                    int publication_row_offset_i32 = (head * dim + dim_index) * dstate;
                    unsigned long long publication_row_offset = (unsigned long long)publication_row_offset_i32;
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(state_values[0 + 0], state_values[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(state_values[0 + 2], state_values[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(state_values[0 + 4], state_values[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(state_values[0 + 6], state_values[0 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[(unsigned long long)source_slot * state_stride_slot + publication_row_offset + (unsigned long long)(member * 8) + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(state_values[8 + 0], state_values[8 + 1]);
                        _pk[1] = __floats2bfloat162_rn(state_values[8 + 2], state_values[8 + 3]);
                        _pk[2] = __floats2bfloat162_rn(state_values[8 + 4], state_values[8 + 5]);
                        _pk[3] = __floats2bfloat162_rn(state_values[8 + 6], state_values[8 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[(unsigned long long)source_slot * state_stride_slot + publication_row_offset + (unsigned long long)(32 + member * 8) + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(state_values[16 + 0], state_values[16 + 1]);
                        _pk[1] = __floats2bfloat162_rn(state_values[16 + 2], state_values[16 + 3]);
                        _pk[2] = __floats2bfloat162_rn(state_values[16 + 4], state_values[16 + 5]);
                        _pk[3] = __floats2bfloat162_rn(state_values[16 + 6], state_values[16 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[(unsigned long long)source_slot * state_stride_slot + publication_row_offset + (unsigned long long)(64 + member * 8) + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(state_values[24 + 0], state_values[24 + 1]);
                        _pk[1] = __floats2bfloat162_rn(state_values[24 + 2], state_values[24 + 3]);
                        _pk[2] = __floats2bfloat162_rn(state_values[24 + 4], state_values[24 + 5]);
                        _pk[3] = __floats2bfloat162_rn(state_values[24 + 6], state_values[24 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[(unsigned long long)source_slot * state_stride_slot + publication_row_offset + (unsigned long long)(96 + member * 8) + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
            if (pair_base + 1 < token_steps) {
                float2 _f2_52 = make_float2(decay_1, decay_1);
                float2 _f2_53 = make_float2(dt_value_1 * (float)s_x[(pair_base + 1) * 128 + dim_index], dt_value_1 * (float)s_x[(pair_base + 1) * 128 + dim_index]);
                float2 _f2_54 = make_float2(0.0f, 0.0f);
                float2 _f2_55 = make_float2(0.0f, 0.0f);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 3]))
                    : "r"(s_B_addr + (unsigned int)(((pair_base + 1) * 128 + member * 8) * 2)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 3]))
                    : "r"(s_C_addr + (unsigned int)(((pair_base + 1) * 128 + member * 8) * 2)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[4])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 3]))
                    : "r"(s_B_addr + (unsigned int)(((pair_base + 1) * 128 + (32 + member * 8)) * 2)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[4])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 3]))
                    : "r"(s_C_addr + (unsigned int)(((pair_base + 1) * 128 + (32 + member * 8)) * 2)));
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[_pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[_pair]));
                }
                float2 _f2_56 = make_float2(state_values[0], state_values[1]);
                float2 _f2_57 = make_float2(b_values[0], b_values[1]);
                float2 _f2_58 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_0_0_1 = fma_f32x2(_f2_56, _f2_52, mul_f32x2(_f2_57, _f2_53));
                float2 _projection_pair_0_0_2 = fma_f32x2(_state_pair_0_0_1, _f2_58, _f2_54);
                state_values[0] = _state_pair_0_0_1.x;
                state_values[1] = _state_pair_0_0_1.y;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[4 + _pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[4 + _pair]));
                }
                float2 _f2_59 = make_float2(state_values[8], state_values[9]);
                float2 _f2_60 = make_float2(b_values[0], b_values[1]);
                float2 _f2_61 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_4_0_3 = fma_f32x2(_f2_59, _f2_52, mul_f32x2(_f2_60, _f2_53));
                float2 _projection_pair_4_0_4 = fma_f32x2(_state_pair_4_0_3, _f2_61, _f2_55);
                state_values[8] = _state_pair_4_0_3.x;
                state_values[9] = _state_pair_4_0_3.y;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[1 + _pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[1 + _pair]));
                }
                float2 _f2_62 = make_float2(state_values[2], state_values[3]);
                float2 _f2_63 = make_float2(b_values[0], b_values[1]);
                float2 _f2_64 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_0_1_5 = fma_f32x2(_f2_62, _f2_52, mul_f32x2(_f2_63, _f2_53));
                float2 _projection_pair_0_1_6 = fma_f32x2(_state_pair_0_1_5, _f2_64, _projection_pair_0_0_2);
                state_values[2] = _state_pair_0_1_5.x;
                state_values[3] = _state_pair_0_1_5.y;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[5 + _pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[5 + _pair]));
                }
                float2 _f2_65 = make_float2(state_values[10], state_values[11]);
                float2 _f2_66 = make_float2(b_values[0], b_values[1]);
                float2 _f2_67 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_4_1_7 = fma_f32x2(_f2_65, _f2_52, mul_f32x2(_f2_66, _f2_53));
                float2 _projection_pair_4_1_8 = fma_f32x2(_state_pair_4_1_7, _f2_67, _projection_pair_4_0_4);
                state_values[10] = _state_pair_4_1_7.x;
                state_values[11] = _state_pair_4_1_7.y;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[2 + _pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[2 + _pair]));
                }
                float2 _f2_68 = make_float2(state_values[4], state_values[5]);
                float2 _f2_69 = make_float2(b_values[0], b_values[1]);
                float2 _f2_70 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_0_2_9 = fma_f32x2(_f2_68, _f2_52, mul_f32x2(_f2_69, _f2_53));
                float2 _projection_pair_0_2_10 = fma_f32x2(_state_pair_0_2_9, _f2_70, _projection_pair_0_1_6);
                state_values[4] = _state_pair_0_2_9.x;
                state_values[5] = _state_pair_0_2_9.y;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[6 + _pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[6 + _pair]));
                }
                float2 _f2_71 = make_float2(state_values[12], state_values[13]);
                float2 _f2_72 = make_float2(b_values[0], b_values[1]);
                float2 _f2_73 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_4_2_11 = fma_f32x2(_f2_71, _f2_52, mul_f32x2(_f2_72, _f2_53));
                float2 _projection_pair_4_2_12 = fma_f32x2(_state_pair_4_2_11, _f2_73, _projection_pair_4_1_8);
                state_values[12] = _state_pair_4_2_11.x;
                state_values[13] = _state_pair_4_2_11.y;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[3 + _pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[3 + _pair]));
                }
                float2 _f2_74 = make_float2(state_values[6], state_values[7]);
                float2 _f2_75 = make_float2(b_values[0], b_values[1]);
                float2 _f2_76 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_0_3_13 = fma_f32x2(_f2_74, _f2_52, mul_f32x2(_f2_75, _f2_53));
                float2 _projection_pair_0_3_14 = fma_f32x2(_state_pair_0_3_13, _f2_76, _projection_pair_0_2_10);
                state_values[6] = _state_pair_0_3_13.x;
                state_values[7] = _state_pair_0_3_13.y;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[7 + _pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[7 + _pair]));
                }
                float2 _f2_77 = make_float2(state_values[14], state_values[15]);
                float2 _f2_78 = make_float2(b_values[0], b_values[1]);
                float2 _f2_79 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_4_3_15 = fma_f32x2(_f2_77, _f2_52, mul_f32x2(_f2_78, _f2_53));
                float2 _projection_pair_4_3_16 = fma_f32x2(_state_pair_4_3_15, _f2_79, _projection_pair_4_2_12);
                state_values[14] = _state_pair_4_3_15.x;
                state_values[15] = _state_pair_4_3_15.y;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(0) + 3]))
                    : "r"(s_B_addr + (unsigned int)(((pair_base + 1) * 128 + (64 + member * 8)) * 2)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(0) + 3]))
                    : "r"(s_C_addr + (unsigned int)(((pair_base + 1) * 128 + (64 + member * 8)) * 2)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[4])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_carriers[(4) + 3]))
                    : "r"(s_B_addr + (unsigned int)(((pair_base + 1) * 128 + (96 + member * 8)) * 2)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[4])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c_carriers[(4) + 3]))
                    : "r"(s_C_addr + (unsigned int)(((pair_base + 1) * 128 + (96 + member * 8)) * 2)));
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[_pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[_pair]));
                }
                float2 _f2_80 = make_float2(state_values[16], state_values[17]);
                float2 _f2_81 = make_float2(b_values[0], b_values[1]);
                float2 _f2_82 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_0_0_17 = fma_f32x2(_f2_80, _f2_52, mul_f32x2(_f2_81, _f2_53));
                float2 _projection_pair_0_0_18 = fma_f32x2(_state_pair_0_0_17, _f2_82, _projection_pair_0_3_14);
                state_values[16] = _state_pair_0_0_17.x;
                state_values[17] = _state_pair_0_0_17.y;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[4 + _pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[4 + _pair]));
                }
                float2 _f2_83 = make_float2(state_values[24], state_values[25]);
                float2 _f2_84 = make_float2(b_values[0], b_values[1]);
                float2 _f2_85 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_4_0_19 = fma_f32x2(_f2_83, _f2_52, mul_f32x2(_f2_84, _f2_53));
                float2 _projection_pair_4_0_20 = fma_f32x2(_state_pair_4_0_19, _f2_85, _projection_pair_4_3_16);
                state_values[24] = _state_pair_4_0_19.x;
                state_values[25] = _state_pair_4_0_19.y;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[1 + _pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[1 + _pair]));
                }
                float2 _f2_86 = make_float2(state_values[18], state_values[19]);
                float2 _f2_87 = make_float2(b_values[0], b_values[1]);
                float2 _f2_88 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_0_1_21 = fma_f32x2(_f2_86, _f2_52, mul_f32x2(_f2_87, _f2_53));
                float2 _projection_pair_0_1_22 = fma_f32x2(_state_pair_0_1_21, _f2_88, _projection_pair_0_0_18);
                state_values[18] = _state_pair_0_1_21.x;
                state_values[19] = _state_pair_0_1_21.y;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[5 + _pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[5 + _pair]));
                }
                float2 _f2_89 = make_float2(state_values[26], state_values[27]);
                float2 _f2_90 = make_float2(b_values[0], b_values[1]);
                float2 _f2_91 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_4_1_23 = fma_f32x2(_f2_89, _f2_52, mul_f32x2(_f2_90, _f2_53));
                float2 _projection_pair_4_1_24 = fma_f32x2(_state_pair_4_1_23, _f2_91, _projection_pair_4_0_20);
                state_values[26] = _state_pair_4_1_23.x;
                state_values[27] = _state_pair_4_1_23.y;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[2 + _pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[2 + _pair]));
                }
                float2 _f2_92 = make_float2(state_values[20], state_values[21]);
                float2 _f2_93 = make_float2(b_values[0], b_values[1]);
                float2 _f2_94 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_0_2_25 = fma_f32x2(_f2_92, _f2_52, mul_f32x2(_f2_93, _f2_53));
                float2 _projection_pair_0_2_26 = fma_f32x2(_state_pair_0_2_25, _f2_94, _projection_pair_0_1_22);
                state_values[20] = _state_pair_0_2_25.x;
                state_values[21] = _state_pair_0_2_25.y;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[6 + _pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[6 + _pair]));
                }
                float2 _f2_95 = make_float2(state_values[28], state_values[29]);
                float2 _f2_96 = make_float2(b_values[0], b_values[1]);
                float2 _f2_97 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_4_2_27 = fma_f32x2(_f2_95, _f2_52, mul_f32x2(_f2_96, _f2_53));
                float2 _projection_pair_4_2_28 = fma_f32x2(_state_pair_4_2_27, _f2_97, _projection_pair_4_1_24);
                state_values[28] = _state_pair_4_2_27.x;
                state_values[29] = _state_pair_4_2_27.y;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[3 + _pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[3 + _pair]));
                }
                float2 _f2_98 = make_float2(state_values[22], state_values[23]);
                float2 _f2_99 = make_float2(b_values[0], b_values[1]);
                float2 _f2_100 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_0_3_29 = fma_f32x2(_f2_98, _f2_52, mul_f32x2(_f2_99, _f2_53));
                float2 _projection_pair_0_3_30 = fma_f32x2(_state_pair_0_3_29, _f2_100, _projection_pair_0_2_26);
                state_values[22] = _state_pair_0_3_29.x;
                state_values[23] = _state_pair_0_3_29.y;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carriers[7 + _pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carriers[7 + _pair]));
                }
                float2 _f2_101 = make_float2(state_values[30], state_values[31]);
                float2 _f2_102 = make_float2(b_values[0], b_values[1]);
                float2 _f2_103 = make_float2(c_values[0], c_values[1]);
                float2 _state_pair_4_3_31 = fma_f32x2(_f2_101, _f2_52, mul_f32x2(_f2_102, _f2_53));
                float2 _projection_pair_4_3_32 = fma_f32x2(_state_pair_4_3_31, _f2_103, _projection_pair_4_2_28);
                state_values[30] = _state_pair_4_3_31.x;
                state_values[31] = _state_pair_4_3_31.y;
                float _shfl_down_2 = __shfl_down_sync(0xFFFFFFFF, _projection_pair_0_3_30.x + _projection_pair_0_3_30.y + (_projection_pair_4_3_32.x + _projection_pair_4_3_32.y), 2, 4);
                float _shfl_down_3 = __shfl_down_sync(0xFFFFFFFF, _projection_pair_0_3_30.x + _projection_pair_0_3_30.y + (_projection_pair_4_3_32.x + _projection_pair_4_3_32.y) + _shfl_down_2, 1, 4);
                if (member == 0) {
                    int output_index_1 = ((token_base + pair_base + 1) * nheads + head) * dim + dim_index;
                    output[output_index_1] = _projection_pair_0_3_30.x + _projection_pair_0_3_30.y + (_projection_pair_4_3_32.x + _projection_pair_4_3_32.y) + _shfl_down_2 + _shfl_down_3 + d_value * (float)s_x[(pair_base + 1) * 128 + dim_index];
                }
                if (source_slot != pad_slot_id) {
                    int publication_row_offset_i32_1 = (head * dim + dim_index) * dstate;
                    unsigned long long publication_row_offset_1 = (unsigned long long)publication_row_offset_i32_1;
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(state_values[0 + 0], state_values[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(state_values[0 + 2], state_values[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(state_values[0 + 4], state_values[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(state_values[0 + 6], state_values[0 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[(unsigned long long)source_slot * state_stride_slot + publication_row_offset_1 + (unsigned long long)(member * 8) + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(state_values[8 + 0], state_values[8 + 1]);
                        _pk[1] = __floats2bfloat162_rn(state_values[8 + 2], state_values[8 + 3]);
                        _pk[2] = __floats2bfloat162_rn(state_values[8 + 4], state_values[8 + 5]);
                        _pk[3] = __floats2bfloat162_rn(state_values[8 + 6], state_values[8 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[(unsigned long long)source_slot * state_stride_slot + publication_row_offset_1 + (unsigned long long)(32 + member * 8) + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(state_values[16 + 0], state_values[16 + 1]);
                        _pk[1] = __floats2bfloat162_rn(state_values[16 + 2], state_values[16 + 3]);
                        _pk[2] = __floats2bfloat162_rn(state_values[16 + 4], state_values[16 + 5]);
                        _pk[3] = __floats2bfloat162_rn(state_values[16 + 6], state_values[16 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[(unsigned long long)source_slot * state_stride_slot + publication_row_offset_1 + (unsigned long long)(64 + member * 8) + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(state_values[24 + 0], state_values[24 + 1]);
                        _pk[1] = __floats2bfloat162_rn(state_values[24 + 2], state_values[24 + 3]);
                        _pk[2] = __floats2bfloat162_rn(state_values[24 + 4], state_values[24 + 5]);
                        _pk[3] = __floats2bfloat162_rn(state_values[24 + 6], state_values[24 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[(unsigned long long)source_slot * state_stride_slot + publication_row_offset_1 + (unsigned long long)(96 + member * 8) + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
        }
    }
}

} // extern "C"

