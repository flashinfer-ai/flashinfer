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
#define SMEM_S_B_STAGE_BYTES 1536
#define SMEM_S_B_STRIDE 1536
#define SMEM_S_C_OFF 1536
#define SMEM_S_C_STAGE_BYTES 1536
#define SMEM_S_C_STRIDE 1536
#define SMEM_S_X_OFF 3072
#define SMEM_S_X_STAGE_BYTES 192
#define SMEM_S_X_STRIDE 192
#define SMEM_S_DT_OFF 3264
#define SMEM_S_DT_STAGE_BYTES 24
#define SMEM_S_DT_STRIDE 24
#define SMEM_S_DECAY_OFF 3288
#define SMEM_S_DECAY_STAGE_BYTES 24
#define SMEM_S_DECAY_STRIDE 24
#define SMEM_S_STATE_OFF 3328
#define SMEM_S_STATE_STAGE_BYTES 4096
#define SMEM_S_STATE_STRIDE 4096
#define SMEM_TOTAL 7424
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

__global__ __launch_bounds__(128, 7) void
kernel_cake_selective_state_update_mtp_cache_bf16_c4_t6(__nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ x, float* __restrict__ dt, float* __restrict__ A, __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C, float* __restrict__ D, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ output, long long* __restrict__ state_batch_indices, __nv_bfloat16* __restrict__ intermediate_state, long long* __restrict__ intermediate_state_indices, int nheads, int ngroups, unsigned long long state_stride_slot, unsigned long long intermediate_stride_slot, long long pad_slot_id)
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
    __nv_bfloat16* s_C = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1536);
    const int s_C_addr = smem + 1536;
    __nv_bfloat16* s_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int s_x_addr = smem + 3072;
    float* s_dt = reinterpret_cast<float*>(smem_raw + 3264);
    const int s_dt_addr = smem + 3264;
    float* s_decay = reinterpret_cast<float*>(smem_raw + 3288);
    const int s_decay_addr = smem + 3288;
    __nv_bfloat16* s_state = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3328);
    const int s_state_addr = smem + 3328;

    // === Task calls (dependency order) ===
    int batch = blockIdx.x;
    int head = blockIdx.y;
    int cta_z = blockIdx.z;
    int dim_base = cta_z * 16;
    long long source_slot = (long long)state_batch_indices[batch];
    int logical_nheads = nheads;
    int heads_per_group = nheads / ngroups;
    int group = head / heads_per_group;
    long long b_batch_base = (long long)(batch * 6 * ngroups * 128);
    long long b_step_stride = (long long)(ngroups * 128);
    long long b_group_stride = 128;
    long long c_batch_base = b_batch_base;
    long long c_step_stride = b_step_stride;
    long long c_group_stride = b_group_stride;
    long long x_batch_base = (long long)(batch * 6 * nheads * 64);
    long long x_step_stride = (long long)(nheads * 64);
    long long x_head_stride = 64;
    long long dt_batch_base = (long long)(batch * 6 * nheads);
    long long dt_step_stride = nheads;
    long long dt_head_stride = 1;
    unsigned long long source_state_stride = state_stride_slot;
    unsigned long long intermediate_slot_stride = intermediate_stride_slot;
    #pragma unroll
    for (int pack_turn = 0; pack_turn < 3; pack_turn++) {
        int pack = lane + pack_turn * 32;
        int step = pack / 16;
        int col = pack % 16 * 8;
        long long b_source_index = b_batch_base + (long long)step * b_step_stride + (long long)group * b_group_stride + (long long)col;
        long long c_source_index = c_batch_base + (long long)step * c_step_stride + (long long)group * c_group_stride + (long long)col;
        if (warp == 0) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(s_B_addr + (unsigned int)((step * 128 + col) * 2)), "l"(B + b_source_index));
        }
        if (warp == 1) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(s_C_addr + (unsigned int)((step * 128 + col) * 2)), "l"(C + c_source_index));
        }
    }
    for (int step_1 = warp; step_1 < 6; step_1 += 4) {
        for (int col_1 = lane * 8; col_1 < 16; col_1 += 256) {
            long long source_index = x_batch_base + (long long)step_1 * x_step_stride + (long long)head * x_head_stride + (long long)(dim_base + col_1);
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(s_x_addr + (unsigned int)((step_1 * 16 + col_1) * 2)), "l"(x + source_index));
        }
    }
    #pragma unroll
    for (int pack_turn_1 = 0; pack_turn_1 < 2; pack_turn_1++) {
        int flat_pack = tid + pack_turn_1 * 128;
        int row = flat_pack / 16;
        int pack_in_row = flat_pack % 16;
        int col_2 = pack_in_row * 8;
        int state_row = (head * 64 + dim_base + row) * 128 + col_2;
        if (source_slot != pad_slot_id) {
            unsigned long long state_index = (unsigned long long)source_slot * source_state_stride + (unsigned long long)state_row;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(s_state_addr + (unsigned int)((row * 128 + col_2) * 2)), "l"(state + state_index));
        } else {
            #pragma unroll
            for (int element = 0; element < 8; element++) {
                s_state[row * 128 + col_2 + element] = 0.0f;
            }
        }
    }
    if (tid < 6) {
        int step_2 = tid;
        long long dt_source_index = dt_batch_base + (long long)step_2 * dt_step_stride + (long long)head * dt_head_stride;
        float dt_value = dt[dt_source_index];
        dt_value += dt_bias[head];
        if (dt_value <= 20.0f) {
            float _expf_0 = __expf(dt_value);
            float _log2_0;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(1.0f + _expf_0));
            dt_value = _log2_0 * 0.6931471805599453f;
        }
        s_dt[step_2] = dt_value;
        {
            float _expf_1 = __expf(A[head] * dt_value);
            s_decay[step_2] = _expf_1;
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    int member = lane % 8;
    int subgroup = lane / 8;
    int local_row = warp * 4 + subgroup;
    float d_value = D[head];
    long long cache_slot = (long long)intermediate_state_indices[batch];
    #pragma unroll
    for (int dim_pass = 0; dim_pass < 1; dim_pass++) {
        int dim_index = dim_base + dim_pass * 16 + local_row;
        float state_values[16];
        unsigned long long intermediate_step_stride = 0;
        unsigned long long intermediate_row_base = 0;
        {
            intermediate_step_stride = (unsigned long long)(logical_nheads * 64 * 128);
            intermediate_row_base = (unsigned long long)cache_slot * intermediate_slot_stride + (unsigned long long)((head * 64 + dim_index) * 128);
        }
        unsigned int state_carrier[1];
        unsigned int b_carrier[4];
        unsigned int c_carrier[4];
        float state_pair_values[2];
        float b_values[8];
        float c_values[8];
        #pragma unroll
        for (int tile = 0; tile < 2; tile++) {
            int state_col = tile * 8 * 8 + member * 8;
            {
                #pragma unroll
                for (int pair = 0; pair < 4; pair++) {
                    int state_element = local_row * 128 + state_col + pair * 2;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&state_carrier[0])) : "r"(s_state_addr + (unsigned int)(state_element * 2)));
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&state_pair_values[_pair * 2])[0]), "=f"((&state_pair_values[_pair * 2])[1])
                            : "r"(state_carrier[_pair]));
                    }
                    state_values[tile * 8 + pair * 2] = state_pair_values[0];
                    state_values[tile * 8 + pair * 2 + 1] = state_pair_values[1];
                }
            }
        }
        #pragma unroll
        for (int step_3 = 0; step_3 < 6; step_3++) {
            float dt_value_1 = s_dt[step_3];
            float decay = 0.0f;
            {
                decay = s_decay[step_3];
            }
            float x_value = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_x) + ((step_3 * 16 + dim_pass * 16 + local_row) * 2))[0];
            float2 _f2_0 = make_float2(decay, decay);
            float2 decay_pair = _f2_0;
            float dtx_value = dt_value_1 * x_value;
            float2 _f2_1 = make_float2(dtx_value, dtx_value);
            float2 dtx_pair = _f2_1;
            float2 _f2_2 = make_float2(0.0f, 0.0f);
            float2 partial_pair = _f2_2;
            #pragma unroll
            for (int tile_1 = 0; tile_1 < 2; tile_1++) {
                int state_col_1 = tile_1 * 8 * 8 + member * 8;
                int operand_index = step_3 * 128 + state_col_1;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&b_carrier[0])), "=r"(*reinterpret_cast<uint32_t*>(&b_carrier[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_carrier[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_carrier[(0) + 3]))
                    : "r"(s_B_addr + (unsigned int)(operand_index * 2)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c_carrier[0])), "=r"(*reinterpret_cast<uint32_t*>(&c_carrier[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c_carrier[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c_carrier[(0) + 3]))
                    : "r"(s_C_addr + (unsigned int)(operand_index * 2)));
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&b_values[_pair * 2])[0]), "=f"((&b_values[_pair * 2])[1])
                        : "r"(b_carrier[_pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&c_values[_pair * 2])[0]), "=f"((&c_values[_pair * 2])[1])
                        : "r"(c_carrier[_pair]));
                }
                #pragma unroll
                for (int pair_1 = 0; pair_1 < 4; pair_1++) {
                    float2 _f2_3 = make_float2(state_values[tile_1 * 8 + pair_1 * 2], state_values[tile_1 * 8 + pair_1 * 2 + 1]);
                    float2 state_pair = _f2_3;
                    float2 _f2_4 = make_float2(b_values[pair_1 * 2], b_values[pair_1 * 2 + 1]);
                    float2 b_pair = _f2_4;
                    float2 _f2_5 = make_float2(c_values[pair_1 * 2], c_values[pair_1 * 2 + 1]);
                    float2 c_pair = _f2_5;
                    float2 dbx_pair = mul_f32x2(b_pair, dtx_pair);
                    state_pair = fma_f32x2(state_pair, decay_pair, dbx_pair);
                    partial_pair = fma_f32x2(state_pair, c_pair, partial_pair);
                    state_values[tile_1 * 8 + pair_1 * 2] = state_pair.x;
                    state_values[tile_1 * 8 + pair_1 * 2 + 1] = state_pair.y;
                }
            }
            float partial = partial_pair.x + partial_pair.y;
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, partial, 4);
            partial += _shfl_xor_0;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, partial, 2);
            partial += _shfl_xor_1;
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, partial, 1);
            partial += _shfl_xor_2;
            if (member == 0) {
                int output_index = ((batch * 6 + step_3) * logical_nheads + head) * 64 + dim_index;
                output[output_index] = partial + d_value * x_value;
            }
            {
                if (source_slot != pad_slot_id) {
                    #pragma unroll
                    for (int tile_2 = 0; tile_2 < 2; tile_2++) {
                        int state_col_2 = tile_2 * 8 * 8 + member * 8;
                        {
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(state_values[tile_2 * 8 + 0], state_values[tile_2 * 8 + 1]);
                                _pk[1] = __floats2bfloat162_rn(state_values[tile_2 * 8 + 2], state_values[tile_2 * 8 + 3]);
                                _pk[2] = __floats2bfloat162_rn(state_values[tile_2 * 8 + 4], state_values[tile_2 * 8 + 5]);
                                _pk[3] = __floats2bfloat162_rn(state_values[tile_2 * 8 + 6], state_values[tile_2 * 8 + 7]);
                                uint4 _st_v4_0 = *reinterpret_cast<uint4*>(&_pk[0]);
                                asm volatile(
                                    "st.global.L1::no_allocate.v4.b32 [%0], {%1, %2, %3, %4};"
                                    :: "l"(&((__nv_bfloat16*)(intermediate_state))[intermediate_row_base + (unsigned long long)state_col_2 + 0]), "r"(_st_v4_0.x), "r"(_st_v4_0.y), "r"(_st_v4_0.z), "r"(_st_v4_0.w) : "memory");
                            }
                        }
                    }
                }
                intermediate_row_base += intermediate_step_stride;
            }
        }
        if (dim_pass + 1 < 1) {
            __syncthreads();
            #pragma unroll
            for (int pack_turn_2 = 0; pack_turn_2 < 2; pack_turn_2++) {
                int flat_pack_1 = tid + pack_turn_2 * 128;
                int row_1 = flat_pack_1 / 16;
                int pack_in_row_1 = flat_pack_1 % 16;
                int col_3 = pack_in_row_1 * 8;
                int next_dim_base = dim_base + (dim_pass + 1) * 16;
                int state_row_1 = (head * 64 + next_dim_base + row_1) * 128 + col_3;
                if (source_slot != pad_slot_id) {
                    unsigned long long state_index_1 = (unsigned long long)source_slot * source_state_stride + (unsigned long long)state_row_1;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_state_addr + (unsigned int)((row_1 * 128 + col_3) * 2)), "l"(state + state_index_1));
                } else {
                    #pragma unroll
                    for (int element_1 = 0; element_1 < 8; element_1++) {
                        s_state[row_1 * 128 + col_3 + element_1] = 0.0f;
                    }
                }
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
        }
    }
}

} // extern "C"

