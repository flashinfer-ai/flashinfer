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
#define SMEM_SHARED_COEFFICIENTS_OFF 0
#define SMEM_SHARED_COEFFICIENTS_STAGE_BYTES 0
#define SMEM_SHARED_COEFFICIENTS_STRIDE 0
#define SMEM_TOTAL 0
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(128, 8) void
kernel_cake_selective_state_update_stp_fp32_identity(float* __restrict__ state, __nv_bfloat16* __restrict__ x, unsigned long long dt_addr, unsigned long long a_addr, __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C, unsigned long long d_addr, __nv_bfloat16* __restrict__ z, unsigned long long dt_bias_addr, __nv_bfloat16* __restrict__ output, long long* __restrict__ state_batch_indices, long long* __restrict__ dst_state_batch_indices, int nheads, int ngroups, int dim_tiles, unsigned long long state_stride_slot, long long dt_batch_stride, long long dt_head_stride, long long a_head_stride, long long d_head_stride, long long dt_bias_head_stride, int dt_softplus, int has_z, int disable_state_update, long long pad_slot_id)
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
    unsigned int* shared_coefficients = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int shared_coefficients_addr = smem + 0;

    // === Task calls (dependency order) ===
    int dim_tile = 0;
    int batch_head = bid;
    int head = batch_head % nheads;
    int batch = batch_head / nheads;
    int heads_per_group = nheads / ngroups;
    int group = head / heads_per_group;
    int rows_per_tile = 128;
    int dim_base = 0;
    int dim_end = 128;
    long long source_slot = state_batch_indices[batch];
    long long destination_slot = ((1) ? source_slot : dst_state_batch_indices[batch]);
    int row_subgroup = lane / 16;
    int row_member = lane % 16;
    int state_col = row_member * 8;
    int bc_base = (batch * ngroups + group) * 128;
    unsigned int b_carriers[4];
    unsigned int c_carriers[4];
    float b_direct_values[8];
    float c_direct_values[8];
    {
        {
            {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(B + bc_base + state_col);
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
                                : "=f"((&b_direct_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&b_direct_values[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(C + bc_base + state_col);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&c_direct_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&c_direct_values[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_1[_pair]));
                        }
                    }
                }
            }
        }
    }
    float dt_lane = 0.0f;
    float decay_lane = 0.0f;
    float d_lane = 0.0f;
    if (lane == 0) {
        float dt_value = reinterpret_cast<float*>(dt_addr)[(long long)batch * dt_batch_stride + (long long)head * dt_head_stride];
        dt_value += reinterpret_cast<float*>(dt_bias_addr)[(long long)head * dt_bias_head_stride];
        {
        }
        float a_value = reinterpret_cast<float*>(a_addr)[(long long)head * a_head_stride];
        dt_lane = dt_value;
        {
            float _exp_2 = expf(a_value * dt_value);
            decay_lane = _exp_2;
        }
        d_lane = reinterpret_cast<float*>(d_addr)[(long long)head * d_head_stride];
    }
    float _shfl_0 = __shfl_sync(0xFFFFFFFF, dt_lane, 0);
    float dt_value_1 = _shfl_0;
    float _shfl_1 = __shfl_sync(0xFFFFFFFF, decay_lane, 0);
    float decay = _shfl_1;
    float _shfl_2 = __shfl_sync(0xFFFFFFFF, d_lane, 0);
    float d_value = _shfl_2;
    {
    }
    for (int dim_group_base = dim_base + warp * 2; dim_group_base < dim_end; dim_group_base += 8) {
        int dim_index = dim_group_base + row_subgroup;
        int x_index = 0;
        {
            x_index = (batch * nheads + head) * 128 + dim_index;
        }
        float x_lane = 0.0f;
        float z_lane = 0.0f;
        if (row_member == 0) {
            x_lane = (float)x[x_index];
            {
            }
        }
        float _shfl_3 = __shfl_sync(0xFFFFFFFF, x_lane, row_subgroup * 16);
        float x_value = _shfl_3;
        float partial = 0.0f;
        {
            {
                float direct_state_values[8];
                #pragma unroll
                for (int element = 0; element < 8; element++) {
                    direct_state_values[element] = 0.0f;
                }
                int row_offset_i32 = (head * 128 + dim_index) * 128 + state_col;
                unsigned long long row_offset = (unsigned long long)row_offset_i32;
                unsigned long long source_index = (unsigned long long)source_slot * state_stride_slot + row_offset;
                if (source_slot != pad_slot_id) {
                    {
                        unsigned _ldv8_2_0;
                        unsigned _ldv8_2_1;
                        unsigned _ldv8_2_2;
                        unsigned _ldv8_2_3;
                        unsigned _ldv8_2_4;
                        unsigned _ldv8_2_5;
                        unsigned _ldv8_2_6;
                        unsigned _ldv8_2_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(state + (source_index))) : "memory");
                        direct_state_values[0 + 0] = __uint_as_float(_ldv8_2_0);
                        direct_state_values[0 + 1] = __uint_as_float(_ldv8_2_1);
                        direct_state_values[0 + 2] = __uint_as_float(_ldv8_2_2);
                        direct_state_values[0 + 3] = __uint_as_float(_ldv8_2_3);
                        direct_state_values[0 + 4] = __uint_as_float(_ldv8_2_4);
                        direct_state_values[0 + 5] = __uint_as_float(_ldv8_2_5);
                        direct_state_values[0 + 6] = __uint_as_float(_ldv8_2_6);
                        direct_state_values[0 + 7] = __uint_as_float(_ldv8_2_7);
                    }
                }
                #pragma unroll
                for (int element_1 = 0; element_1 < 8; element_1++) {
                    float d_b = b_direct_values[element_1] * dt_value_1;
                    float decayed_state = direct_state_values[element_1] * decay;
                    float new_state = decayed_state + d_b * x_value;
                    direct_state_values[element_1] = new_state;
                    partial += new_state * c_direct_values[element_1];
                }
                {
                    if (source_slot != pad_slot_id) {
                        unsigned long long destination_index = (unsigned long long)source_slot * state_stride_slot + row_offset;
                        {
                            unsigned _stv8_3_0 = __float_as_uint(direct_state_values[0 + 0]);
                            unsigned _stv8_3_1 = __float_as_uint(direct_state_values[0 + 1]);
                            unsigned _stv8_3_2 = __float_as_uint(direct_state_values[0 + 2]);
                            unsigned _stv8_3_3 = __float_as_uint(direct_state_values[0 + 3]);
                            unsigned _stv8_3_4 = __float_as_uint(direct_state_values[0 + 4]);
                            unsigned _stv8_3_5 = __float_as_uint(direct_state_values[0 + 5]);
                            unsigned _stv8_3_6 = __float_as_uint(direct_state_values[0 + 6]);
                            unsigned _stv8_3_7 = __float_as_uint(direct_state_values[0 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(state + (destination_index))), "r"(_stv8_3_0), "r"(_stv8_3_1), "r"(_stv8_3_2), "r"(_stv8_3_3), "r"(_stv8_3_4), "r"(_stv8_3_5), "r"(_stv8_3_6), "r"(_stv8_3_7) : "memory");
                        }
                    }
                }
            }
        }
        float row_sum = partial;
        {
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 8);
            row_sum += _shfl_xor_1;
        }
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 4);
        row_sum += _shfl_xor_2;
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 2);
        row_sum += _shfl_xor_3;
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 1);
        row_sum += _shfl_xor_4;
        if (row_member == 0) {
            float result = row_sum + d_value * x_lane;
            {
            }
            output[x_index] = result;
        }
    }
}

} // extern "C"

