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
#define SMEM_S_B_STAGE_BYTES 2048
#define SMEM_S_B_STRIDE 2048
#define SMEM_S_C_OFF 2048
#define SMEM_S_C_STAGE_BYTES 2048
#define SMEM_S_C_STRIDE 2048
#define SMEM_S_X_OFF 4096
#define SMEM_S_X_STAGE_BYTES 256
#define SMEM_S_X_STRIDE 256
#define SMEM_S_DT_OFF 4352
#define SMEM_S_DT_STAGE_BYTES 32
#define SMEM_S_DT_STRIDE 32
#define SMEM_S_DST_OFF 4384
#define SMEM_S_DST_STAGE_BYTES 8
#define SMEM_S_DST_STRIDE 8
#define SMEM_S_STATE_OFF 4480
#define SMEM_S_STATE_STAGE_BYTES 8192
#define SMEM_S_STATE_STRIDE 8192
#define SMEM_TOTAL 12672
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 8) void
kernel_cake_selective_state_update_dynamic_checkpoint3(float* __restrict__ state, __nv_bfloat16* __restrict__ x, float* __restrict__ dt, float* __restrict__ A, __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C, float* __restrict__ D, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ output, long long* __restrict__ state_batch_indices, long long* __restrict__ dst_state_batch_indices, int batch_size, int token_steps, int previous_tokens, unsigned long long state_stride_slot, long long pad_slot_id)
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
    __nv_bfloat16* s_C = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int s_C_addr = smem + 2048;
    __nv_bfloat16* s_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 4096);
    const int s_x_addr = smem + 4096;
    float* s_dt = reinterpret_cast<float*>(smem_raw + 4352);
    const int s_dt_addr = smem + 4352;
    long long* s_dst = reinterpret_cast<long long*>(smem_raw + 4384);
    const int s_dst_addr = smem + 4384;
    float* s_state = reinterpret_cast<float*>(smem_raw + 4480);
    const int s_state_addr = smem + 4480;

    // === Task calls (dependency order) ===
    int batch = blockIdx.x;
    int head = blockIdx.y;
    int dim_tile = blockIdx.z;
    int dim_base = dim_tile * 16;
    int token_base = batch * token_steps;
    long long source_slot = state_batch_indices[batch];
    if (warp == 0) {
        #pragma unroll
        for (int pack_turn = 0; pack_turn < 4; pack_turn++) {
            int pack = lane + pack_turn * 32;
            int step = pack / 16;
            int col = pack % 16 * 8;
            int source_step = step;
            if (source_step >= token_steps) {
                source_step = 0;
            }
            int src_index = (token_base + source_step) * 128 + col;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(s_B_addr + (unsigned int)((step * 128 + col) * 2)), "l"(B + src_index));
        }
    }
    if (warp == 1) {
        #pragma unroll
        for (int pack_turn_1 = 0; pack_turn_1 < 4; pack_turn_1++) {
            int pack_1 = lane + pack_turn_1 * 32;
            int step_1 = pack_1 / 16;
            int col_1 = pack_1 % 16 * 8;
            int source_step_1 = step_1;
            if (source_step_1 >= token_steps) {
                source_step_1 = 0;
            }
            int src_index_1 = (token_base + source_step_1) * 128 + col_1;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(s_C_addr + (unsigned int)((step_1 * 128 + col_1) * 2)), "l"(C + src_index_1));
        }
    }
    #pragma unroll
    for (int token_turn = 0; token_turn < 2; token_turn++) {
        int step_2 = warp + token_turn * 4;
        int source_step_2 = step_2;
        if (source_step_2 >= token_steps) {
            source_step_2 = 0;
        }
        if (lane < 2) {
            int col_2 = lane * 8;
            int src_index_2 = ((token_base + source_step_2) * 16 + head) * 64 + dim_base + col_2;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(s_x_addr + (unsigned int)((step_2 * 16 + col_2) * 2)), "l"(x + src_index_2));
        }
    }
    #pragma unroll
    for (int pack_turn_2 = 0; pack_turn_2 < 4; pack_turn_2++) {
        int flat_pack = tid + pack_turn_2 * 128;
        int row = flat_pack / 32;
        int col_3 = flat_pack % 32 * 4;
        int state_row_i32 = (head * 64 + dim_base + row) * 128 + col_3;
        if (source_slot != pad_slot_id) {
            unsigned long long state_index = (unsigned long long)source_slot * state_stride_slot + (unsigned long long)state_row_i32;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(s_state_addr + (unsigned int)((row * 128 + col_3) * 4)), "l"(state + state_index));
        } else {
            #pragma unroll
            for (int element = 0; element < 4; element++) {
                s_state[row * 128 + col_3 + element] = 0.0f;
            }
        }
    }
    if (tid < token_steps) {
        int step_3 = tid;
        float dt_value = dt[(token_base + step_3) * 16 + head];
        dt_value += dt_bias[head];
        float _exp_0 = expf(dt_value);
        float _log1p_0 = log1pf(_exp_0);
        s_dt[step_3] = _log1p_0;
    }
    int checkpoint_step = 3;
    if (tid == 0) {
        s_dst[0] = dst_state_batch_indices[batch * token_steps + checkpoint_step];
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    int member = lane % 8;
    int subgroup = lane / 8;
    int local_row = warp * 4 + subgroup;
    int dim_index = dim_base + local_row;
    float state_values[16];
    #pragma unroll
    for (int tile = 0; tile < 4; tile++) {
        int state_col = tile * 32 + member * 4;
        #pragma unroll
        for (int element_1 = 0; element_1 < 4; element_1++) {
            state_values[tile * 4 + element_1] = s_state[local_row * 128 + state_col + element_1];
        }
    }
    float a_value = A[head];
    float d_value = D[head];
    #pragma unroll
    for (int step_4 = 0; step_4 < 8; step_4++) {
        if (step_4 < token_steps) {
            float dt_value_1 = s_dt[step_4];
            float _exp_1 = expf(a_value * dt_value_1);
            float decay = _exp_1;
            float x_value = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_x) + ((step_4 * 16 + local_row) * 2))[0];
            float dtx = dt_value_1 * x_value;
            float partial = 0.0f;
            #pragma unroll
            for (int tile_1 = 0; tile_1 < 4; tile_1++) {
                int state_col_1 = tile_1 * 32 + member * 4;
                #pragma unroll
                for (int element_2 = 0; element_2 < 4; element_2++) {
                    int col_4 = state_col_1 + element_2;
                    float b_value = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_B) + ((step_4 * 128 + col_4) * 2))[0];
                    float c_value = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_C) + ((step_4 * 128 + col_4) * 2))[0];
                    state_values[tile_1 * 4 + element_2] = state_values[tile_1 * 4 + element_2] * decay + b_value * dtx;
                    partial += state_values[tile_1 * 4 + element_2] * c_value;
                }
            }
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, partial, 4);
            partial += _shfl_xor_0;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, partial, 2);
            partial += _shfl_xor_1;
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, partial, 1);
            partial += _shfl_xor_2;
            if (member == 0) {
                int output_index = ((token_base + step_4) * 16 + head) * 64 + dim_index;
                output[output_index] = partial + d_value * x_value;
            }
            long long destination_slot = pad_slot_id;
            if (step_4 == checkpoint_step) {
                destination_slot = s_dst[0];
            }
            if (source_slot != pad_slot_id) {
                if (destination_slot != pad_slot_id) {
                    #pragma unroll
                    for (int tile_2 = 0; tile_2 < 4; tile_2++) {
                        int state_col_2 = tile_2 * 32 + member * 4;
                        int destination_row_i32 = (head * 64 + dim_index) * 128 + state_col_2;
                        unsigned long long destination_index = (unsigned long long)destination_slot * state_stride_slot + (unsigned long long)destination_row_i32;
                        {
                            float4 _v4 = make_float4(state_values[tile_2 * 4 + 0], state_values[tile_2 * 4 + 1], state_values[tile_2 * 4 + 2], state_values[tile_2 * 4 + 3]);
                            *reinterpret_cast<float4*>(state + destination_index) = _v4;
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

