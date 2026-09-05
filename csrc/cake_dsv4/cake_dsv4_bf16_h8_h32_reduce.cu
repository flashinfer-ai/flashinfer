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
#define SMEM_SPLIT_WEIGHTS_OFF 0
#define SMEM_SPLIT_WEIGHTS_STAGE_BYTES 12
#define SMEM_SPLIT_WEIGHTS_STRIDE 12
#define SMEM_TOTAL 128
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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_dsv4_bf16_h8_h32_reduce(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, int num_heads, int num_splits)
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
    float* split_weights = reinterpret_cast<float*>(smem_raw + 0);
    const int split_weights_addr = smem + 0;

    // === Task calls (dependency order) ===
    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int stat_base = (query_idx * num_heads + head_idx) * num_splits;
    if (warp == 0) {
        float local_lse = -CAKE_INF;
        if (lane < num_splits) {
            local_lse = partial_lse[stat_base + lane];
        }
        float _warp_reduce_0 = local_lse;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
        float global_max = _warp_reduce_0;
        float local_weight = 0.0f;
        if (lane < num_splits) {
            float _exp2_0 = approx_exp2(local_lse - global_max);
            local_weight = ((local_lse == -CAKE_INF) ? 0.0f : _exp2_0);
        }
        float _warp_reduce_1 = local_weight;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        float global_sum = _warp_reduce_1;
        if (lane < num_splits) {
            float _rcp_0 = approx_rcp(global_sum);
            split_weights[lane] = ((global_sum > 0.0f) ? local_weight * _rcp_0 : 0.0f);
        }
    }
    __syncthreads();
    int partial_base = stat_base * 512;
    int output_base = (query_idx * num_heads + head_idx) * 512;
    int d_base = tid * 4;
    float acc[4];
    #pragma unroll
    for (int elem = 0; elem < 4; elem++) {
        acc[elem] = 0.0f;
    }
    #pragma unroll 3
    for (int split = 0; split < num_splits; split++) {
        float _vec_load_0[4];
        {
            uint2 _vld_0 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + split * 512 + d_base) + 0);
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_0[0 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _pair * 2])[1])
                    : "r"(_vpairs_0[_pair]));
            }
        }
        float weight = split_weights[split];
        #pragma unroll
        for (int elem_1 = 0; elem_1 < 4; elem_1++) {
            acc[elem_1] = acc[elem_1] + weight * _vec_load_0[elem_1];
        }
    }
    {
        uint2 _pk2;
        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
        _pk[0] = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(acc[0 + 2], acc[0 + 3]);
        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O + (output_base + d_base)))[0]) = _pk2;
    }
}

} // extern "C"

