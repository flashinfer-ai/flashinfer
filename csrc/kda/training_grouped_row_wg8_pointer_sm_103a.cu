typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) LoomTensorMapPack { LoomTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define D 128

extern "C" {

__global__ __launch_bounds__(32) void
kernel_flashkda_backward_preprocess_bf16_norm(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ beta, float* __restrict__ A_log, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ q_norm, __nv_bfloat16* __restrict__ k_norm, __nv_bfloat16* __restrict__ decay, float* __restrict__ beta_active, int total_tokens, int num_heads, float lower_bound)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int token = blockIdx.x;
    int head = blockIdx.y;
    int elem = lane * 4;
    long long base = ((long long)token * (long long)num_heads + (long long)head) * (long long)D;
    float q_frag[4];
    float k_frag[4];
    float g_frag[4];
    {
        uint2 _vld_0;
        _vld_0 = *reinterpret_cast<const uint2*>(q + base + (long long)elem);
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&q_frag[0 + _pair * 2])[0]), "=f"((&q_frag[0 + _pair * 2])[1])
                : "r"(_vpairs_0[_pair]));
        }
    }
    {
        uint2 _vld_1;
        _vld_1 = *reinterpret_cast<const uint2*>(k + base + (long long)elem);
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&k_frag[0 + _pair * 2])[0]), "=f"((&k_frag[0 + _pair * 2])[1])
                : "r"(_vpairs_1[_pair]));
        }
    }
    {
        uint2 _vld_2;
        _vld_2 = *reinterpret_cast<const uint2*>(g + base + (long long)elem);
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&g_frag[0 + _pair * 2])[0]), "=f"((&g_frag[0 + _pair * 2])[1])
                : "r"(_vpairs_2[_pair]));
        }
    }
    float q_sq = 0.0f;
    float k_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float _fma_0 = __fmaf_rn(q_frag[i], q_frag[i], q_sq);
        q_sq = _fma_0;
        float _fma_1 = __fmaf_rn(k_frag[i], k_frag[i], k_sq);
        k_sq = _fma_1;
    }
    float _warp_reduce_0 = q_sq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    q_sq = _warp_reduce_0;
    float _warp_reduce_1 = k_sq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    k_sq = _warp_reduce_1;
    float _rsqrt_0 = rsqrtf(q_sq + 1e-06f);
    float q_inv = _rsqrt_0;
    float _rsqrt_1 = rsqrtf(k_sq + 1e-06f);
    float k_inv = _rsqrt_1;
    float gate_a_lane = 0.0f;
    if (lane == 0) {
        float _expf_0 = __expf(A_log[head]);
        gate_a_lane = _expf_0;
    }
    float _shfl_0 = __shfl_sync(0xFFFFFFFF, gate_a_lane, 0);
    float gate_a = _shfl_0;
    float q_packed[4];
    float k_packed[4];
    #pragma unroll
    for (int i2 = 0; i2 < 4; i2++) {
        int dim = elem + i2;
        float biased = g_frag[i2] + dt_bias[head * D + dim];
        float _expf_1 = __expf((-gate_a) * biased);
        float _rcp_0 = approx_rcp(1.0f + _expf_1);
        float gate_sigmoid = _rcp_0;
        q_packed[i2] = q_frag[i2] * q_inv;
        k_packed[i2] = k_frag[i2] * k_inv;
        float _expf_2 = __expf(lower_bound * gate_sigmoid);
        __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_expf_2);
        decay[base + (long long)dim] = _cvt_bf16_0;
    }
    {
        uint2 _pk2;
        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
        _pk[0] = __floats2bfloat162_rn(q_packed[0 + 0], q_packed[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(q_packed[0 + 2], q_packed[0 + 3]);
        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(q_norm + (base + (long long)elem)))[0]) = _pk2;
    }
    {
        uint2 _pk2;
        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
        _pk[0] = __floats2bfloat162_rn(k_packed[0 + 0], k_packed[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(k_packed[0 + 2], k_packed[0 + 3]);
        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(k_norm + (base + (long long)elem)))[0]) = _pk2;
    }
    if (lane == 0) {
        long long beta_index = (long long)token * (long long)num_heads + (long long)head;
        float beta_raw = (float)beta[beta_index];
        float _expf_3 = __expf(-beta_raw);
        float _rcp_1 = approx_rcp(1.0f + _expf_3);
        beta_active[beta_index] = _rcp_1;
    }
}

} // extern "C"

#undef D
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128
#define D 128
#define V 128

extern "C" {

__global__ __launch_bounds__(128) void
kernel_flashkda_backward_checkpoint_wg4(__nv_bfloat16* __restrict__ k_norm, __nv_bfloat16* __restrict__ decay, float* __restrict__ beta_active, __nv_bfloat16* __restrict__ v, float* __restrict__ initial_state, long long* __restrict__ cu_seqlens, __nv_bfloat16* __restrict__ checkpoint, int num_sequences, int num_heads)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int work = blockIdx.x;
    int groups_per_head = V / 4;
    int groups_per_sequence = num_heads * groups_per_head;
    int sequence = work / groups_per_sequence;
    int remainder = work - sequence * groups_per_sequence;
    int head = remainder / groups_per_head;
    int row_group = remainder - head * groups_per_head;
    int value_row = row_group * 4 + warp;
    int elem = lane * 4;
    long long bos = cu_seqlens[sequence];
    long long eos = cu_seqlens[sequence + 1];
    long long state_base = (((long long)sequence * (long long)num_heads + (long long)head) * (long long)V + (long long)value_row) * (long long)D;
    float state[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(initial_state + state_base + (long long)elem);
        state[0 + 0] = _v4.x;
        state[0 + 1] = _v4.y;
        state[0 + 2] = _v4.z;
        state[0 + 3] = _v4.w;
    }
    #pragma unroll 1
    for (long long token = bos; token < eos; token++) {
        long long token_base = (token * (long long)num_heads + (long long)head) * (long long)D;
        float k_frag[4];
        float d_frag[4];
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(k_norm + token_base + (long long)elem);
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&k_frag[0 + _pair * 2])[0]), "=f"((&k_frag[0 + _pair * 2])[1])
                    : "r"(_vpairs_1[_pair]));
            }
        }
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(decay + token_base + (long long)elem);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&d_frag[0 + _pair * 2])[0]), "=f"((&d_frag[0 + _pair * 2])[1])
                    : "r"(_vpairs_2[_pair]));
            }
        }
        float pred = 0.0f;
        float decayed[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            decayed[i] = state[i] * d_frag[i];
            float _fma_0 = __fmaf_rn(k_frag[i], decayed[i], pred);
            pred = _fma_0;
        }
        float _warp_reduce_0 = pred;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        pred = _warp_reduce_0;
        long long beta_index = token * (long long)num_heads + (long long)head;
        long long value_index = token_base + (long long)value_row;
        float residual = beta_active[beta_index] * ((float)v[value_index] - pred);
        long long checkpoint_base = ((token * (long long)num_heads + (long long)head) * (long long)V + (long long)value_row) * (long long)D;
        #pragma unroll
        for (int i2 = 0; i2 < 4; i2++) {
            float _fma_1 = __fmaf_rn(residual, k_frag[i2], decayed[i2]);
            state[i2] = _fma_1;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(state[0 + 0], state[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(state[0 + 2], state[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(checkpoint + (checkpoint_base + (long long)elem)))[0]) = _pk2;
        }
    }
}

} // extern "C"

#undef D
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef V

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_DQ_OFF 0
#define SMEM_SMEM_DQ_STAGE_BYTES 4096
#define SMEM_SMEM_DQ_STRIDE 4096
#define SMEM_SMEM_DK_OFF 4096
#define SMEM_SMEM_DK_STAGE_BYTES 4096
#define SMEM_SMEM_DK_STRIDE 4096
#define SMEM_SMEM_DLOG_OFF 8192
#define SMEM_SMEM_DLOG_STAGE_BYTES 4096
#define SMEM_SMEM_DLOG_STRIDE 4096
#define SMEM_SMEM_DBETA_OFF 12288
#define SMEM_SMEM_DBETA_STAGE_BYTES 32
#define SMEM_SMEM_DBETA_STRIDE 32
#define SMEM_TOTAL 12416
#define THREADS 256
#define D 128
#define V 128

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashkda_backward_reverse_wg8(__nv_bfloat16* __restrict__ q_norm, __nv_bfloat16* __restrict__ k_norm, __nv_bfloat16* __restrict__ decay, float* __restrict__ beta_active, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ do_, float* __restrict__ initial_state, float* __restrict__ dfinal_state, long long* __restrict__ cu_seqlens, __nv_bfloat16* __restrict__ checkpoint, float* __restrict__ dq_normalized, float* __restrict__ dk_normalized, float* __restrict__ dlog_decay, float* __restrict__ dbeta_active, __nv_bfloat16* __restrict__ dv, float* __restrict__ dinitial_state, int num_sequences, int num_heads, float scale)
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
    float* smem_dq = reinterpret_cast<float*>(smem_raw + 0);
    const int smem_dq_addr = smem + 0;
    float* smem_dk = reinterpret_cast<float*>(smem_raw + 4096);
    const int smem_dk_addr = smem + 4096;
    float* smem_dlog = reinterpret_cast<float*>(smem_raw + 8192);
    const int smem_dlog_addr = smem + 8192;
    float* smem_dbeta = reinterpret_cast<float*>(smem_raw + 12288);
    const int smem_dbeta_addr = smem + 12288;

    // === Task calls (dependency order) ===
    int work = blockIdx.x;
    int groups_per_head = V / 8;
    int groups_per_sequence = num_heads * groups_per_head;
    int sequence = work / groups_per_sequence;
    int remainder = work - sequence * groups_per_sequence;
    int head = remainder / groups_per_head;
    int row_group = remainder - head * groups_per_head;
    int value_row = row_group * 8 + warp;
    int elem = lane * 4;
    long long bos = cu_seqlens[sequence];
    long long eos = cu_seqlens[sequence + 1];
    long long sequence_length = eos - bos;
    long long state_base = (((long long)sequence * (long long)num_heads + (long long)head) * (long long)V + (long long)value_row) * (long long)D;
    float dstate[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(dfinal_state + state_base + (long long)elem);
        dstate[0 + 0] = _v4.x;
        dstate[0 + 1] = _v4.y;
        dstate[0 + 2] = _v4.z;
        dstate[0 + 3] = _v4.w;
    }
    #pragma unroll 1
    for (long long reverse_step = 0; reverse_step < sequence_length; reverse_step++) {
        long long token = eos - 1 - reverse_step;
        long long token_base = (token * (long long)num_heads + (long long)head) * (long long)D;
        long long checkpoint_base = ((token * (long long)num_heads + (long long)head) * (long long)V + (long long)value_row) * (long long)D;
        long long previous_base = checkpoint_base - (long long)num_heads * (long long)V * (long long)D;
        if (token == bos) {
            previous_base = state_base;
        }
        float q_frag[4];
        float k_frag[4];
        float d_frag[4];
        float state_now[4];
        float state_prev[4];
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(q_norm + token_base + (long long)elem);
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&q_frag[0 + _pair * 2])[0]), "=f"((&q_frag[0 + _pair * 2])[1])
                    : "r"(_vpairs_1[_pair]));
            }
        }
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(k_norm + token_base + (long long)elem);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&k_frag[0 + _pair * 2])[0]), "=f"((&k_frag[0 + _pair * 2])[1])
                    : "r"(_vpairs_2[_pair]));
            }
        }
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(decay + token_base + (long long)elem);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&d_frag[0 + _pair * 2])[0]), "=f"((&d_frag[0 + _pair * 2])[1])
                    : "r"(_vpairs_3[_pair]));
            }
        }
        {
            uint2 _vld_4;
            _vld_4 = *reinterpret_cast<const uint2*>(checkpoint + checkpoint_base + (long long)elem);
            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&state_now[0 + _pair * 2])[0]), "=f"((&state_now[0 + _pair * 2])[1])
                    : "r"(_vpairs_4[_pair]));
            }
        }
        if (token == bos) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(initial_state + previous_base + (long long)elem);
                state_prev[0 + 0] = _v4.x;
                state_prev[0 + 1] = _v4.y;
                state_prev[0 + 2] = _v4.z;
                state_prev[0 + 3] = _v4.w;
            }
        } else {
            {
                uint2 _vld_6;
                _vld_6 = *reinterpret_cast<const uint2*>(checkpoint + previous_base + (long long)elem);
                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&state_prev[0 + _pair * 2])[0]), "=f"((&state_prev[0 + _pair * 2])[1])
                        : "r"(_vpairs_6[_pair]));
                }
            }
        }
        long long value_index = token_base + (long long)value_row;
        float output_adjoint = (float)do_[value_index];
        float pred = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float _fma_0 = __fmaf_rn(scale * output_adjoint, q_frag[i], dstate[i]);
            dstate[i] = _fma_0;
            float _fma_1 = __fmaf_rn(k_frag[i], state_prev[i] * d_frag[i], pred);
            pred = _fma_1;
        }
        float _warp_reduce_0 = pred;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        pred = _warp_reduce_0;
        float dr = 0.0f;
        #pragma unroll
        for (int i2 = 0; i2 < 4; i2++) {
            float _fma_2 = __fmaf_rn(dstate[i2], k_frag[i2], dr);
            dr = _fma_2;
        }
        float _warp_reduce_1 = dr;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        dr = _warp_reduce_1;
        long long beta_index = token * (long long)num_heads + (long long)head;
        float beta_value = beta_active[beta_index];
        float value_raw = (float)v[value_index];
        float residual = beta_value * (value_raw - pred);
        float dpred = (-dr) * beta_value;
        #pragma unroll
        for (int i3 = 0; i3 < 4; i3++) {
            int dim = elem + i3;
            float decayed_state = state_prev[i3] * d_frag[i3];
            float _fma_3 = __fmaf_rn(dpred, k_frag[i3], dstate[i3]);
            float d_p = _fma_3;
            float _fma_4 = __fmaf_rn(dpred, decayed_state, residual * dstate[i3]);
            float d_k = _fma_4;
            smem_dq[warp * D + dim] = scale * output_adjoint * state_now[i3];
            smem_dk[warp * D + dim] = d_k;
            smem_dlog[warp * D + dim] = d_p * state_prev[i3] * d_frag[i3];
            dstate[i3] = d_p * d_frag[i3];
        }
        if (lane == 0) {
            smem_dbeta[warp] = dr * (value_raw - pred);
            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(dr * beta_value);
            dv[value_index] = _cvt_bf16_0;
        }
        __syncthreads();
        if (warp == 0) {
            #pragma unroll
            for (int i4 = 0; i4 < 4; i4++) {
                int out_dim = elem + i4;
                float dq_sum = 0.0f;
                float dk_sum = 0.0f;
                float dlog_sum = 0.0f;
                #pragma unroll
                for (int source_warp = 0; source_warp < 8; source_warp++) {
                    dq_sum += smem_dq[source_warp * D + out_dim];
                    dk_sum += smem_dk[source_warp * D + out_dim];
                    dlog_sum += smem_dlog[source_warp * D + out_dim];
                }
                atomicAdd(&dq_normalized[token_base + (long long)out_dim], dq_sum);
                atomicAdd(&dk_normalized[token_base + (long long)out_dim], dk_sum);
                atomicAdd(&dlog_decay[token_base + (long long)out_dim], dlog_sum);
            }
            if (lane == 0) {
                float dbeta_sum = 0.0f;
                #pragma unroll
                for (int source_warp2 = 0; source_warp2 < 8; source_warp2++) {
                    dbeta_sum += smem_dbeta[source_warp2];
                }
                atomicAdd(&dbeta_active[beta_index], dbeta_sum);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i5 = 0; i5 < 4; i5++) {
        dinitial_state[state_base + (long long)elem + (long long)i5] = dstate[i5];
    }
}

} // extern "C"

#undef D
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DBETA_OFF
#undef SMEM_SMEM_DBETA_STAGE_BYTES
#undef SMEM_SMEM_DBETA_STRIDE
#undef SMEM_SMEM_DK_OFF
#undef SMEM_SMEM_DK_STAGE_BYTES
#undef SMEM_SMEM_DK_STRIDE
#undef SMEM_SMEM_DLOG_OFF
#undef SMEM_SMEM_DLOG_STAGE_BYTES
#undef SMEM_SMEM_DLOG_STRIDE
#undef SMEM_SMEM_DQ_OFF
#undef SMEM_SMEM_DQ_STAGE_BYTES
#undef SMEM_SMEM_DQ_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef V
#undef smem_dbeta_addr
#undef smem_dk_addr
#undef smem_dlog_addr
#undef smem_dq_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define D 128
#define USE_NORM_TAPE 0

extern "C" {

__global__ __launch_bounds__(32) void
kernel_flashkda_backward_finalize_tokens_bf16_norm(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, float* __restrict__ norm_inv, __nv_bfloat16* __restrict__ g, float* __restrict__ beta_active, float* __restrict__ A_log, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ q_norm, __nv_bfloat16* __restrict__ k_norm, float* __restrict__ dq_normalized, float* __restrict__ dk_normalized, float* __restrict__ gate_common, float* __restrict__ dbeta_active, __nv_bfloat16* __restrict__ dq, __nv_bfloat16* __restrict__ dk, __nv_bfloat16* __restrict__ dg, __nv_bfloat16* __restrict__ dbeta, int num_heads, float lower_bound)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int token = blockIdx.x;
    int head = blockIdx.y;
    int elem = lane * 4;
    long long base = ((long long)token * (long long)num_heads + (long long)head) * (long long)D;
    float g_raw[4];
    float qn[4];
    float kn[4];
    float dqn[4];
    float dkn[4];
    float dlog[4];
    {
        uint2 _vld_0;
        _vld_0 = *reinterpret_cast<const uint2*>(g + base + (long long)elem);
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&g_raw[0 + _pair * 2])[0]), "=f"((&g_raw[0 + _pair * 2])[1])
                : "r"(_vpairs_0[_pair]));
        }
    }
    {
        uint2 _vld_1;
        _vld_1 = *reinterpret_cast<const uint2*>(q_norm + base + (long long)elem);
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&qn[0 + _pair * 2])[0]), "=f"((&qn[0 + _pair * 2])[1])
                : "r"(_vpairs_1[_pair]));
        }
    }
    {
        uint2 _vld_2;
        _vld_2 = *reinterpret_cast<const uint2*>(k_norm + base + (long long)elem);
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&kn[0 + _pair * 2])[0]), "=f"((&kn[0 + _pair * 2])[1])
                : "r"(_vpairs_2[_pair]));
        }
    }
    {
        float4 _v4 = *reinterpret_cast<const float4*>(dq_normalized + base + (long long)elem);
        dqn[0 + 0] = _v4.x;
        dqn[0 + 1] = _v4.y;
        dqn[0 + 2] = _v4.z;
        dqn[0 + 3] = _v4.w;
    }
    {
        float4 _v4 = *reinterpret_cast<const float4*>(dk_normalized + base + (long long)elem);
        dkn[0 + 0] = _v4.x;
        dkn[0 + 1] = _v4.y;
        dkn[0 + 2] = _v4.z;
        dkn[0 + 3] = _v4.w;
    }
    {
        float4 _v4 = *reinterpret_cast<const float4*>(gate_common + base + (long long)elem);
        dlog[0 + 0] = _v4.x;
        dlog[0 + 1] = _v4.y;
        dlog[0 + 2] = _v4.z;
        dlog[0 + 3] = _v4.w;
    }
    float q_dot = 0.0f;
    float k_dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float _fma_0 = __fmaf_rn(dqn[i], qn[i], q_dot);
        q_dot = _fma_0;
        float _fma_1 = __fmaf_rn(dkn[i], kn[i], k_dot);
        k_dot = _fma_1;
    }
    float _warp_reduce_0 = q_dot;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    q_dot = _warp_reduce_0;
    float _warp_reduce_1 = k_dot;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    k_dot = _warp_reduce_1;
    float q_inv = 0.0f;
    float k_inv = 0.0f;
    {
        float q_raw[4];
        float k_raw[4];
        {
            uint2 _vld_6;
            _vld_6 = *reinterpret_cast<const uint2*>(q + base + (long long)elem);
            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&q_raw[0 + _pair * 2])[0]), "=f"((&q_raw[0 + _pair * 2])[1])
                    : "r"(_vpairs_6[_pair]));
            }
        }
        {
            uint2 _vld_7;
            _vld_7 = *reinterpret_cast<const uint2*>(k + base + (long long)elem);
            uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&k_raw[0 + _pair * 2])[0]), "=f"((&k_raw[0 + _pair * 2])[1])
                    : "r"(_vpairs_7[_pair]));
            }
        }
        float q_sq = 0.0f;
        float k_sq = 0.0f;
        #pragma unroll
        for (int norm_i = 0; norm_i < 4; norm_i++) {
            float _fma_2 = __fmaf_rn(q_raw[norm_i], q_raw[norm_i], q_sq);
            q_sq = _fma_2;
            float _fma_3 = __fmaf_rn(k_raw[norm_i], k_raw[norm_i], k_sq);
            k_sq = _fma_3;
        }
        float _warp_reduce_2 = q_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
        q_sq = _warp_reduce_2;
        float _warp_reduce_3 = k_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
        k_sq = _warp_reduce_3;
        float _rsqrt_0 = rsqrtf(q_sq + 1e-06f);
        q_inv = _rsqrt_0;
        float _rsqrt_1 = rsqrtf(k_sq + 1e-06f);
        k_inv = _rsqrt_1;
    }
    float gate_a_lane = 0.0f;
    if (lane == 0) {
        float _expf_0 = __expf(A_log[head]);
        gate_a_lane = _expf_0;
    }
    float _shfl_2 = __shfl_sync(0xFFFFFFFF, gate_a_lane, 0);
    float gate_a = _shfl_2;
    float2 _f2_0 = make_float2(lower_bound * gate_a, lower_bound * gate_a);
    float2 gate_scale_pair = _f2_0;
    float2 _f2_1 = make_float2(q_inv, q_inv);
    float2 q_inv_pair = _f2_1;
    float2 _f2_2 = make_float2(k_inv, k_inv);
    float2 k_inv_pair = _f2_2;
    float2 _f2_3 = make_float2(-q_dot, -q_dot);
    float2 q_dot_pair = _f2_3;
    float2 _f2_4 = make_float2(-k_dot, -k_dot);
    float2 k_dot_pair = _f2_4;
    float2 _f2_5 = make_float2(1.0f, 1.0f);
    float2 one_pair = _f2_5;
    #pragma unroll
    for (int i2 = 0; i2 < 4; i2 += 2) {
        int dim0 = elem + i2;
        int dim1 = dim0 + 1;
        long long index0 = base + (long long)dim0;
        long long index1 = index0 + 1;
        float biased0 = g_raw[i2] + dt_bias[head * D + dim0];
        float biased1 = g_raw[i2 + 1] + dt_bias[head * D + dim1];
        float _expf_1 = __expf((-gate_a) * biased0);
        float _rcp_0 = approx_rcp(1.0f + _expf_1);
        float gate_sigmoid0 = _rcp_0;
        float _expf_2 = __expf((-gate_a) * biased1);
        float _rcp_1 = approx_rcp(1.0f + _expf_2);
        float gate_sigmoid1 = _rcp_1;
        float2 _f2_6 = make_float2(gate_sigmoid0, gate_sigmoid1);
        float2 sigmoid_pair = _f2_6;
        float2 _f2_7 = make_float2(dlog[i2], dlog[i2 + 1]);
        float2 common_pair = mul_f32x2(mul_f32x2(_f2_7, gate_scale_pair), mul_f32x2(sigmoid_pair, sub_f32x2(one_pair, sigmoid_pair)));
        float2 _f2_8 = make_float2(qn[i2], qn[i2 + 1]);
        float2 _f2_9 = make_float2(dqn[i2], dqn[i2 + 1]);
        float2 dq_pair = mul_f32x2(q_inv_pair, fma_f32x2(_f2_8, q_dot_pair, _f2_9));
        float2 _f2_10 = make_float2(kn[i2], kn[i2 + 1]);
        float2 _f2_11 = make_float2(dkn[i2], dkn[i2 + 1]);
        float2 dk_pair = mul_f32x2(k_inv_pair, fma_f32x2(_f2_10, k_dot_pair, _f2_11));
        float output_pair[2];
        output_pair[0] = dq_pair.x;
        output_pair[1] = dq_pair.y;
        {
            __nv_bfloat162 _pk = __floats2bfloat162_rn(output_pair[0 + 0], output_pair[0 + 1]);
            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dq))[index0]) = _pk;
        }
        output_pair[0] = dk_pair.x;
        output_pair[1] = dk_pair.y;
        {
            __nv_bfloat162 _pk = __floats2bfloat162_rn(output_pair[0 + 0], output_pair[0 + 1]);
            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dk))[index0]) = _pk;
        }
        output_pair[0] = common_pair.x;
        output_pair[1] = common_pair.y;
        {
            __nv_bfloat162 _pk = __floats2bfloat162_rn(output_pair[0 + 0], output_pair[0 + 1]);
            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dg))[index0]) = _pk;
        }
        {
            float2 _v2 = make_float2(output_pair[0 + 0], output_pair[0 + 1]);
            *reinterpret_cast<float2*>(gate_common + index0) = _v2;
        }
    }
    if (lane == 0) {
        long long beta_index = (long long)token * (long long)num_heads + (long long)head;
        float beta_sigmoid = beta_active[beta_index];
        __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(dbeta_active[beta_index] * beta_sigmoid * (1.0f - beta_sigmoid));
        dbeta[beta_index] = _cvt_bf16_0;
    }
}

} // extern "C"

#undef D
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef USE_NORM_TAPE
