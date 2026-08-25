typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed int int32_t;
typedef short int int16_t;
struct __align__(128) LoomTensorMap {
  uint64_t opaque[16];
};
template <int N>
struct __align__(128) LoomTensorMapPack {
  LoomTensorMap maps[N];
};

typedef struct __align__(64) {
  uint64_t opaque[16];
} CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
  int result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;" : "=r"(result) : "r"(x));
  return result;
}

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

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_WARP_MAXIMA_OFF 0
#define SMEM_WARP_MAXIMA_STAGE_BYTES 32
#define SMEM_WARP_MAXIMA_STRIDE 32
#define SMEM_WARMUP_OFF 32
#define SMEM_WARMUP_STAGE_BYTES 8
#define SMEM_WARMUP_STRIDE 8
#define SMEM_TOTAL 128
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void kernel_flashkda_refine_forgetting_horizons(
    __nv_bfloat16* __restrict__ g, float* __restrict__ A_log, float* __restrict__ dt_bias,
    int* __restrict__ work_items, int* __restrict__ boundaries,
    unsigned int* __restrict__ persistent_counters, int num_heads, float lower_bound,
    float log2_threshold) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  // Kernel setup ops
  float* warp_maxima = reinterpret_cast<float*>(smem_raw + 0);
  const int warp_maxima_addr = smem + 0;
  int* warmup = reinterpret_cast<int*>(smem_raw + 32);
  const int warmup_addr = smem + 32;

  // === Task calls (dependency order) ===
  int boundary = blockIdx.x;
  if (boundary == 0 && tid < num_heads + 2) {
    persistent_counters[tid] = 0;
  }
  int previous_row = boundaries[boundary * 2];
  int next_row = boundaries[boundary * 2 + 1];
  int previous_base = previous_row * 8;
  int next_base = next_row * 8;
  int head = work_items[previous_base + 1];
  int cut = work_items[previous_base + 3];
  int bos = work_items[previous_base + 6];
  int eos = work_items[previous_base + 7];
  int num_chunks = (eos - bos) / 16;
  int channel = tid;
  float _expf_0 = __expf(A_log[head]);
  float gate_rate = _expf_0;
  float gate_bias = dt_bias[head * 128 + channel];
  float left_sum = 0.0f;
  float right_sum = 0.0f;
  if (tid == 0) {
    warmup[0] = 0;
    warmup[1] = 0;
  }
  __syncthreads();
#pragma unroll 1
  for (int depth = 0; depth < 32; depth++) {
    if (warmup[0] == 0) {
      int left_chunk = cut - 1 - depth;
      if (left_chunk >= 0) {
#pragma unroll
        for (int sample = 0; sample < 4; sample++) {
          int left_token = bos + left_chunk * 16 + sample * 4;
          long long left_index =
              ((long long)left_token * (long long)num_heads + (long long)head) * 128 +
              (long long)channel;
          float _tanh_approx_0;
          asm volatile("tanh.approx.f32 %0, %1;"
                       : "=f"(_tanh_approx_0)
                       : "f"(gate_rate * ((float)g[left_index] + gate_bias) * 0.5f));
          float left_sigmoid = _tanh_approx_0 * 0.5f + 0.5f;
          left_sum += lower_bound * 1.4426950408889634f * left_sigmoid;
        }
      }
    }
    if (warmup[1] == 0) {
      int right_chunk = cut + depth;
      if (right_chunk < num_chunks) {
#pragma unroll
        for (int sample_1 = 0; sample_1 < 4; sample_1++) {
          int right_token = bos + right_chunk * 16 + sample_1 * 4;
          long long right_index =
              ((long long)right_token * (long long)num_heads + (long long)head) * 128 +
              (long long)channel;
          float _tanh_approx_1;
          asm volatile("tanh.approx.f32 %0, %1;"
                       : "=f"(_tanh_approx_1)
                       : "f"(gate_rate * ((float)g[right_index] + gate_bias) * 0.5f));
          float right_sigmoid = _tanh_approx_1 * 0.5f + 0.5f;
          right_sum += lower_bound * 1.4426950408889634f * right_sigmoid;
        }
      }
    }
    float left_max = left_sum;
    float right_max = right_sum;
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, left_max, 16);
    float left_other1 = _shfl_xor_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, right_max, 16);
    float right_other1 = _shfl_xor_1;
    if (left_other1 > left_max) {
      left_max = left_other1;
    }
    if (right_other1 > right_max) {
      right_max = right_other1;
    }
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, left_max, 8);
    float left_other2 = _shfl_xor_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, right_max, 8);
    float right_other2 = _shfl_xor_3;
    if (left_other2 > left_max) {
      left_max = left_other2;
    }
    if (right_other2 > right_max) {
      right_max = right_other2;
    }
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, left_max, 4);
    float left_other3 = _shfl_xor_4;
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, right_max, 4);
    float right_other3 = _shfl_xor_5;
    if (left_other3 > left_max) {
      left_max = left_other3;
    }
    if (right_other3 > right_max) {
      right_max = right_other3;
    }
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, left_max, 2);
    float left_other4 = _shfl_xor_6;
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, right_max, 2);
    float right_other4 = _shfl_xor_7;
    if (left_other4 > left_max) {
      left_max = left_other4;
    }
    if (right_other4 > right_max) {
      right_max = right_other4;
    }
    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, left_max, 1);
    float left_other5 = _shfl_xor_8;
    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, right_max, 1);
    float right_other5 = _shfl_xor_9;
    if (left_other5 > left_max) {
      left_max = left_other5;
    }
    if (right_other5 > right_max) {
      right_max = right_other5;
    }
    if (lane == 0) {
      warp_maxima[warp * 2] = left_max;
      warp_maxima[warp * 2 + 1] = right_max;
    }
    __syncthreads();
    if (tid == 0) {
      float cta_left_max = warp_maxima[0];
      float cta_right_max = warp_maxima[1];
#pragma unroll
      for (int reduce_warp = 1; reduce_warp < 4; reduce_warp++) {
        float warp_left = warp_maxima[reduce_warp * 2];
        float warp_right = warp_maxima[reduce_warp * 2 + 1];
        if (warp_left > cta_left_max) {
          cta_left_max = warp_left;
        }
        if (warp_right > cta_right_max) {
          cta_right_max = warp_right;
        }
      }
      if (warmup[0] == 0 && cta_left_max <= log2_threshold) {
        warmup[0] = depth + 1;
      }
      if (warmup[1] == 0 && cta_right_max <= log2_threshold) {
        warmup[1] = depth + 1;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    int left_warmup = warmup[0];
    int right_warmup = warmup[1];
    int cstart = 0;
    int cend = num_chunks;
    if (left_warmup != 0) {
      cstart = cut - left_warmup;
    }
    if (right_warmup != 0) {
      cend = cut + right_warmup;
    }
    work_items[next_base + 4] = cstart;
    work_items[previous_base + 5] = cend;
  }
}

}  // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_TOTAL
#undef SMEM_WARMUP_OFF
#undef SMEM_WARMUP_STAGE_BYTES
#undef SMEM_WARMUP_STRIDE
#undef SMEM_WARP_MAXIMA_OFF
#undef SMEM_WARP_MAXIMA_STAGE_BYTES
#undef SMEM_WARP_MAXIMA_STRIDE
#undef THREADS
#undef warmup_addr
#undef warp_maxima_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_PARTIAL_A_OFF 0
#define SMEM_PARTIAL_A_STAGE_BYTES 2048
#define SMEM_PARTIAL_A_STRIDE 2048
#define SMEM_PARTIAL_DT_OFF 2048
#define SMEM_PARTIAL_DT_STAGE_BYTES 2048
#define SMEM_PARTIAL_DT_STRIDE 2048
#define SMEM_FINISH_FLAG_OFF 4096
#define SMEM_FINISH_FLAG_STAGE_BYTES 4
#define SMEM_FINISH_FLAG_STRIDE 4
#define SMEM_TOTAL 4224
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void kernel_flashkda_backward_param_reduce_c16_partial(
    __nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ beta_active,
    float* __restrict__ A_log, float* __restrict__ dt_bias, float* __restrict__ dlog_decay,
    float* __restrict__ dlog_boundary, float* __restrict__ dbeta_active,
    __nv_bfloat16* __restrict__ dg, __nv_bfloat16* __restrict__ dbeta,
    float* __restrict__ gate_part_a, float* __restrict__ gate_part_dt,
    unsigned int* __restrict__ gate_finish_counters, float* __restrict__ dA_log,
    float* __restrict__ ddt_bias, int total_tokens, int num_heads, int beta_active_stride,
    int slice_len, float lower_bound) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  // Kernel setup ops
  float* partial_a = reinterpret_cast<float*>(smem_raw + 0);
  const int partial_a_addr = smem + 0;
  float* partial_dt = reinterpret_cast<float*>(smem_raw + 2048);
  const int partial_dt_addr = smem + 2048;
  int* finish_flag = reinterpret_cast<int*>(smem_raw + 4096);
  const int finish_flag_addr = smem + 4096;

  // === Task calls (dependency order) ===
  int stripe = blockIdx.x;
  int head = blockIdx.y;
  int warp_0 = warp;
  int dim0 = lane * 4;
  float _expf_0 = __expf(A_log[head]);
  float gate_rate = _expf_0;
  float bias[4];
  float acc_a[4];
  float acc_dt[4];
#pragma unroll
  for (int q0 = 0; q0 < 4; q0++) {
    bias[q0] = dt_bias[head * 128 + dim0 + q0];
    acc_a[q0] = 0.0f;
    acc_dt[q0] = 0.0f;
  }
  int token_begin = stripe * slice_len;
  int token_end = token_begin + slice_len;
  if (token_end > total_tokens) {
    token_end = total_tokens;
  }
#pragma unroll 1
  for (int token = token_begin + warp_0; token < token_end; token += 4) {
    long long gate_index =
        ((long long)token * (long long)num_heads + (long long)head) * 128 + (long long)dim0;
    float dgate_frag[4];
    float gate_frag[4];
    float dg_frag[4];
    {
      float4 _v4 = *reinterpret_cast<const float4*>(dlog_decay + gate_index);
      dgate_frag[0 + 0] = _v4.x;
      dgate_frag[0 + 1] = _v4.y;
      dgate_frag[0 + 2] = _v4.z;
      dgate_frag[0 + 3] = _v4.w;
    }
    if (token % 16 == 0) {
      long long boundary_index =
          ((long long)token / 16 * (long long)num_heads + (long long)head) * 128 + (long long)dim0;
      {
        float4 _v4 = *reinterpret_cast<const float4*>(dlog_boundary + boundary_index);
        dgate_frag[0 + 0] = _v4.x;
        dgate_frag[0 + 1] = _v4.y;
        dgate_frag[0 + 2] = _v4.z;
        dgate_frag[0 + 3] = _v4.w;
      }
      {
        float4 _v4 =
            make_float4(dgate_frag[0 + 0], dgate_frag[0 + 1], dgate_frag[0 + 2], dgate_frag[0 + 3]);
        *reinterpret_cast<float4*>(dlog_decay + gate_index) = _v4;
      }
    }
    {
      uint2 _vld_2;
      _vld_2 = *reinterpret_cast<const uint2*>(g + gate_index);
      uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&gate_frag[0 + _pair * 2])[0]), "=f"((&gate_frag[0 + _pair * 2])[1])
            : "r"(_vpairs_2[_pair]));
      }
    }
#pragma unroll
    for (int q1 = 0; q1 < 4; q1++) {
      float biased = gate_frag[q1] + bias[q1];
      float z = gate_rate * biased;
      float _tanh_approx_0;
      asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(z * 0.5f));
      float gate_sigmoid = _tanh_approx_0 * 0.5f + 0.5f;
      float sigmoid_prime = gate_sigmoid * (1.0f - gate_sigmoid);
      float weighted = dgate_frag[q1] * lower_bound * sigmoid_prime;
      float raw = weighted * gate_rate;
      float _fma_0 = __fmaf_rn(weighted, z, acc_a[q1]);
      acc_a[q1] = _fma_0;
      __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(raw);
      float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
      acc_dt[q1] = acc_dt[q1] + _cvt_f32_0;
      dg_frag[q1] = raw;
    }
    {
      __nv_bfloat162 _pk = __floats2bfloat162_rn(dg_frag[0 + 0], dg_frag[0 + 1]);
      *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dg))[gate_index]) = _pk;
    }
    {
      __nv_bfloat162 _pk = __floats2bfloat162_rn(dg_frag[2 + 0], dg_frag[2 + 1]);
      *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dg))[gate_index + 2]) = _pk;
    }
  }
#pragma unroll
  for (int q2 = 0; q2 < 4; q2++) {
    partial_a[warp_0 * 128 + dim0 + q2] = acc_a[q2];
    partial_dt[warp_0 * 128 + dim0 + q2] = acc_dt[q2];
  }
  __syncthreads();
  if (warp_0 == 0) {
#pragma unroll
    for (int q3 = 0; q3 < 4; q3++) {
      int dim = dim0 + q3;
      int base = (stripe * num_heads + head) * 128 + dim;
      gate_part_a[base] =
          partial_a[dim] + partial_a[128 + dim] + partial_a[256 + dim] + partial_a[384 + dim];
      gate_part_dt[base] =
          partial_dt[dim] + partial_dt[128 + dim] + partial_dt[256 + dim] + partial_dt[384 + dim];
    }
  }
#pragma unroll 1
  for (int beta_token = token_begin + tid; beta_token < token_end; beta_token += 128) {
    long long beta_index = (long long)beta_token * (long long)num_heads + (long long)head;
    long long beta_active_index =
        (long long)beta_token * (long long)beta_active_stride + (long long)head;
    float beta_value = (float)beta_active[beta_active_index];
    __nv_bfloat16 _cvt_bf16_1 =
        __float2bfloat16(dbeta_active[beta_index] * beta_value * (1.0f - beta_value));
    dbeta[beta_index] = _cvt_bf16_1;
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    unsigned int _atomic_old_0 = atomicAdd(&gate_finish_counters[head], 1);
    unsigned int old_count = _atomic_old_0;
    finish_flag[0] = ((old_count + 1 == 128) ? 1 : 0);
  }
  __syncthreads();
  if (finish_flag[0] != 0) {
    __threadfence();
    int dim_1 = tid;
    float a8[8];
    float dt8[8];
#pragma unroll
    for (int j0 = 0; j0 < 8; j0++) {
      a8[j0] = 0.0f;
      dt8[j0] = 0.0f;
    }
#pragma unroll 1
    for (int chain = 0; chain < 16; chain++) {
#pragma unroll
      for (int j1 = 0; j1 < 8; j1++) {
        int part_index = ((chain * 8 + j1) * num_heads + head) * 128 + dim_1;
        a8[j1] = a8[j1] + gate_part_a[part_index];
        dt8[j1] = dt8[j1] + gate_part_dt[part_index];
      }
    }
    float p0a = a8[0] + a8[1];
    float p1a = a8[2] + a8[3];
    float p2a = a8[4] + a8[5];
    float p3a = a8[6] + a8[7];
    float p0dt = dt8[0] + dt8[1];
    float p1dt = dt8[2] + dt8[3];
    float p2dt = dt8[4] + dt8[5];
    float p3dt = dt8[6] + dt8[7];
    float col_a = p0a + p1a + (p2a + p3a);
    float col_dt = p0dt + p1dt + (p2dt + p3dt);
    __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(col_dt);
    float _cvt_f32_1 = __bfloat162float(_cvt_bf16_2);
    ddt_bias[head * 128 + dim_1] = _cvt_f32_1;
    float _warp_reduce_0 = col_a;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    col_a = _warp_reduce_0;
    if (lane == 0) {
      partial_a[warp] = col_a;
    }
    __syncthreads();
    if (warp == 0) {
      if (elect_sync()) {
        dA_log[head] = partial_a[0] + partial_a[1] + partial_a[2] + partial_a[3];
      }
    }
  }
}

}  // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_FINISH_FLAG_OFF
#undef SMEM_FINISH_FLAG_STAGE_BYTES
#undef SMEM_FINISH_FLAG_STRIDE
#undef SMEM_PARTIAL_A_OFF
#undef SMEM_PARTIAL_A_STAGE_BYTES
#undef SMEM_PARTIAL_A_STRIDE
#undef SMEM_PARTIAL_DT_OFF
#undef SMEM_PARTIAL_DT_STAGE_BYTES
#undef SMEM_PARTIAL_DT_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef finish_flag_addr
#undef partial_a_addr
#undef partial_dt_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void kernel_flashkda_grouped_qk_expand(
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ q_value_heads, __nv_bfloat16* __restrict__ k_value_heads,
    int total_tokens, int num_qk_heads, int num_v_heads) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  // === Task calls (dependency order) ===
  int token_begin = blockIdx.x * 32;
  int value_head = blockIdx.y;
  int channel = tid;
  int group_size = num_v_heads / num_qk_heads;
  int qk_head = value_head / group_size;
#pragma unroll
  for (int token_local = 0; token_local < 32; token_local++) {
    int token = token_begin + token_local;
    if (token < total_tokens) {
      long long source_index =
          ((long long)token * (long long)num_qk_heads + (long long)qk_head) * 128 +
          (long long)channel;
      long long output_index =
          ((long long)token * (long long)num_v_heads + (long long)value_head) * 128 +
          (long long)channel;
      q_value_heads[output_index] = q[source_index];
      k_value_heads[output_index] = k[source_index];
    }
  }
}

}  // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void kernel_flashkda_grouped_qk_reduce(
    __nv_bfloat16* __restrict__ dq_value_heads, __nv_bfloat16* __restrict__ dk_value_heads,
    __nv_bfloat16* __restrict__ dq, __nv_bfloat16* __restrict__ dk, int total_tokens,
    int num_qk_heads, int num_v_heads) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  // === Task calls (dependency order) ===
  int token_local = tid / 8;
  int channel_base = tid % 8 * 16;
  int token = blockIdx.x * 16 + token_local;
  int qk_head = blockIdx.y;
  int group_size = num_v_heads / num_qk_heads;
  int value_head_begin = qk_head * group_size;
  if (token < total_tokens) {
    float dq_sum[16];
    float dk_sum[16];
    dq_sum[0] = 0.0f;
    dq_sum[1] = 0.0f;
    dq_sum[2] = 0.0f;
    dq_sum[3] = 0.0f;
    dq_sum[4] = 0.0f;
    dq_sum[5] = 0.0f;
    dq_sum[6] = 0.0f;
    dq_sum[7] = 0.0f;
    dq_sum[8] = 0.0f;
    dq_sum[9] = 0.0f;
    dq_sum[10] = 0.0f;
    dq_sum[11] = 0.0f;
    dq_sum[12] = 0.0f;
    dq_sum[13] = 0.0f;
    dq_sum[14] = 0.0f;
    dq_sum[15] = 0.0f;
    dk_sum[0] = 0.0f;
    dk_sum[1] = 0.0f;
    dk_sum[2] = 0.0f;
    dk_sum[3] = 0.0f;
    dk_sum[4] = 0.0f;
    dk_sum[5] = 0.0f;
    dk_sum[6] = 0.0f;
    dk_sum[7] = 0.0f;
    dk_sum[8] = 0.0f;
    dk_sum[9] = 0.0f;
    dk_sum[10] = 0.0f;
    dk_sum[11] = 0.0f;
    dk_sum[12] = 0.0f;
    dk_sum[13] = 0.0f;
    dk_sum[14] = 0.0f;
    dk_sum[15] = 0.0f;
#pragma unroll 1
    for (int group_head = 0; group_head < group_size; group_head++) {
      int value_head = value_head_begin + group_head;
      long long source_index =
          ((long long)token * (long long)num_v_heads + (long long)value_head) * 128 +
          (long long)channel_base;
      float _vec_load_0[16];
      {
        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(dq_value_heads + source_index);
        uint4 _vld_0[2];
#pragma unroll
        for (int _blk = 0; _blk < 2; _blk++) {
          _vld_0[_blk] = _vptr_0[_blk];
          uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]),
                  "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                : "r"(_vpairs_0[_pair]));
          }
        }
      }
      float _vec_load_1[16];
      {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(dk_value_heads + source_index);
        uint4 _vld_1[2];
#pragma unroll
        for (int _blk = 0; _blk < 2; _blk++) {
          _vld_1[_blk] = _vptr_1[_blk];
          uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]),
                  "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                : "r"(_vpairs_1[_pair]));
          }
        }
      }
#pragma unroll
      for (int channel_local = 0; channel_local < 16; channel_local++) {
        dq_sum[channel_local] = dq_sum[channel_local] + _vec_load_0[channel_local];
        dk_sum[channel_local] = dk_sum[channel_local] + _vec_load_1[channel_local];
      }
    }
    long long output_index =
        ((long long)token * (long long)num_qk_heads + (long long)qk_head) * 128 +
        (long long)channel_base;
    {
      __nv_bfloat162 _pk[8];
      _pk[0] = __floats2bfloat162_rn(dq_sum[0 + 0], dq_sum[0 + 1]);
      _pk[1] = __floats2bfloat162_rn(dq_sum[0 + 2], dq_sum[0 + 3]);
      _pk[2] = __floats2bfloat162_rn(dq_sum[0 + 4], dq_sum[0 + 5]);
      _pk[3] = __floats2bfloat162_rn(dq_sum[0 + 6], dq_sum[0 + 7]);
      _pk[4] = __floats2bfloat162_rn(dq_sum[0 + 8], dq_sum[0 + 9]);
      _pk[5] = __floats2bfloat162_rn(dq_sum[0 + 10], dq_sum[0 + 11]);
      _pk[6] = __floats2bfloat162_rn(dq_sum[0 + 12], dq_sum[0 + 13]);
      _pk[7] = __floats2bfloat162_rn(dq_sum[0 + 14], dq_sum[0 + 15]);
      *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dq))[output_index + 0]) =
          *reinterpret_cast<uint4*>(&_pk[0]);
      *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dq))[output_index + 8]) =
          *reinterpret_cast<uint4*>(&_pk[4]);
    }
    {
      __nv_bfloat162 _pk[8];
      _pk[0] = __floats2bfloat162_rn(dk_sum[0 + 0], dk_sum[0 + 1]);
      _pk[1] = __floats2bfloat162_rn(dk_sum[0 + 2], dk_sum[0 + 3]);
      _pk[2] = __floats2bfloat162_rn(dk_sum[0 + 4], dk_sum[0 + 5]);
      _pk[3] = __floats2bfloat162_rn(dk_sum[0 + 6], dk_sum[0 + 7]);
      _pk[4] = __floats2bfloat162_rn(dk_sum[0 + 8], dk_sum[0 + 9]);
      _pk[5] = __floats2bfloat162_rn(dk_sum[0 + 10], dk_sum[0 + 11]);
      _pk[6] = __floats2bfloat162_rn(dk_sum[0 + 12], dk_sum[0 + 13]);
      _pk[7] = __floats2bfloat162_rn(dk_sum[0 + 14], dk_sum[0 + 15]);
      *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dk))[output_index + 0]) =
          *reinterpret_cast<uint4*>(&_pk[0]);
      *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dk))[output_index + 8]) =
          *reinterpret_cast<uint4*>(&_pk[4]);
    }
  }
}

}  // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS
