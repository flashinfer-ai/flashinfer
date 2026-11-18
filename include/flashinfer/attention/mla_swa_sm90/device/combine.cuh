template <class InputT, int HeadDimV = 1024, int MaxSplits = 2, class PartialT = float>
__global__ void dense_swa_roles_split_combine_kernel(const PartialT* partial, const float* lse,
                                                     InputT* out, int total_rows, int num_splits) {
  const int row = blockIdx.x;
  if (row >= total_rows) {
    return;
  }
  __shared__ float weights[MaxSplits];
  __shared__ float weight_sum;
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    const float value =
        lane < num_splits && lane < MaxSplits ? lse[row * num_splits + lane] : -CUDART_INF_F;
    const float maximum = softmax_ordered_warp_max(value);
    const float weight = isfinite(value) ? exp2f(value - maximum) : 0.0f;
    if (lane < num_splits && lane < MaxSplits) weights[lane] = weight;
    float sum = weight;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) weight_sum = sum;
  }
  __syncthreads();
  const float inv_sum = weight_sum > 0.0f ? 1.0f / weight_sum : 0.0f;
  float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int split = 0; split < MaxSplits; ++split) {
    if (split < num_splits) {
      const float weight = weights[split];
      const PartialT* split_row = partial + (row * num_splits + split) * HeadDimV;
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int dim = threadIdx.x + item * 256;
        values[item] += weight * float(split_row[dim]);
      }
    }
  }
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int dim = threadIdx.x + item * 256;
    out[row * HeadDimV + dim] = InputT(values[item] * inv_sum);
  }
}

// A bounded cooperative grid reuses the unchanged producer body and FP32
// workspace. No DSM, per-invocation counters, or workspace initialization is
// required: grid.sync() publishes every partial before any consumer runs.
template <typename T, typename TmaParamsT>
__global__ void __launch_bounds__(T::NUM_THREADS, 1) dense_swa_cooperative_split_kernel(
    const typename T::InputT* q, const typename T::InputT* kv, const int* block_table,
    const int* cache_seqlens, typename T::InputT* partial_out, float* lse, int num_splits,
    int64_t out_stride_split, int batch_size, int s_q, int num_heads, int max_pages,
    int window_size, float softmax_scale, int64_t q_stride_b, int64_t q_stride_s,
    int64_t q_stride_h, int64_t kv_stride_token, int64_t block_table_stride_b, int64_t out_stride_b,
    int64_t out_stride_s, int64_t out_stride_h, __grid_constant__ const TmaParamsT tma_params,
    typename T::InputT* final_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using C = typename T::Config;
  using InputT = typename T::InputT;
  static_assert(((C::M == 16 && C::S == 4 && C::PvWG == 4 && C::QkPV == 0) ||
                 (C::M == 8 && (C::S == 3 || C::S == 5 || C::S == 9) && C::PvWG == 3)) &&
                C::Backend == mla_swa::Reduce::GlobalFP32);
  attention_body<T>(q, kv, block_table, cache_seqlens, partial_out, lse, num_splits,
                    out_stride_split, batch_size, s_q, num_heads, max_pages, window_size,
                    softmax_scale, q_stride_b, q_stride_s, q_stride_h, kv_stride_token,
                    block_table_stride_b, out_stride_b, out_stride_s, out_stride_h, tma_params);
  cooperative_groups::this_grid().sync();
  if constexpr (C::M == 8) {
    // Two independent 256-thread row owners. Split rank owns every S-th
    // row, so S5 has at most two rows and S9 at most one per producer CTA.
    extern __shared__ char cooperative_smem[];
    float* weights = reinterpret_cast<float*>(cooperative_smem);
    constexpr int owner_threads = C::S == 3 ? 128 : 256;
    constexpr int chunks = 1024 / (owner_threads * 4);
    const int group = threadIdx.x / owner_threads;
    const int idx = threadIdx.x % owner_threads;
    const int row_in_tile = blockIdx.z + group * C::S;
    const int total_rows = s_q * num_heads;
    const int row_in_request = blockIdx.y * C::M + row_in_tile;
    const bool valid = row_in_tile < C::M && row_in_request < total_rows;
    const int row = blockIdx.x * total_rows + row_in_request;
    float* group_weights = weights + group * (C::S + 1);
    float4 payload[chunks][C::S];
    if (valid) {
      const float* partial = reinterpret_cast<const float*>(partial_out);
#pragma unroll
      for (int chunk = 0; chunk < chunks; ++chunk) {
#pragma unroll
        for (int split = 0; split < C::S; ++split) {
          const int dim = idx * 4 + chunk * owner_threads * 4;
          payload[chunk][split] =
              *reinterpret_cast<const float4*>(partial + (row * C::S + split) * 1024 + dim);
        }
      }
    }
    if constexpr (C::S == 3) {
      if (valid && idx < 32) {
        const float value = idx < C::S ? lse[row * C::S + idx] : -CUDART_INF_F;
        const float maximum = softmax_ordered_warp_max(value);
        const float weight = isfinite(value) ? exp2f(value - maximum) : 0.0f;
        if (idx < C::S) group_weights[idx] = weight;
        float sum = weight;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (idx == 0) group_weights[C::S] = sum;
      }
      // All 512 threads participate even when their row group is inactive.
      __syncthreads();
    }
    if (valid) {
      float sum;
      float lane_weight = 0.0f;
      if constexpr (C::S == 3) {
        sum = group_weights[C::S];
      } else {
        const int lane = idx & 31;
        const float value = lane < C::S ? lse[row * C::S + lane] : -CUDART_INF_F;
        const float maximum = softmax_ordered_warp_max(value);
        lane_weight = isfinite(value) ? exp2f(value - maximum) : 0.0f;
        float warp_sum = lane_weight;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          warp_sum += __shfl_down_sync(0xffffffffu, warp_sum, offset);
        sum = __shfl_sync(0xffffffffu, warp_sum, 0);
      }
      const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
#pragma unroll
      for (int chunk = 0; chunk < chunks; ++chunk) {
        const int dim = idx * 4 + chunk * owner_threads * 4;
        float accum[4] = {};
#pragma unroll
        for (int split = 0; split < C::S; ++split) {
          const float weight = [&]() {
            if constexpr (C::S == 3)
              return group_weights[split];
            else
              return __shfl_sync(0xffffffffu, lane_weight, split);
          }();
          const float4 value = payload[chunk][split];
          accum[0] += weight * value.x;
          accum[1] += weight * value.y;
          accum[2] += weight * value.z;
          accum[3] += weight * value.w;
        }
        cutlass::AlignedArray<InputT, 4, 8> packed;
#pragma unroll
        for (int item = 0; item < 4; ++item) packed[item] = InputT(accum[item] * inv_sum);
        *reinterpret_cast<cutlass::AlignedArray<InputT, 4, 8>*>(final_out + row * 1024 + dim) =
            packed;
      }
    }
  } else {
    const int wg = threadIdx.x / 128;
    if (wg > 0) {
      // Four owners per split, one row per PV WG. Each output row is written
      // once; the QK WG does not participate in this final reduce.
      const int row_in_request = blockIdx.y * C::M + blockIdx.z * 4 + wg - 1;
      const int total_rows = s_q * num_heads;
      if (row_in_request < total_rows) {
        const int row = blockIdx.x * total_rows + row_in_request;
        const int lane = threadIdx.x & 31;
        const int idx = threadIdx.x & 127;
        const float* partial = reinterpret_cast<const float*>(partial_out);
        float4 payload[2][C::S];
#pragma unroll
        for (int chunk = 0; chunk < 2; ++chunk) {
#pragma unroll
          for (int split = 0; split < C::S; ++split) {
            const int dim = idx * 4 + chunk * 512;
            payload[chunk][split] =
                *reinterpret_cast<const float4*>(partial + (row * C::S + split) * 1024 + dim);
          }
        }
        const float value = lane < C::S ? lse[row * C::S + lane] : -CUDART_INF_F;
        const float maximum = softmax_ordered_warp_max(value);
        const float lane_weight = isfinite(value) ? exp2f(value - maximum) : 0.0f;
        float sum = lane_weight;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          sum += __shfl_down_sync(0xffffffffu, sum, offset);
        const float weight_sum = __shfl_sync(0xffffffffu, sum, 0);
        const float inv_sum = weight_sum > 0.0f ? 1.0f / weight_sum : 0.0f;
#pragma unroll
        for (int chunk = 0; chunk < 2; ++chunk) {
          const int dim = idx * 4 + chunk * 512;
          float accum[4] = {};
#pragma unroll
          for (int split = 0; split < C::S; ++split) {
            const float weight = __shfl_sync(0xffffffffu, lane_weight, split);
            const float4 packed = payload[chunk][split];
            accum[0] += weight * packed.x;
            accum[1] += weight * packed.y;
            accum[2] += weight * packed.z;
            accum[3] += weight * packed.w;
          }
          cutlass::AlignedArray<InputT, 4, 8> packed;
#pragma unroll
          for (int item = 0; item < 4; ++item) packed[item] = InputT(accum[item] * inv_sum);
          *reinterpret_cast<cutlass::AlignedArray<InputT, 4, 8>*>(final_out + row * 1024 + dim) =
              packed;
        }
      }
    }
  }
#endif
}
