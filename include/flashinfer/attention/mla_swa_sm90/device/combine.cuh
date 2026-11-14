template <class InputT, int HeadDimV = 1024, int MaxSplits = 2, class PartialT = float>
__global__ void dense_swa_roles_split_combine_kernel(const PartialT* partial, const float* lse,
                                                     InputT* out, int total_rows, int num_splits) {
  const int row = blockIdx.x;
  if (row >= total_rows) {
    return;
  }
  __shared__ float weights[MaxSplits];
  __shared__ float weight_sum;
  if (threadIdx.x == 0) {
    float row_max = -CUDART_INF_F;
#pragma unroll
    for (int split = 0; split < MaxSplits; ++split) {
      if (split < num_splits) {
        row_max = fmaxf(row_max, lse[row * num_splits + split]);
      }
    }
    float sum = 0.0f;
#pragma unroll
    for (int split = 0; split < MaxSplits; ++split) {
      if (split < num_splits) {
        const float value = lse[row * num_splits + split];
        const float weight = isfinite(value) ? exp2f(value - row_max) : 0.0f;
        weights[split] = weight;
        sum += weight;
      }
    }
    weight_sum = sum;
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
