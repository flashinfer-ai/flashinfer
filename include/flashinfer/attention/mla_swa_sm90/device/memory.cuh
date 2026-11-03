#pragma once

__forceinline__ __device__ void fragment_store4(float* ptr, float a, float b, float c, float d) {
  const uint32_t address = cute::cast_smem_ptr_to_uint(ptr);
  asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" ::"r"(address), "f"(a), "f"(b), "f"(c),
               "f"(d)
               : "memory");
}

__forceinline__ __device__ void fragment_load4(uint32_t address, float& a, float& b, float& c,
                                               float& d) {
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(a), "=f"(b), "=f"(c), "=f"(d)
               : "r"(address)
               : "memory");
}

__forceinline__ __device__ int fragment_m0(int warp_in_group, int lane) {
  return warp_in_group * 16 + lane / 4;
}

template <int MaxSlices, typename T, typename TensorOutput, typename Plan>
__forceinline__ __device__ void write_output_slices(
    TensorOutput const& output_fragment_all, int output_slice_offset, int slice_begin,
    int slice_end, Plan& plan, typename T::InputT* out, int batch_idx, int m_block_idx,
    int total_rows, int num_heads, int64_t out_stride_b, int64_t out_stride_s, int64_t out_stride_h,
    int warp_in_group, int lane) {
  static_assert(MaxSlices > 0);
  const int m0 = fragment_m0(warp_in_group, lane);
  const int m1 = m0 + 8;
#pragma unroll
  for (int group = 0; group < T::BLOCK_M / 8; ++group) {
    const int n0 = group * 8 + (lane % 4) * 2;
    const int n1 = n0 + 1;
    const int base = group * 4;
    const int global_row0 = m_block_idx * T::BLOCK_M + n0;
    const int global_row1 = global_row0 + 1;
    typename T::InputT* row0 = nullptr;
    typename T::InputT* row1 = nullptr;
    if (global_row0 < total_rows) {
      const int query_idx = global_row0 >> 6;
      const int head_idx = global_row0 & 63;
      row0 = out + batch_idx * out_stride_b + query_idx * out_stride_s + head_idx * out_stride_h;
    }
    if (global_row1 < total_rows) {
      const int query_idx = global_row1 >> 6;
      const int head_idx = global_row1 & 63;
      row1 = out + batch_idx * out_stride_b + query_idx * out_stride_s + head_idx * out_stride_h;
    }
    const float inv_sum0 = plan.row_sum[n0] > 0.0f ? 1.0f / plan.row_sum[n0] : 0.0f;
    const float inv_sum1 = plan.row_sum[n1] > 0.0f ? 1.0f / plan.row_sum[n1] : 0.0f;
#pragma unroll
    for (int slice = 0; slice < MaxSlices; ++slice) {
      if (slice_begin + slice >= slice_end) {
        continue;
      }
      Tensor output_fragment_flat = output_fragment_all(_, output_slice_offset + slice, _0{});
      Tensor output_fragment =
          make_tensor(output_fragment_flat.data(), typename T::OutputSliceFragmentLayout{});
      const int global_slice = slice_begin + slice;
      const int dim0 = global_slice * T::TILE_K + m0;
      const int dim1 = global_slice * T::TILE_K + m1;
      const float value00 = output_fragment(base + 0) * inv_sum0;
      const float value01 = output_fragment(base + 1) * inv_sum1;
      const float value10 = output_fragment(base + 2) * inv_sum0;
      const float value11 = output_fragment(base + 3) * inv_sum1;
      const float peer00 = __shfl_down_sync(0xffffffff, value00, 4);
      const float peer01 = __shfl_down_sync(0xffffffff, value01, 4);
      const float peer10 = __shfl_down_sync(0xffffffff, value10, 4);
      const float peer11 = __shfl_down_sync(0xffffffff, value11, 4);
      if ((lane & 4) == 0) {
        cutlass::Array<typename T::InputT, 2> pair;
        if (row0 != nullptr) {
          pair[0] = static_cast<typename T::InputT>(value00);
          pair[1] = static_cast<typename T::InputT>(peer00);
          *reinterpret_cast<cutlass::Array<typename T::InputT, 2>*>(row0 + dim0) = pair;
          pair[0] = static_cast<typename T::InputT>(value10);
          pair[1] = static_cast<typename T::InputT>(peer10);
          *reinterpret_cast<cutlass::Array<typename T::InputT, 2>*>(row0 + dim1) = pair;
        }
        if (row1 != nullptr) {
          pair[0] = static_cast<typename T::InputT>(value01);
          pair[1] = static_cast<typename T::InputT>(peer01);
          *reinterpret_cast<cutlass::Array<typename T::InputT, 2>*>(row1 + dim0) = pair;
          pair[0] = static_cast<typename T::InputT>(value11);
          pair[1] = static_cast<typename T::InputT>(peer11);
          *reinterpret_cast<cutlass::Array<typename T::InputT, 2>*>(row1 + dim1) = pair;
        }
      }
    }
  }
}

__forceinline__ __device__ void load_dsm_float2(uint32_t remote_base, int element_idx,
                                                float& value0, float& value1) {
  const uint32_t address = remote_base + element_idx * sizeof(float);
  asm volatile("ld.shared::cluster.v2.f32 {%0, %1}, [%2];"
               : "=f"(value0), "=f"(value1)
               : "r"(address)
               : "memory");
}

__forceinline__ __device__ float load_dsm_float(uint32_t remote_base, int element_idx) {
  float value;
  const uint32_t address = remote_base + element_idx * sizeof(float);
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(value) : "r"(address));
  return value;
}

__forceinline__ __device__ int row_direct_offset(int slice, int row_bit, int idx, int warp,
                                                 int lane) {
  const int row = (lane % 4) * 2 + row_bit;
  const int row_lane = warp * 8 + lane / 4;
  return slice * 512 + row * 64 + row_lane * 2;
}

template <typename T, typename TensorScore, typename TensorSScore>
__forceinline__ __device__ void store_score_fragment(TensorScore const& score,
                                                     TensorSScore const& sScore, int warp_in_group,
                                                     int lane) {
  const int m0 = fragment_m0(warp_in_group, lane);
  const int m1 = m0 + 8;
#pragma unroll
  for (int group = 0; group < T::BLOCK_M / 8; ++group) {
    const int n0 = group * 8 + (lane % 4) * 2;
    const int n1 = n0 + 1;
    const int base = group * 4;
    sScore(n0, m0) = score(base + 0);
    sScore(n1, m0) = score(base + 1);
    sScore(n0, m1) = score(base + 2);
    sScore(n1, m1) = score(base + 3);
  }
}

template <typename Plan>
__forceinline__ __device__ float* p1_mailbox(Plan& plan) {
  return reinterpret_cast<float*>(plan.q.data());
}
