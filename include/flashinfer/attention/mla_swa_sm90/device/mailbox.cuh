#pragma once

template <bool ProduceWeights, int MaxSlices, typename T, typename TensorOutput, typename Plan>
__forceinline__ __device__ void dsm_row_direct_m8_s3_slices(
    TensorOutput const& outputs, int output_slice_offset, int slice_begin, int slice_end,
    Plan& plan, typename T::InputT* out, int batch, int m_block, int total_rows, int heads,
    int64_t stride_b, int64_t stride_s, int64_t stride_h, int idx, int warp, int lane) {
  using InputT = typename T::InputT;
  static_assert(T::BLOCK_M == 8 && T::TILE_K == 64);
  static_assert(3 == 3);
  static_assert(1 == 1);
  constexpr int kElements = 8 * 1024;
  const int rank = cute::block_id_in_cluster().x;
  const int row_begin = 8 * rank / 3;
  const int row_end = 8 * (rank + 1) / 3;
  const int peer0 = (rank + 1) % 3;
  const int peer1 = (rank + 2) % 3;
  const int n0 = (lane % 4) * 2;
  const int n1 = n0 + 1;
  const float inv0 = 1.0f;
  const float inv1 = 1.0f;
  float* mailbox = reinterpret_cast<float*>(plan.kv.data());
  float* lse_local = mailbox + kElements;
#pragma unroll
  for (int local = 0; local < MaxSlices; ++local) {
    const int slice = slice_begin + local;
    if (!T::Config::DenseRowMailbox && slice >= slice_end) continue;
    Tensor flat = outputs(_, output_slice_offset + local, _0{});
    Tensor acc = make_tensor(flat.data(), typename T::OutputSliceFragmentLayout{});
    if (n0 < row_begin || n0 >= row_end) {
      const int offset = row_direct_offset(slice, 0, idx, warp, lane);
      *reinterpret_cast<float2*>(mailbox + offset) = make_float2(acc(0) * inv0, acc(2) * inv0);
    }
    if (n1 < row_begin || n1 >= row_end) {
      const int offset = row_direct_offset(slice, 1, idx, warp, lane);
      *reinterpret_cast<float2*>(mailbox + offset) = make_float2(acc(1) * inv1, acc(3) * inv1);
    }
  }
  if constexpr (ProduceWeights) {
    if (idx < 8) {
      const float sum = plan.row_sum[idx];
      lse_local[idx] = plan.row_max[idx];
      lse_local[8 + idx] = sum;
    }
  }
  cute::cluster_sync();
  if constexpr (ProduceWeights) {
    if (idx < 8) {
      const uint32_t lse_base = cute::cast_smem_ptr_to_uint(lse_local);
      float maximum = -CUDART_INF_F;
      float score_max[3], row_sum[3], weight[3];
#pragma unroll
      for (int source = 0; source < 3; ++source) {
        const uint32_t address = cute::set_block_rank(lse_base, source);
        score_max[source] = source == rank ? lse_local[idx] : load_dsm_float(address, idx);
        row_sum[source] = source == rank ? lse_local[8 + idx] : load_dsm_float(address, 8 + idx);
        maximum = fmaxf(maximum, score_max[source]);
      }
      float denominator = 0.0f;
#pragma unroll
      for (int source = 0; source < 3; ++source) {
        weight[source] = raw_exp_weight(score_max[source], row_sum[source], maximum);
        denominator += weight[source] * row_sum[source];
      }
      const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
#pragma unroll
      for (int source = 0; source < 3; ++source)
        plan.reduction[source * 8 + idx] = weight[source] * inv;
    }
  }
  role_barrier();
  const uint32_t base = cute::cast_smem_ptr_to_uint(mailbox);
  const uint32_t remote0 = cute::set_block_rank(base, peer0);
  const uint32_t remote1 = cute::set_block_rank(base, peer1);
  const int dim0 = fragment_m0(warp, lane);
#pragma unroll
  for (int row_bit = 0; row_bit < 2; ++row_bit) {
    const int row = n0 + row_bit;
    const bool owned = row >= row_begin && row < row_end;
    const float inv = row_bit == 0 ? inv0 : inv1;
    const float w_local = plan.reduction[rank * 8 + row];
    const float w_peer0 = plan.reduction[peer0 * 8 + row];
    const float w_peer1 = plan.reduction[peer1 * 8 + row];
    const int global_row = m_block * 8 + row;
    InputT* output_row =
        out + batch * stride_b + (global_row / heads) * stride_s + (global_row % heads) * stride_h;
    float ahead00 = 0.0f, ahead01 = 0.0f, ahead10 = 0.0f, ahead11 = 0.0f;
    if (owned && (T::Config::DenseRowMailbox || slice_begin < slice_end)) {
      const int offset = row_direct_offset(slice_begin, row_bit, idx, warp, lane);
      load_dsm_float2(remote0, offset, ahead00, ahead01);
      load_dsm_float2(remote1, offset, ahead10, ahead11);
    }
#pragma unroll
    for (int local = 0; local < MaxSlices; ++local) {
      const int slice = slice_begin + local;
      if (!T::Config::DenseRowMailbox && slice >= slice_end) continue;
      Tensor flat = outputs(_, output_slice_offset + local, _0{});
      Tensor acc = make_tensor(flat.data(), typename T::OutputSliceFragmentLayout{});
      float value0 = 0.0f, value1 = 0.0f;
      if (owned) {
        const int offset = row_direct_offset(slice, row_bit, idx, warp, lane);
        float p00, p01, p10, p11;
        p00 = ahead00;
        p01 = ahead01;
        p10 = ahead10;
        p11 = ahead11;
        if (local + 1 < MaxSlices && (T::Config::DenseRowMailbox || slice + 1 < slice_end)) {
          load_dsm_float2(remote0, offset + 512, ahead00, ahead01);
          load_dsm_float2(remote1, offset + 512, ahead10, ahead11);
        }
        value0 = fmaf(w_peer1, p10, fmaf(w_peer0, p00, w_local * (acc(row_bit) * inv)));
        value1 = fmaf(w_peer1, p11, fmaf(w_peer0, p01, w_local * (acc(row_bit + 2) * inv)));
      }
      const float adjacent0 = __shfl_down_sync(0xffffffff, value0, 4);
      const float adjacent1 = __shfl_down_sync(0xffffffff, value1, 4);
      if (owned && global_row < total_rows && (lane & 4) == 0) {
        const int dim = slice * 64 + dim0;
        cutlass::Array<InputT, 2> pair;
        pair[0] = InputT(value0);
        pair[1] = InputT(adjacent0);
        *reinterpret_cast<cutlass::Array<InputT, 2>*>(output_row + dim) = pair;
        pair[0] = InputT(value1);
        pair[1] = InputT(adjacent1);
        *reinterpret_cast<cutlass::Array<InputT, 2>*>(output_row + dim + 8) = pair;
      }
    }
  }
  cute::cluster_sync();
}

template <bool ProduceWeights, int MaxSlices, typename T, typename TensorOutput, typename Plan>
__forceinline__ __device__ void dsm_direct_accum_m8_s3_slices(
    TensorOutput const& output_fragment_all, int output_slice_offset, int slice_begin,
    int slice_end, Plan& plan, typename T::InputT* out, int batch_idx, int m_block_idx,
    int total_rows, int num_heads, int64_t out_stride_b, int64_t out_stride_s, int64_t out_stride_h,
    int idx_in_warpgroup, int warp_in_group, int lane) {
  dsm_row_direct_m8_s3_slices<ProduceWeights, MaxSlices, T>(
      output_fragment_all, output_slice_offset, slice_begin, slice_end, plan, out, batch_idx,
      m_block_idx, total_rows, num_heads, out_stride_b, out_stride_s, out_stride_h,
      idx_in_warpgroup, warp_in_group, lane);
}

template <typename T, typename Plan>
__forceinline__ __device__ void dsm_direct_accum_m32_qk_s2(Plan& plan, int idx,
                                                           typename T::InputT* out, int batch,
                                                           int m_block, int total_rows, int heads,
                                                           int64_t stride_b, int64_t stride_s,
                                                           int64_t stride_h) {
  static_assert(T::BLOCK_M == 32 && T::NUM_PV_WARPGROUPS == 3);
  static_assert(2 == 2 && 1);
  constexpr int words = 16 * 2 * 128 * 4;
  cute::cluster_wait();
  static_assert((words + 32) * sizeof(float) <=
                T::BLOCK_M * T::HEAD_DIM_K * sizeof(typename T::InputT));
  float* state = p1_mailbox(plan) + words;
  const int rank = cute::block_rank_in_cluster();
  const int owner = rank * 16;
  const int publish = (1 - rank) * 16;
  if (idx < 16) {
    const float sum = plan.row_sum[publish + idx];
    state[idx] = plan.row_max[publish + idx];
    state[16 + idx] = sum;
  }
  p1_wg_sync(0);
  if (idx == 0) p1_release(plan.p1_ready[0], rank ^ 1);
  p1_acquire(plan.p1_ready[0]);
  if (idx < 16) {
    const int row = owner + idx;
    const uint32_t remote = cute::set_block_rank(cute::cast_smem_ptr_to_uint(state), rank ^ 1);
    const float local_sum = plan.row_sum[row];
    const float local_max = plan.row_max[row];
    const float peer_max = load_dsm_float(remote, idx);
    const float peer_sum = load_dsm_float(remote, 16 + idx);
    const float2 weights = raw_pair_weights(local_max, local_sum, peer_max, peer_sum);
    plan.row_max[row] = weights.x;
    plan.row_sum[row] = weights.y;
  }
  p1_wg_sync(0);
  if (idx == 0) p1_release(plan.p1_ack[0], rank ^ 1);
  role_barrier();
  p1_acquire(plan.p1_ack[0]);
  role_barrier();
}

template <int OwnerGroup, int PublishGroup, int MaxSlices, typename T, typename TensorOutput,
          typename Plan>
__forceinline__ __device__ void dsm_direct_accum_pv_s2(
    TensorOutput const& output_fragment_all, int slice_begin, int slice_end, int slice_stride,
    int pv_group, Plan& plan, typename T::InputT* out, int batch_idx, int m_block_idx,
    int total_rows, int num_heads, int64_t out_stride_b, int64_t out_stride_s, int64_t out_stride_h,
    int idx_in_warpgroup, int warp_in_group, int lane) {
  using InputT = typename T::InputT;
  using Layout = DsmDirectAccumLayout<T>;
  static_assert(OwnerGroup == 0 || OwnerGroup == 1);
  static_assert(PublishGroup == 1 - OwnerGroup);
  static_assert(T::BLOCK_M == 16);
  static_assert(T::NUM_PV_WARPGROUPS == Layout::kPvGroups);
  static_assert(MaxSlices == Layout::kSlicesPerPvGroup);
  static_assert(1);
  static_assert(0 == 0);
  constexpr bool Raw = (T::Config::PartialFormat == mla_swa::PartialForm::Raw);
  uint32_t* mailbox = reinterpret_cast<uint32_t*>(plan.q.data());
  float* mailbox_fp32 = reinterpret_cast<float*>(mailbox);
  constexpr int kPayloadBytes = Layout::kMailboxWords * sizeof(uint32_t);
  constexpr int kStateBytes = Layout::kStateFloats * sizeof(float);
  constexpr int kQBytes = T::BLOCK_M * T::HEAD_DIM_K * sizeof(InputT);
  static_assert(kPayloadBytes + kStateBytes <= kQBytes);
  cutlass::Array<float, MaxSlices * 4> owner_values;
  const int m0 = fragment_m0(warp_in_group, lane);
  const int m1 = m0 + 8;
  const int owner_n0 = OwnerGroup * 8 + (lane % 4) * 2;
  const int owner_n1 = owner_n0 + 1;
  const int publish_n0 = PublishGroup * 8 + (lane % 4) * 2;
  const int publish_n1 = publish_n0 + 1;
  const float owner_inv_sum0 =
      Raw ? 1.0f : (plan.row_sum[owner_n0] > 0.0f ? 1.0f / plan.row_sum[owner_n0] : 0.0f);
  const float owner_inv_sum1 =
      Raw ? 1.0f : (plan.row_sum[owner_n1] > 0.0f ? 1.0f / plan.row_sum[owner_n1] : 0.0f);
  const float publish_inv_sum0 =
      Raw ? 1.0f : (plan.row_sum[publish_n0] > 0.0f ? 1.0f / plan.row_sum[publish_n0] : 0.0f);
  const float publish_inv_sum1 =
      Raw ? 1.0f : (plan.row_sum[publish_n1] > 0.0f ? 1.0f / plan.row_sum[publish_n1] : 0.0f);
#pragma unroll
  for (int local_slice = 0; local_slice < MaxSlices; ++local_slice) {
    const int global_slice = slice_begin + local_slice * slice_stride;
    if (global_slice >= slice_end) {
      continue;
    }
    Tensor output_fragment_flat = output_fragment_all(_, local_slice, _0{});
    Tensor output_fragment =
        make_tensor(output_fragment_flat.data(), typename T::OutputSliceFragmentLayout{});
    constexpr int kOwnerBase = OwnerGroup * 4;
    constexpr int kPublishBase = PublishGroup * 4;
    const float owner00 = output_fragment(kOwnerBase + 0) * owner_inv_sum0;
    const float owner01 = output_fragment(kOwnerBase + 1) * owner_inv_sum1;
    const float owner10 = output_fragment(kOwnerBase + 2) * owner_inv_sum0;
    const float owner11 = output_fragment(kOwnerBase + 3) * owner_inv_sum1;
    owner_values[local_slice * 4 + 0] = owner00;
    owner_values[local_slice * 4 + 1] = owner01;
    owner_values[local_slice * 4 + 2] = owner10;
    owner_values[local_slice * 4 + 3] = owner11;
    const float publish00 = output_fragment(kPublishBase + 0) * publish_inv_sum0;
    const float publish01 = output_fragment(kPublishBase + 1) * publish_inv_sum1;
    const float publish10 = output_fragment(kPublishBase + 2) * publish_inv_sum0;
    const float publish11 = output_fragment(kPublishBase + 3) * publish_inv_sum1;
    const int offset = (global_slice * 128 + idx_in_warpgroup) * 4;
    fragment_store4(mailbox_fp32 + offset, publish00, publish01, publish10, publish11);
  }
  role_barrier();
  asm volatile("fence.acq_rel.cluster;" ::: "memory");
  cute::cluster_sync();
  asm volatile("fence.acq_rel.cluster;" ::: "memory");
  role_barrier();
  const int cluster_rank = cute::block_rank_in_cluster();
  const int peer_rank = cluster_rank ^ 1;
  const uint32_t mailbox_base = cute::cast_smem_ptr_to_uint(mailbox);
  const uint32_t mailbox_peer = cute::set_block_rank(mailbox_base, peer_rank);
  const float weight0_row0 = plan.row_max[owner_n0];
  const float weight1_row0 = plan.row_sum[owner_n0];
  const float weight0_row1 = plan.row_max[owner_n1];
  const float weight1_row1 = plan.row_sum[owner_n1];
  float ahead00, ahead01, ahead10, ahead11;
  const int first_offset = (slice_begin * 128 + idx_in_warpgroup) * 4;
  fragment_load4(mailbox_peer + first_offset * sizeof(float), ahead00, ahead01, ahead10, ahead11);
#pragma unroll
  for (int local_slice = 0; local_slice < MaxSlices; ++local_slice) {
    const int global_slice = slice_begin + local_slice * slice_stride;
    if (global_slice >= slice_end) continue;
    const float peer00 = ahead00, peer01 = ahead01;
    const float peer10 = ahead10, peer11 = ahead11;
    if (local_slice + 1 < MaxSlices && global_slice + slice_stride < slice_end) {
      const int next_offset = ((global_slice + slice_stride) * 128 + idx_in_warpgroup) * 4;
      fragment_load4(mailbox_peer + next_offset * sizeof(float), ahead00, ahead01, ahead10,
                     ahead11);
    }
    const float local00 = owner_values[local_slice * 4 + 0];
    const float local01 = owner_values[local_slice * 4 + 1];
    const float local10 = owner_values[local_slice * 4 + 2];
    const float local11 = owner_values[local_slice * 4 + 3];
    float output00;
    float output01;
    float output10;
    float output11;
    if constexpr (OwnerGroup == 0) {
      output00 = weight0_row0 * local00 + weight1_row0 * peer00;
      output01 = weight0_row1 * local01 + weight1_row1 * peer01;
      output10 = weight0_row0 * local10 + weight1_row0 * peer10;
      output11 = weight0_row1 * local11 + weight1_row1 * peer11;
    } else {
      output00 = weight0_row0 * peer00 + weight1_row0 * local00;
      output01 = weight0_row1 * peer01 + weight1_row1 * local01;
      output10 = weight0_row0 * peer10 + weight1_row0 * local10;
      output11 = weight0_row1 * peer11 + weight1_row1 * local11;
    }
    const int global_row0 = m_block_idx * T::BLOCK_M + owner_n0;
    const int global_row1 = global_row0 + 1;
    const int dim0 = global_slice * T::TILE_K + m0;
    const int dim1 = global_slice * T::TILE_K + m1;
    if (global_row0 < total_rows) {
      const int query_idx = global_row0 >> 6;
      const int head_idx = global_row0 & 63;
      InputT* output_row =
          out + batch_idx * out_stride_b + query_idx * out_stride_s + head_idx * out_stride_h;
      output_row[dim0] = InputT(output00);
      output_row[dim1] = InputT(output10);
    }
    if (global_row1 < total_rows) {
      const int query_idx = global_row1 >> 6;
      const int head_idx = global_row1 & 63;
      InputT* output_row =
          out + batch_idx * out_stride_b + query_idx * out_stride_s + head_idx * out_stride_h;
      output_row[dim0] = InputT(output01);
      output_row[dim1] = InputT(output11);
    }
  }
  cute::cluster_sync();
}

template <int MaxSlices, typename T, typename TensorOutput, typename Plan>
__forceinline__ __device__ void dsm_direct_accum_m32_pv_s2(TensorOutput& outputs, int begin,
                                                           int end, Plan& plan,
                                                           typename T::InputT* out, int batch,
                                                           int m_block, int total_rows, int heads,
                                                           int64_t stride_b, int64_t stride_s,
                                                           int64_t stride_h, int warp, int lane) {
  using InputT = typename T::InputT;
  cute::cluster_wait();
  const int rank = cute::block_rank_in_cluster();
  const int owner_group = rank * 2;
  const int publish_group = (1 - rank) * 2;
  const int idx = warp * 32 + lane;
  const int pv_wg = threadIdx.x / 128;
  float* mailbox = p1_mailbox(plan);
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    const int n0 = group * 8 + (lane % 4) * 2;
    const int n1 = n0 + 1;
#pragma unroll
    for (int local = 0; local < MaxSlices; ++local) {
      const int slice = begin + local;
      if (slice >= end) continue;
      Tensor flat = outputs(_, local, _0{});
      Tensor acc = make_tensor(flat.data(), typename T::OutputSliceFragmentLayout{});
      const int base = group * 4;
      if (group >= publish_group && group < publish_group + 2) {
        const int offset = ((slice * 2 + group - publish_group) * 128 + idx) * 4;
        fragment_store4(mailbox + offset, acc(base), acc(base + 1), acc(base + 2), acc(base + 3));
      }
    }
  }
  p1_wg_sync(pv_wg);
  if (idx == 0) p1_release(plan.p1_ready[pv_wg], rank ^ 1);
  role_barrier();
  p1_acquire(plan.p1_ready[pv_wg]);
  const uint32_t remote = cute::set_block_rank(cute::cast_smem_ptr_to_uint(mailbox), rank ^ 1);
  const int dim0 = begin * T::TILE_K + fragment_m0(warp, lane);
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    if (group < owner_group || group >= owner_group + 2) continue;
    const int n0 = group * 8 + (lane % 4) * 2;
    const int n1 = n0 + 1;
    const float a0 = plan.row_max[n0], b0 = plan.row_sum[n0];
    const float a1 = plan.row_max[n1], b1 = plan.row_sum[n1];
    const int global0 = m_block * 32 + n0;
    const int global1 = global0 + 1;
    InputT* row0 =
        out + batch * stride_b + (global0 / heads) * stride_s + (global0 % heads) * stride_h;
    InputT* row1 =
        out + batch * stride_b + (global1 / heads) * stride_s + (global1 % heads) * stride_h;
    float ahead0, ahead1, ahead2, ahead3;
    const int first_offset = ((begin * 2 + group - owner_group) * 128 + idx) * 4;
    fragment_load4(remote + first_offset * sizeof(float), ahead0, ahead1, ahead2, ahead3);
#pragma unroll
    for (int local = 0; local < MaxSlices; ++local) {
      const int slice = begin + local;
      if (slice >= end) continue;
      Tensor flat = outputs(_, local, _0{});
      Tensor acc = make_tensor(flat.data(), typename T::OutputSliceFragmentLayout{});
      const int base = group * 4;
      const int offset = ((slice * 2 + group - owner_group) * 128 + idx) * 4;
      float p0, p1, p2, p3;
      p0 = ahead0;
      p1 = ahead1;
      p2 = ahead2;
      p3 = ahead3;
      if (local + 1 < MaxSlices && slice + 1 < end && !(0 && slice + 1 == 15)) {
        fragment_load4(remote + (offset + 2 * 128 * 4) * sizeof(float), ahead0, ahead1, ahead2,
                       ahead3);
      }
      const float v0 = a0 * acc(base) + b0 * p0;
      const float v1 = a1 * acc(base + 1) + b1 * p1;
      const float v2 = a0 * acc(base + 2) + b0 * p2;
      const float v3 = a1 * acc(base + 3) + b1 * p3;
      const float next0 = __shfl_down_sync(0xffffffff, v0, 4);
      const float next1 = __shfl_down_sync(0xffffffff, v1, 4);
      const float next2 = __shfl_down_sync(0xffffffff, v2, 4);
      const float next3 = __shfl_down_sync(0xffffffff, v3, 4);
      if (group == owner_group + 1 && slice == end - 1 - (0 && pv_wg == 3)) {
        __syncwarp();
        if (lane == 0) p1_release(plan.p1_ack[pv_wg], rank ^ 1);
      }
      if ((lane & 4) == 0) {
        const int dim = dim0 + local * T::TILE_K;
        cutlass::Array<InputT, 2> pair;
        if (global0 < total_rows) {
          pair[0] = InputT(v0);
          pair[1] = InputT(next0);
          *reinterpret_cast<cutlass::Array<InputT, 2>*>(row0 + dim) = pair;
          pair[0] = InputT(v2);
          pair[1] = InputT(next2);
          *reinterpret_cast<cutlass::Array<InputT, 2>*>(row0 + dim + 8) = pair;
        }
        if (global1 < total_rows) {
          pair[0] = InputT(v1);
          pair[1] = InputT(next1);
          *reinterpret_cast<cutlass::Array<InputT, 2>*>(row1 + dim) = pair;
          pair[0] = InputT(v3);
          pair[1] = InputT(next3);
          *reinterpret_cast<cutlass::Array<InputT, 2>*>(row1 + dim + 8) = pair;
        }
      }
    }
  }
  p1_acquire(plan.p1_ack[pv_wg]);
  role_barrier();
}

template <int OwnerGroup, typename T, typename Plan>
__forceinline__ __device__ void dsm_direct_accum_qk_s2(Plan& plan, int idx_in_warpgroup) {
  using Layout = DsmDirectAccumLayout<T>;
  static_assert(OwnerGroup == 0 || OwnerGroup == 1);
  static_assert(T::BLOCK_M == 16);
  uint32_t* mailbox = reinterpret_cast<uint32_t*>(plan.q.data());
  float* state_mailbox = reinterpret_cast<float*>(mailbox + Layout::kMailboxWords);
  constexpr int kPayloadBytes = Layout::kMailboxWords * sizeof(uint32_t);
  constexpr int kStateBytes = Layout::kStateFloats * sizeof(float);
  constexpr int kQBytes = T::BLOCK_M * T::HEAD_DIM_K * sizeof(typename T::InputT);
  static_assert(kPayloadBytes + kStateBytes <= kQBytes);
  constexpr int kPublishGroup = 1 - OwnerGroup;
  if (idx_in_warpgroup < Layout::kRowsPerOwner) {
    const int publish_row = kPublishGroup * Layout::kRowsPerOwner + idx_in_warpgroup;
    const float sum = plan.row_sum[publish_row];
    if constexpr ((T::Config::PartialFormat == mla_swa::PartialForm::Raw)) {
      state_mailbox[idx_in_warpgroup] = plan.row_max[publish_row];
      state_mailbox[Layout::kRowsPerOwner + idx_in_warpgroup] = sum;
    } else {
      state_mailbox[idx_in_warpgroup] =
          sum > 0.0f ? plan.row_max[publish_row] + log2f(sum) : -CUDART_INF_F;
    }
  }
  role_barrier();
  asm volatile("fence.acq_rel.cluster;" ::: "memory");
  cute::cluster_sync();
  asm volatile("fence.acq_rel.cluster;" ::: "memory");
  if (idx_in_warpgroup < Layout::kRowsPerOwner) {
    const int owner_row = OwnerGroup * Layout::kRowsPerOwner + idx_in_warpgroup;
    const float local_sum = plan.row_sum[owner_row];
    if constexpr ((T::Config::PartialFormat == mla_swa::PartialForm::Raw)) {
      const int peer_rank = cute::block_rank_in_cluster() ^ 1;
      const uint32_t peer =
          cute::set_block_rank(cute::cast_smem_ptr_to_uint(state_mailbox), peer_rank);
      const float peer_max = load_dsm_float(peer, idx_in_warpgroup);
      const float peer_sum = load_dsm_float(peer, Layout::kRowsPerOwner + idx_in_warpgroup);
      const float local_max = plan.row_max[owner_row];
      const float2 weights = raw_pair_weights(local_max, local_sum, peer_max, peer_sum);
      plan.row_max[owner_row] = OwnerGroup == 0 ? weights.x : weights.y;
      plan.row_sum[owner_row] = OwnerGroup == 0 ? weights.y : weights.x;
    } else {
      const float local_lse =
          local_sum > 0.0f ? plan.row_max[owner_row] + log2f(local_sum) : -CUDART_INF_F;
      const int cluster_rank = cute::block_rank_in_cluster();
      const int peer_rank = cluster_rank ^ 1;
      const uint32_t state_base = cute::cast_smem_ptr_to_uint(state_mailbox);
      const uint32_t state_peer = cute::set_block_rank(state_base, peer_rank);
      const float peer_lse = load_dsm_float(state_peer, idx_in_warpgroup);
      const float lse0 = OwnerGroup == 0 ? local_lse : peer_lse;
      const float lse1 = OwnerGroup == 0 ? peer_lse : local_lse;
      const float row_max = fmaxf(lse0, lse1);
      const float unnormalized0 = isfinite(lse0) ? exp2f(lse0 - row_max) : 0.0f;
      const float unnormalized1 = isfinite(lse1) ? exp2f(lse1 - row_max) : 0.0f;
      const float weight_sum = unnormalized0 + unnormalized1;
      const float inv_weight_sum = weight_sum > 0.0f ? 1.0f / weight_sum : 0.0f;
      plan.row_max[owner_row] = unnormalized0 * inv_weight_sum;
      plan.row_sum[owner_row] = unnormalized1 * inv_weight_sum;
    }
  }
  role_barrier();
  cute::cluster_sync();
}
