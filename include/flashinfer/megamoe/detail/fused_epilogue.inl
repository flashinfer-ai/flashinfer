// Shared NVFP4/MXFP4 fused SwiGLU, FP8 quantization and BF16 scatter.
if constexpr (!kBlockIsL2) {
  const float l1_global_scale =
      l1_global_scales == nullptr ? 1.0f : __ldg(l1_global_scales + local_expert_idx);
  if constexpr (kSwapABRequested) {
    auto silu = [](float x) -> float {
      const float e = kFastMath ? __expf(-x) : expf(-x);
      const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
      return x * sig;
    };
    auto clamp_gate = [](float& x) {
      if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
        x = cute::min(x, kActivationClamp);
    };
    auto clamp_up = [](float& x) {
      if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
        x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
    };

    constexpr uint32_t reduce_warp_start = 0;
    constexpr uint32_t reduce_warp_count = kNumEpilogueWarps;
    const uint32_t scale_token_thread = epilogue_thread_idx;
    constexpr uint32_t scale_token_stride = kNumEpilogueThreads;
    const uint32_t sf_base_k_idx = n_block_idx * L1_OUT_BLOCK_N / kL2ActsSFGranK;
    float swap_v0[kSwapABWeightHalves][kSwapABTokenChunks] = {};
    float swap_v1[kSwapABWeightHalves][kSwapABTokenChunks] = {};

    auto store_l1_swap_chunk = [&](const uint32_t& i) {
      const uint32_t token_0 = i * 8 + col_idx * 2;
      const uint32_t token_1 = token_0 + 1;

      float v0_amax = 0.0f;
      float v1_amax = 0.0f;
#pragma unroll
      for (uint32_t half = 0; half < kSwapABWeightHalves; ++half) {
        const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
        float v0 = 0.0f;
        if (token_0 < valid_m) {
          float g0 = final_accum[accum_offset + 0] * l1_global_scale;
          float u0 = final_accum[accum_offset + 2] * l1_global_scale;
          clamp_gate(g0);
          clamp_up(u0);
          const float weight_0 = *l1_topk_weights_buffer.get_data_buffer(m_idx + token_0)
                                      .template get_base_ptr<float>();
          v0 = silu(g0) * u0 * weight_0;
          swap_v0[half][i] = v0;
          v0_amax = cute::max(v0_amax, cute::abs(v0));
        }

        float v1 = 0.0f;
        if (token_1 < valid_m) {
          float g1 = final_accum[accum_offset + 1] * l1_global_scale;
          float u1 = final_accum[accum_offset + 3] * l1_global_scale;
          clamp_gate(g1);
          clamp_up(u1);
          const float weight_1 = *l1_topk_weights_buffer.get_data_buffer(m_idx + token_1)
                                      .template get_base_ptr<float>();
          v1 = silu(g1) * u1 * weight_1;
          swap_v1[half][i] = v1;
          v1_amax = cute::max(v1_amax, cute::abs(v1));
        }
      }

      const float amax0 = math::warp_reduce<4, true>(v0_amax, math::ReduceMax<float>());
      const float amax1 = math::warp_reduce<4, true>(v1_amax, math::ReduceMax<float>());
      if (row_idx == 0) {
        if (token_0 < valid_m)
          smem_cd_l1_shared_sf[token_0 * kNumEpilogueWarps + epilogue_warp_idx] = amax0;
        if (token_1 < valid_m)
          smem_cd_l1_shared_sf[token_1 * kNumEpilogueWarps + epilogue_warp_idx] = amax1;
      }
    };

    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
    store_l1_swap_chunk(0);
    if (valid_m > 8) {
#pragma unroll
      for (uint32_t i = 1; i < kSwapABTokenChunks; ++i) {
        if (i < num_swap_token_chunks) store_l1_swap_chunk(i);
      }
    }

    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

    for (uint32_t token = scale_token_thread; token < valid_m; token += scale_token_stride) {
      float amax = 0.0f;
#pragma unroll
      for (uint32_t w = 0; w < reduce_warp_count; ++w)
        amax = cute::max(amax,
                         smem_cd_l1_shared_sf[token * kNumEpilogueWarps + reduce_warp_start + w]);
      float2 amax_pair = {amax, amax};
      float2 sf_pair, sf_inv_pair;
      math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);

      auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
      const uint32_t token_idx = m_idx + token;
      sf_base_ptr[sf_base_k_idx * kNumPaddedSFPoolTokens + token_idx] = sf_pair.x;
      smem_cd_l1_shared_sf[token * kNumEpilogueWarps + reduce_warp_start] = sf_inv_pair.x;
    }

    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

#pragma unroll
    for (uint32_t i = 0; i < kSwapABTokenChunks; ++i) {
      const uint32_t token_0 = i * 8 + col_idx * 2;
      const uint32_t token_1 = token_0 + 1;
#pragma unroll
      for (uint32_t half = 0; half < kSwapABWeightHalves; ++half) {
        const uint32_t out_col_base = wg_l1_out_n_idx + half * 32u + warp_idx_in_wg * 8 + row_idx;
        if (token_0 < valid_m) {
          const float sf_inv =
              smem_cd_l1_shared_sf[token_0 * kNumEpilogueWarps + reduce_warp_start];
          const __nv_fp8_e4m3 q(swap_v0[half][i] * sf_inv);
          reinterpret_cast<uint8_t*>(smem_cd_l1)[token_0 * L1_OUT_BLOCK_N + out_col_base] =
              *reinterpret_cast<const uint8_t*>(&q);
        }
        if (token_1 < valid_m) {
          const float sf_inv =
              smem_cd_l1_shared_sf[token_1 * kNumEpilogueWarps + reduce_warp_start];
          const __nv_fp8_e4m3 q(swap_v1[half][i] * sf_inv);
          reinterpret_cast<uint8_t*>(smem_cd_l1)[token_1 * L1_OUT_BLOCK_N + out_col_base] =
              *reinterpret_cast<const uint8_t*>(&q);
        }
      }
    }
    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
    if (epilogue_wg_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
      const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
      cute::tma_store_fence();
      cute::SM90_TMA_STORE_2D::copy(&tensor_map_l1_output, smem_cd_l1, out_n_idx, m_idx);
      cute::tma_store_arrive();
    }
    __syncwarp();
    ptx::tma_store_wait<0>();
    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
    notify_l1_ready(pool_block_idx, n_block_idx);
  } else {
    // ---------------- L1 EPILOGUE: SwiGLU + FP8 quantize + TMA store ----------------
    const bool valid_r0 = row_offset_r0 < valid_m;
    const bool valid_r1 = row_offset_r1 < valid_m;
    // Layout in `final_accum`:
    //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
    //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
    //   Pair `p` ∈ [0, 8): gate chunk = 2p, up chunk = 2p+1.
    //
    // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
    // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

    constexpr uint32_t kNumPairs = kAccumPerThread / 8;
    constexpr uint32_t kNumSFGroups = 1;
    float swiglu_r0[kNumPairs][2];
    float swiglu_r1[kNumPairs][2];

    // Per-row amax, one scale for each 64-col L1 output group.
    float amax_r0[kNumSFGroups] = {};
    float amax_r1[kNumSFGroups] = {};

// Compute SwiGLU + per-group amax.
#pragma unroll
    for (uint32_t p = 0; p < kNumPairs; ++p) {
      const uint32_t gate = 2 * p, up = 2 * p + 1;
      const uint32_t sf_group = p / 8;

      auto clamp_gate = [](float& x) {
        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
          x = cute::min(x, kActivationClamp);
      };
      auto clamp_up = [](float& x) {
        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
          x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
      };
      float g_r0_c0 = final_accum[gate * 4 + 0] * l1_global_scale;
      clamp_gate(g_r0_c0);
      float g_r0_c1 = final_accum[gate * 4 + 1] * l1_global_scale;
      clamp_gate(g_r0_c1);
      float g_r1_c0 = final_accum[gate * 4 + 2] * l1_global_scale;
      clamp_gate(g_r1_c0);
      float g_r1_c1 = final_accum[gate * 4 + 3] * l1_global_scale;
      clamp_gate(g_r1_c1);
      float u_r0_c0 = final_accum[up * 4 + 0] * l1_global_scale;
      clamp_up(u_r0_c0);
      float u_r0_c1 = final_accum[up * 4 + 1] * l1_global_scale;
      clamp_up(u_r0_c1);
      float u_r1_c0 = final_accum[up * 4 + 2] * l1_global_scale;
      clamp_up(u_r1_c0);
      float u_r1_c1 = final_accum[up * 4 + 3] * l1_global_scale;
      clamp_up(u_r1_c1);

      auto silu = [](float x) -> float {
        const float e = kFastMath ? __expf(-x) : expf(-x);
        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
        return x * sig;
      };

      if (valid_r0) {
        swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
        swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
        amax_r0[sf_group] = cute::max(
            amax_r0[sf_group], cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
      } else {
        swiglu_r0[p][0] = 0.0f;
        swiglu_r0[p][1] = 0.0f;
      }
      if (valid_r1) {
        swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
        swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
        amax_r1[sf_group] = cute::max(
            amax_r1[sf_group], cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
      } else {
        swiglu_r1[p][0] = 0.0f;
        swiglu_r1[p][1] = 0.0f;
      }
    }

    const float weight_r0 = valid_r0
                                ? *l1_topk_weights_buffer.get_data_buffer(m_idx + row_offset_r0)
                                       .template get_base_ptr<float>()
                                : 0.0f;
    const float weight_r1 = valid_r1
                                ? *l1_topk_weights_buffer.get_data_buffer(m_idx + row_offset_r1)
                                       .template get_base_ptr<float>()
                                : 0.0f;
#pragma unroll
    for (uint32_t p = 0; p < kNumPairs; ++p) {
      swiglu_r0[p][0] *= weight_r0;
      swiglu_r0[p][1] *= weight_r0;
      swiglu_r1[p][0] *= weight_r1;
      swiglu_r1[p][1] *= weight_r1;
    }
#pragma unroll
    for (uint32_t g = 0; g < kNumSFGroups; ++g) {
      amax_r0[g] *= cute::abs(weight_r0);
      amax_r1[g] *= cute::abs(weight_r1);
    }
#pragma unroll
    for (uint32_t g = 0; g < kNumSFGroups; ++g) {
      amax_r0[g] = math::warp_reduce<4, false>(amax_r0[g], math::ReduceMax<float>());
      amax_r1[g] = math::warp_reduce<4, false>(amax_r1[g], math::ReduceMax<float>());
    }

    if (col_idx == 0) {
      smem_cd_l1_shared_sf[epilogue_wg_idx * BLOCK_M + row_offset_r0] = amax_r0[0];
      smem_cd_l1_shared_sf[epilogue_wg_idx * BLOCK_M + row_offset_r1] = amax_r1[0];
    }
    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
    if constexpr (kSplitMDecodedWeightReuse) {
      amax_r0[0] = smem_cd_l1_shared_sf[epilogue_wg_idx * BLOCK_M + row_offset_r0];
      amax_r1[0] = smem_cd_l1_shared_sf[epilogue_wg_idx * BLOCK_M + row_offset_r1];
    } else {
      amax_r0[0] = cute::max(smem_cd_l1_shared_sf[row_offset_r0],
                             smem_cd_l1_shared_sf[BLOCK_M + row_offset_r0]);
      amax_r1[0] = cute::max(smem_cd_l1_shared_sf[row_offset_r1],
                             smem_cd_l1_shared_sf[BLOCK_M + row_offset_r1]);
    }

    float sf_r0[kNumSFGroups], sf_inv_r0[kNumSFGroups];
    float sf_r1[kNumSFGroups], sf_inv_r1[kNumSFGroups];
#pragma unroll
    for (uint32_t g = 0; g < kNumSFGroups; ++g) {
      float2 amax_pair = {amax_r0[g], amax_r1[g]};
      float2 sf_pair, sf_inv_pair;
      math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
      sf_r0[g] = sf_pair.x;
      sf_inv_r0[g] = sf_inv_pair.x;
      sf_r1[g] = sf_pair.y;
      sf_inv_r1[g] = sf_inv_pair.y;
    }

// Quantize and write to smem_cd_l1 (row-major, no swizzle).
#pragma unroll
    for (uint32_t p = 0; p < kNumPairs; ++p) {
      const uint32_t sf_group = p / 8;
      const float v00 = swiglu_r0[p][0] * sf_inv_r0[sf_group];
      const float v01 = swiglu_r0[p][1] * sf_inv_r0[sf_group];
      const float v10 = swiglu_r1[p][0] * sf_inv_r1[sf_group];
      const float v11 = swiglu_r1[p][1] * sf_inv_r1[sf_group];

      const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
      const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));

      const uint32_t col = p * 8 + col_idx * 2;
      auto* p0 = reinterpret_cast<uint16_t*>(smem_cd_l1 + row_offset_r0 * L1_OUT_BLOCK_N +
                                             wg_l1_out_n_idx + col);
      auto* p1 = reinterpret_cast<uint16_t*>(smem_cd_l1 + row_offset_r1 * L1_OUT_BLOCK_N +
                                             wg_l1_out_n_idx + col);
      if (valid_r0) *p0 = r0_pair.__x;
      if (valid_r1) *p1 = r1_pair.__x;
    }

    // Write one physical L2-activation scale per 128 output columns.
    if (col_idx == 0) {
      auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
      const uint32_t token_r0 = m_idx + row_offset_r0;
      const uint32_t token_r1 = m_idx + row_offset_r1;
      const uint32_t base_k_sf_idx =
          (n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx) / kL2ActsSFGranK;
#pragma unroll
      for (uint32_t g = 0; g < kNumSFGroups; ++g) {
        const uint32_t sf_k_idx = base_k_sf_idx + g;
        if ((kSplitMDecodedWeightReuse || epilogue_wg_idx == 0) && valid_r0)
          sf_base_ptr[sf_k_idx * kNumPaddedSFPoolTokens + token_r0] = sf_r0[g];
        if ((kSplitMDecodedWeightReuse || epilogue_wg_idx == 0) && valid_r1)
          sf_base_ptr[sf_k_idx * kNumPaddedSFPoolTokens + token_r1] = sf_r1[g];
      }
    }

    // Issue TMA store of the entire tile. Padding rows beyond
    // `valid_m` are written with stale/garbage FP8 to the L1-output
    // pool buffer, but they are never consumed downstream: the L2
    // GEMM tile loads them, but its NVLink-scatter epilogue is
    // gated by `m_idx_in_block >= valid_m`, and stale SF in the
    // padding rows can produce NaN accumulators that simply stay
    // in registers (only valid rows are converted to BF16 and
    // STSM'd into smem). Using TMA for partial tiles is a large
    // win for low-batch / decode where every tile is partial.
    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
      const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
      cute::tma_store_fence();
      cute::SM90_TMA_STORE_2D::copy(&tensor_map_l1_output, smem_cd_l1, out_n_idx, m_idx);
      cute::tma_store_arrive();
    }
    __syncwarp();
    ptx::tma_store_wait<0>();
    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
    notify_l1_ready(pool_block_idx, n_block_idx);
  }
} else {
  // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------
  if constexpr (kSwapABRequested) {
    // Each active warp scatters a contiguous group of up to 16 rows.
    constexpr uint32_t kNumRowsPerWarp = BLOCK_M == 8 ? 4u : 8u;
    auto store_swap_bf16 = [&](const uint32_t& token, const uint32_t& col, const float& value) {
      if (token < valid_m)
        smem_cd_l2[token * BLOCK_N + wg_n_idx + col] = __float2bfloat16_rn(value * l2_global_scale);
    };

    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
    auto store_l2_swap_chunk = [&](const uint32_t& i) {
      const uint32_t token_0 = i * 8 + col_idx * 2;
      const uint32_t token_1 = token_0 + 1;
#pragma unroll
      for (uint32_t half = 0; half < kSwapABWeightHalves; ++half) {
        const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
        const uint32_t col_offset = half * 64u;
        store_swap_bf16(token_0, col_offset + r_0, final_accum[accum_offset + 0]);
        store_swap_bf16(token_0, col_offset + r_1, final_accum[accum_offset + 2]);
        store_swap_bf16(token_1, col_offset + r_0, final_accum[accum_offset + 1]);
        store_swap_bf16(token_1, col_offset + r_1, final_accum[accum_offset + 3]);
      }
    };

    store_l2_swap_chunk(0);
    if (valid_m > 8) {
#pragma unroll
      for (uint32_t i = 1; i < kSwapABTokenChunks; ++i) {
        if (i < num_swap_token_chunks) store_l2_swap_chunk(i);
      }
    }

    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

    const uint32_t row_in_warp_block = lane_idx / 16;
    const uint32_t lane_in_row = lane_idx % 16;
    constexpr uint32_t kColsPerScatterLane = WG_BLOCK_N / 16;
    FI_MEGAMOE_STATIC_ASSERT(WG_BLOCK_N % 16 == 0,
                             "SwapAB L2 scatter expects an even lane partition");
    FI_MEGAMOE_STATIC_ASSERT(kColsPerScatterLane == 4 || kColsPerScatterLane == 8,
                             "SwapAB L2 scatter supports WG_BLOCK_N=64 or 128");

#pragma unroll
    for (uint32_t j = 0; j < kNumRowsPerWarp; ++j) {
      const uint32_t token = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
      if (token >= valid_m) break;

      const auto src_metadata =
          *workspace.get_token_src_metadata_ptr(pool_block_idx * BLOCK_M + token);
      const uint32_t dst_rank_idx = src_metadata.rank_idx;
      const uint32_t dst_token_idx = src_metadata.token_idx;
      const uint32_t dst_topk_idx = src_metadata.topk_idx;
      const auto dst_token =
          combine_token_buffer.get_rank_buffer(dst_topk_idx).get_data_buffer(dst_token_idx);
      auto smem_ptr = smem_cd_l2 + token * BLOCK_N + wg_n_idx + lane_in_row * kColsPerScatterLane;
      if constexpr (kColsPerScatterLane == 8) {
        const auto packed = *reinterpret_cast<uint4*>(smem_ptr);
        auto dst_ptr = math::advance_ptr<uint4>(
            dst_token.get_base_ptr(), n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
        *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
      } else {
        const auto packed = *reinterpret_cast<uint2*>(smem_ptr);
        auto dst_ptr = math::advance_ptr<uint2>(
            dst_token.get_base_ptr(), n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint2));
        *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
      }
    }
    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
  } else {
    FI_MEGAMOE_STATIC_ASSERT(WG_BLOCK_N == 64 || WG_BLOCK_N == 128,
                             "Direct L2 scatter requires N64/N128");
    const bool valid_r0 = row_offset_r0 < valid_m;
    const bool valid_r1 = row_offset_r1 < valid_m;

    auto scatter_direct_row = [&](const uint32_t& row_offset, const bool& valid_row,
                                  const uint32_t& row_accum_offset) {
      if (valid_row) {
        const auto src_metadata =
            *workspace.get_token_src_metadata_ptr(pool_block_idx * BLOCK_M + row_offset);
        const uint32_t dst_rank_idx = src_metadata.rank_idx;
        const uint32_t dst_token_idx = src_metadata.token_idx;
        const uint32_t dst_topk_idx = src_metadata.topk_idx;
        const auto dst_token =
            combine_token_buffer.get_rank_buffer(dst_topk_idx).get_data_buffer(dst_token_idx);
        auto dst_base =
            math::advance_ptr<uint8_t>(dst_token.get_base_ptr(), n_idx * sizeof(nv_bfloat16));
        auto mapped_dst_base = sym_buffer.map(dst_base, dst_rank_idx);

#pragma unroll
        for (uint32_t i = 0; i < kAccumPerThread / 8; ++i) {
          const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
          const uint32_t col_lo = chunk_lo * 8 + col_idx * 2;
          const uint32_t col_hi = chunk_hi * 8 + col_idx * 2;
          const uint32_t packed_lo =
              cast_l2_scaled_bf16_pair(final_accum[chunk_lo * 4 + row_accum_offset + 0],
                                       final_accum[chunk_lo * 4 + row_accum_offset + 1]);
          const uint32_t packed_hi =
              cast_l2_scaled_bf16_pair(final_accum[chunk_hi * 4 + row_accum_offset + 0],
                                       final_accum[chunk_hi * 4 + row_accum_offset + 1]);
          *reinterpret_cast<uint32_t*>(mapped_dst_base + col_lo * sizeof(nv_bfloat16)) = packed_lo;
          *reinterpret_cast<uint32_t*>(mapped_dst_base + col_hi * sizeof(nv_bfloat16)) = packed_hi;
        }
      }
    };

    scatter_direct_row(row_offset_r0, valid_r0, 0);
    scatter_direct_row(row_offset_r1, valid_r1, 2);
    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
  }
}
