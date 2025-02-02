#ifndef FLASHINFER_POD_CUH_
#define FLASHINFER_POD_CUH_

#include "prefill.cuh"
#include "decode.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "../cp_async.cuh"
#include "../fastdiv.cuh"
#include "../frag_layout_swizzle.cuh"
#include "../layout.cuh"
#include "../math.cuh"
#include "../mma.cuh"
#include "../page.cuh"
#include "../permuted_smem.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "cascade.cuh"
#include "mask.cuh"
#include "variants.cuh"

namespace flashinfer {

namespace cg = cooperative_groups;
using cp_async::SharedMemFillMode;
using mma::MMAMode;

enum Operation {
  PREFILL = 0,
  DECODE = 1,
};

/*!
 * \brief POD-Attention hybrid batch CUDA kernel for a single request.
 * \tparam partition_kv Whether to split kv_len into chunks.
 * \tparam mask_mode The mask mode used in the attention operation.
 * \tparam POS_ENCODING_MODE The positional encoding mode.
 * \tparam NUM_MMA_Q The number of fragments in x dimension.
 * \tparam NUM_MMA_D The number of fragments in y dimension.
 * \tparam NUM_MMA_KV The number of fragments in z dimension.
 * \tparam num_warps The number of warps in the threadblock.
 * \tparam DTypeQ The data type of the query tensor.
 * \tparam DTypeKV The data type of the key/value tensor.
 * \tparam DTypeO The data type of the output tensor.
 * \param q The query tensor.
 * \param k The key tensor.
 * \param v The value tensor.
 * \param o The output tensor.
 * \param tmp The temporary buffer (used when partition_kv is true).
 * \param lse The logsumexp value.
 * \param rope_rcp_scale 1/(rope_scale), where rope_scale is the scaling
 *   factor used in RoPE interpolation.
 * \param rope_rcp_theta 1/(rope_theta), where rope_theta is the theta
 *   used in RoPE.
 */
template <MaskMode MASK_MODE, PosEncodingMode POS_ENCODING_MODE, uint32_t NUM_MMA_Q,
          uint32_t NUM_MMA_D_QK, uint32_t NUM_MMA_D_VO, uint32_t NUM_MMA_KV, uint32_t NUM_WARPS_Q,
          uint32_t NUM_WARPS_KV, typename DTypeQKAccum, typename AttentionVariant, typename Params>
__global__
__launch_bounds__(NUM_WARPS_Q* NUM_WARPS_KV* WARP_SIZE) void PODWithKVCacheKernel(
    const uint_fastdiv group_size, const __grid_constant__ Params prefill_params) {

    extern __shared__ uint8_t smem[];
    //dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);

    Operation op = PREFILL;
    int linear_blk_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    if(op == PREFILL) {
      if(threadIdx.x >= 32 || threadIdx.y >= NUM_WARPS_Q || threadIdx.z >= NUM_WARPS_KV)
        return;
      constexpr uint32_t num_rows_per_cta = NUM_MMA_Q * NUM_WARPS_Q * 16;
      const uint32_t qo_len = prefill_params.qo_len;
      const uint32_t xsize = ceil_div(qo_len * group_size, num_rows_per_cta);
      const uint32_t num_kv_heads = prefill_params.num_kv_heads;

      // Not partition_kv
      if (!prefill_params.partition_kv) {
        const uint32_t lane_idx = threadIdx.x, warp_idx = get_warp_idx<NUM_WARPS_Q, NUM_WARPS_KV>();
        //dim3 nblks(ceil_div(qo_len * group_size, num_rows_per_cta), 1, num_kv_heads);

        // BlockID exceeds limit
        if(linear_blk_id >= xsize * num_kv_heads) return;

        const uint32_t bx = linear_blk_id % xsize;
        const uint32_t chunk_idx = 0;
        const uint32_t kv_head_idx = linear_blk_id / xsize;
        SinglePrefillWithKVCacheDevice<MASK_MODE, POS_ENCODING_MODE, NUM_MMA_Q, NUM_MMA_D_QK,
          NUM_MMA_D_VO, NUM_MMA_KV, NUM_WARPS_Q, NUM_WARPS_KV, DTypeQKAccum, AttentionVariant>
          (group_size, prefill_params, smem, lane_idx, warp_idx, bx, chunk_idx, kv_head_idx, num_kv_heads);
      } else {
        const uint32_t lane_idx = threadIdx.x, warp_idx = get_warp_idx<NUM_WARPS_Q, NUM_WARPS_KV>();
        //dim3 nblks(ceil_div(qo_len * group_size, num_rows_per_cta), num_chunks, num_kv_heads);
        // BlockID exceeds limit
        const uint32_t num_chunks = prefill_params.partition_kv;
        if(linear_blk_id >= xsize * num_chunks * num_kv_heads) return;
        const uint32_t bx = linear_blk_id % xsize;
        const uint32_t chunk_idx = (linear_blk_id / xsize) % num_chunks;
        const uint32_t kv_head_idx = linear_blk_id / (xsize * num_chunks);
        SinglePrefillWithKVCacheDevice<MASK_MODE, POS_ENCODING_MODE, NUM_MMA_Q, NUM_MMA_D_QK,
          NUM_MMA_D_VO, NUM_MMA_KV, NUM_WARPS_Q, NUM_WARPS_KV, DTypeQKAccum, AttentionVariant>
          (group_size, prefill_params, smem, lane_idx, warp_idx, bx, chunk_idx, kv_head_idx, num_kv_heads);
    }
  }
}


template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE, 
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename AttentionVariant, 
          typename PrefillParams>
cudaError_t PODWithKVCacheDispatched(PrefillParams prefill_params, 
                                     typename PrefillParams::DTypeO* prefill_tmp,
                                     cudaStream_t stream) {
  using DTypeQ = typename PrefillParams::DTypeQ;
  using DTypeKV = typename PrefillParams::DTypeKV;
  using DTypeO = typename PrefillParams::DTypeO;
  const uint32_t num_qo_heads = prefill_params.num_qo_heads;
  const uint32_t num_kv_heads = prefill_params.num_kv_heads;
  const uint32_t qo_len = prefill_params.qo_len;
  const uint32_t kv_len = prefill_params.kv_len;
  if (kv_len < qo_len && MASK_MODE == MaskMode::kCausal) {
    std::ostringstream err_msg;
    err_msg << "When mask_mode is set to MaskMode::kCausal, kv_len must be greater than or equal "
               "to qo_len, got kv_len"
            << kv_len << " and qo_len " << qo_len;
    FLASHINFER_ERROR(err_msg.str());
  }

  const uint32_t group_size = num_qo_heads / num_kv_heads;
  const uint_fastdiv group_size_fastdiv(group_size); 
  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  uint32_t cta_tile_q = 0;
  int64_t unpacked_qo_len = qo_len * group_size;
  if (unpacked_qo_len > 64 && HEAD_DIM_VO < 256) {
    cta_tile_q = 128;
  } else {
    auto compute_capacity = GetCudaComputeCapability();
    if (compute_capacity.first >= 8) {
      // Ampere or newer
      if (unpacked_qo_len > 16) {
        // avg_packed_qo_len <= 64
        cta_tile_q = 64;
      } else {
        // avg_packed_qo_len <= 16
        cta_tile_q = 16;
      }
    } else {
      // NOTE(Zihao): not enough shared memory on Turing for 1x4 warp layout
      cta_tile_q = 64;
    }
  }

  DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
    constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
    constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);
    constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);

    using DTypeQKAccum =
        typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                  float>::type;

    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    int max_smem_per_sm = 0;
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
    // we expect each sm execute two threadblocks
    // TODO(Zihao): fix the following computation
    const int num_ctas_per_sm = max_smem_per_sm > (16 * HEAD_DIM_QK * sizeof(DTypeQ) * 16) ? 2 : 1;
    const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

    const uint32_t max_num_mma_kv_reg =
        (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
         !USE_FP16_QK_REDUCTION)
            ? 2
            : (8 / NUM_MMA_Q);
    // TODO(Zihao): fix the following computation
    const uint32_t max_num_mma_kv_smem =
        (max_smem_per_threadblock / (16 * HEAD_DIM_QK * sizeof(DTypeQ)) - NUM_MMA_Q * NUM_WARPS_Q) /
        (2 * NUM_WARPS_KV);

    // control NUM_MMA_KV for maximum warp occupancy
    DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
      if constexpr (is_invalid_configuration<POS_ENCODING_MODE, DTypeKV, DTypeQKAccum>(
                        NUM_MMA_Q, NUM_MMA_D_QK, NUM_MMA_D_VO, NUM_MMA_KV, NUM_WARPS_Q, 
                        NUM_WARPS_KV)) {
        // Invalid configuration, skip
        std::ostringstream err_msg;
        err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=" << NUM_MMA_Q
                << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
                << " NUM_MMA_KV=" << NUM_MMA_KV << " NUM_WARPS_Q=" << NUM_WARPS_Q
                << " NUM_WARPS_KV=" << NUM_WARPS_KV
                << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                   " and report the issue to the developers.";
        FLASHINFER_ERROR(err_msg.str());
      } else {
        constexpr uint32_t num_threads = (NUM_WARPS_Q * NUM_WARPS_KV) * WARP_SIZE;
        constexpr uint32_t num_rows_per_cta = NUM_MMA_Q * NUM_WARPS_Q * 16;
        auto kernel =
            PODWithKVCacheKernel<MASK_MODE, POS_ENCODING_MODE, NUM_MMA_Q, NUM_MMA_D_QK,
                                           NUM_MMA_D_VO, NUM_MMA_KV, NUM_WARPS_Q, 
                                           NUM_WARPS_KV, DTypeQKAccum, AttentionVariant, 
                                           PrefillParams>;

        // TODO(Zihao): fix the following computation
        size_t smem_size =
            max(SmemSizeThreadBlockAttnSync<NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_Q, HEAD_DIM_VO,
                                            DTypeO>(),
                NUM_MMA_Q * NUM_WARPS_Q * 16 * HEAD_DIM_QK * sizeof(DTypeQ) +
                    NUM_MMA_KV * NUM_WARPS_KV * 16 * (HEAD_DIM_QK + HEAD_DIM_VO) * sizeof(DTypeKV));
        FLASHINFER_CUDA_CALL(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        int num_blocks_per_sm = 0;
        int num_sm = 0;
        FLASHINFER_CUDA_CALL(
            cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
        FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, kernel, num_threads, smem_size));
        uint32_t max_num_kv_chunks =
            (num_blocks_per_sm * num_sm) /
            (num_kv_heads * ceil_div(qo_len * group_size, num_rows_per_cta));
        uint32_t num_chunks;
        if (max_num_kv_chunks > 0) {
          uint32_t chunk_size = max(ceil_div(kv_len, max_num_kv_chunks), 256);
          num_chunks = ceil_div(kv_len, chunk_size);
        } else {
          num_chunks = 0;
        }

        if (num_chunks <= 1 || prefill_tmp == nullptr) {
          // Enough parallelism, do not split-kv
          prefill_params.partition_kv = 0;
          void* args[] = {(void*)&group_size_fastdiv, (void*)&prefill_params};
          //dim3 nblks(ceil_div(qo_len * group_size, num_rows_per_cta), 1, num_kv_heads);
          dim3 nblks(ceil_div(qo_len * group_size, num_rows_per_cta), 1, num_kv_heads);
          dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        } else {
          // Use cooperative groups to increase occupancy
          prefill_params.partition_kv = num_chunks;
          float* tmp_lse = (float*)(prefill_tmp + num_chunks * qo_len * num_qo_heads * HEAD_DIM_VO);
          auto o = prefill_params.o;
          auto lse = prefill_params.lse;
          prefill_params.o = prefill_tmp;
          prefill_params.lse = tmp_lse;
          void* args[] = {(void*)&group_size_fastdiv, (void*)&prefill_params};
          dim3 nblks(ceil_div(qo_len * group_size, num_rows_per_cta), num_chunks, num_kv_heads);
          dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
          if constexpr (AttentionVariant::use_softmax) {
            FLASHINFER_CUDA_CALL(MergeStates(prefill_tmp, tmp_lse, o, lse, num_chunks, qo_len, num_qo_heads,
                                             HEAD_DIM_VO, stream));
          } else {
            FLASHINFER_CUDA_CALL(
                AttentionSum(prefill_tmp, o, num_chunks, qo_len, num_qo_heads, HEAD_DIM_VO, stream));
          }
        }
      }
    })
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_CUH_
