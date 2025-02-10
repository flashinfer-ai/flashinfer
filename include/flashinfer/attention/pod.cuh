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
template <// Prefill template
          MaskMode MASK_MODE, PosEncodingMode POS_ENCODING_MODE, uint32_t NUM_MMA_Q,
          uint32_t NUM_MMA_D_QK, uint32_t NUM_MMA_D_VO, uint32_t NUM_MMA_KV, uint32_t NUM_WARPS_Q,
          uint32_t NUM_WARPS_KV, typename DTypeQKAccum, typename PrefillAttentionVariant,
          // Decode template
          uint32_t num_stages_smem, uint32_t tile_size_per_bdx, uint32_t vec_size, 
          uint32_t bdx, uint32_t bdy, uint32_t bdz, typename DecodeAttentionVariant,
          typename PrefillParams, typename DecodeParams>
__global__
__launch_bounds__(NUM_WARPS_Q * NUM_WARPS_KV * WARP_SIZE) void PODWithKVCacheKernel(
    const uint_fastdiv group_size, const __grid_constant__ PrefillParams prefill_params,
    const __grid_constant__ DecodeParams decode_params) {

  extern __shared__ uint8_t smem[];
  //dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);

  // PREFILL VARS
  const uint32_t num_kv_heads_p = prefill_params.num_kv_heads;
  const uint32_t num_chunks = prefill_params.partition_kv;
  constexpr uint32_t num_rows_per_cta = NUM_MMA_Q * NUM_WARPS_Q * 16;
  const uint32_t qo_len = prefill_params.qo_len;
  const uint32_t xsize = ceil_div(qo_len * group_size, num_rows_per_cta);

  // DECODE VARS
  const uint32_t num_kv_heads_d = decode_params.paged_kv.num_heads;
  const uint32_t padded_batch_size = decode_params.padded_batch_size;

  // THREADBLOCKS
  const uint32_t prefill_blocks = num_kv_heads_p * xsize * (num_chunks ? num_chunks : 1);
  const uint32_t decode_blocks = num_kv_heads_d * padded_batch_size;

  Operation op = PREFILL;
  int linear_bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  if(linear_bid >= prefill_blocks) {
    linear_bid -= prefill_blocks;
    op = DECODE;
  }
  /*if(threadIdx.x == 0) {
      // SM-aware threadblock scheduler code
      // Find out which SM this threadblock is scheduled on
      int num_SMs;
      // WARNING: nsmid has only been tested on A100, and matches SM count
      // No guarantee this will work on other GPUs
      asm volatile("mov.u32 %0, %nsmid;" : "=r"(num_SMs));
      asm volatile("mov.u32 %0, %smid;" : "=r"(linear_bid));
      const int prefill_slots = (prefill_blocks + blk_factor_p - 1) / blk_factor_p;
      const int decode_slots = (decode_blocks + blk_factor_d - 1) / blk_factor_d;
      
      if constexpr(FusedOp & 1) {
          if(prefill_slots <= decode_slots) {
              // Total tags = (decode + prefill) / min(decode, prefill)
              // = 1 + decode / prefill; when prefill < decode
              const int total_tags = decode_slots / prefill_slots + 1;
              // For this SM, what's the next operation we want to run?
              op = (atomicAdd(&tbAssign[linear_bid], 1) % total_tags);
              if(op > 0) {
                  op = 1;
              }
          } else {
              // Total tags = (decode + prefill) / min(decode, prefill)
              // = 1 + prefill / decode; when decode < prefill
              const int pref_tags = prefill_slots / decode_slots;
              
              // For this SM, what's the next operation we want to run?
              op = (atomicAdd(&tbAssign[linear_bid], 1) % (pref_tags + 1));
              if(op < pref_tags) {
                  op = 0;
              } else {
                  op = 1;
              }
          }
      } else {
          op = atomicAdd(&tbAssign[linear_bid], 1) % 2;
      }

      // Get the next blockId for that operation
      linear_bid = atomicAdd(&tbAssign[num_SMs + op], 1);
      // If the blockId obtained exceeds the max blockIds for that op, switch to the other op
      if(op == 0 && linear_bid >= prefill_slots) {
          linear_bid = atomicAdd(&tbAssign[num_SMs + 1], 1);
          op = !op;
      } else if (op == 1 && linear_bid >= decode_slots) {
          op = !op;
          linear_bid = atomicAdd(&tbAssign[num_SMs + 0], 1);
      }
      // Write the blockId and operation to shared memory
      ((int*)smem_)[0] = linear_bid;
      ((int*)smem_)[1] = op;
  }
  // Sync to wait for dynamic scheduler to finish
  __syncthreads();
  // Fetch from shared memory the assigned blockId and operation.
  linear_bid = ((int*)smem_)[0];
  op = ((int*)smem_)[1];
  // Sync to force all threads to wait
  __syncthreads();*/

  if(op == PREFILL) {
    const uint32_t linear_tid = threadIdx.x;
    // Return if threadId exceeds number of threads for this op
    if(linear_tid >= 32 * NUM_WARPS_Q * NUM_WARPS_KV)
      return;

    const dim3 tid = dim3(linear_tid % 32, (linear_tid / 32) % NUM_WARPS_Q, (linear_tid / 32) / NUM_WARPS_Q);
    //dim3 nblks(ceil_div(qo_len * group_size, num_rows_per_cta), 1, num_kv_heads);
    //dim3 nblks(ceil_div(qo_len * group_size, num_rows_per_cta), num_chunks, num_kv_heads);
    // BlockID exceeds limit
    if(linear_bid >= prefill_blocks) return;

    // Not partition_kv
    if (!prefill_params.partition_kv) {
      const uint32_t bx = linear_bid % xsize;
      const uint32_t chunk_idx = 0;
      const uint32_t kv_head_idx = linear_bid / xsize;

      SinglePrefillWithKVCacheDevice<MASK_MODE, POS_ENCODING_MODE, NUM_MMA_Q, NUM_MMA_D_QK,
        NUM_MMA_D_VO, NUM_MMA_KV, NUM_WARPS_Q, NUM_WARPS_KV, DTypeQKAccum, PrefillAttentionVariant>
        (group_size, prefill_params, smem, tid, bx, chunk_idx, kv_head_idx, 1, num_kv_heads_p);
    } else {
      const uint32_t bx = linear_bid % xsize;
      const uint32_t chunk_idx = (linear_bid / xsize) % num_chunks;
      const uint32_t kv_head_idx = linear_bid / (xsize * num_chunks);

      SinglePrefillWithKVCacheDevice<MASK_MODE, POS_ENCODING_MODE, NUM_MMA_Q, NUM_MMA_D_QK,
        NUM_MMA_D_VO, NUM_MMA_KV, NUM_WARPS_Q, NUM_WARPS_KV, DTypeQKAccum, PrefillAttentionVariant>
        (group_size, prefill_params, smem, tid, bx, chunk_idx, kv_head_idx, num_chunks, num_kv_heads_p);
    }
  } else /* OP == DECODE */ {
    //dim3 nblks_d(padded_batch_size, num_kv_heads);
    if(linear_bid >= decode_blocks)
      return;
    const uint32_t bx = linear_bid % padded_batch_size;
    const uint32_t by = linear_bid / padded_batch_size;

    //dim3 nthrs_d(bdx, bdy, bdz);
    const uint32_t linear_tid = threadIdx.x;
    // Return if threadId exceeds number of threads for this op
    if(linear_tid >= bdx * bdy * bdz)
      return;

    const uint32_t tx = linear_tid % bdx;
    const uint32_t ty = (linear_tid / bdx) % bdy;
    const uint32_t tz = (linear_tid / bdx) / bdy;
    BatchDecodeWithPagedKVCacheDevice<POS_ENCODING_MODE, num_stages_smem, tile_size_per_bdx,
      vec_size, bdx, bdy, bdz, DecodeAttentionVariant>(decode_params, smem, bx, by, tx, ty, tz);
  }
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename PrefillAttentionVariant,
          typename DecodeAttentionVariant, typename PrefillParams, typename DecodeParams>
cudaError_t PODWithKVCacheDispatched(PrefillParams prefill_params, 
                                     typename PrefillParams::DTypeO* tmp_p,
                                     DecodeParams decode_params,
                                     typename DecodeParams::DTypeO* tmp_v,
                                     float *tmp_s, cudaStream_t stream) {
  // Ensure KV heads match
  assert(prefill_params.num_kv_heads == decode_params.paged_kv.num_heads);
  // Prefill variable setup
  using DTypeQ_P = typename PrefillParams::DTypeQ;
  using DTypeKV_P = typename PrefillParams::DTypeKV;
  using DTypeO_P = typename PrefillParams::DTypeO;
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

  // Decode variable setup
  using DTypeKV_D = typename DecodeParams::DTypeKV;
  const uint32_t num_qo_heads_d = decode_params.num_qo_heads;
  //const uint32_t num_kv_heads = decode_params.paged_kv.num_heads;
  const uint32_t padded_batch_size = decode_params.padded_batch_size;

  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV_D), HEAD_DIM_QK / 32UL);
  auto compute_capacity = GetCudaComputeCapability();
  constexpr uint32_t bdx = HEAD_DIM_QK / vec_size;
  static_assert(bdx <= 32);

  // Now figure out which kernel to run

  //DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
    constexpr uint32_t CTA_TILE_Q = 128;
    constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
    constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);
    constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);

    using DTypeQKAccum =
        typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ_P, half>, half,
                                  float>::type;

    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    int max_smem_per_sm = 0;
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
    // we expect each sm execute two threadblocks
    // TODO(Zihao): fix the following computation
    const int num_ctas_per_sm = max_smem_per_sm > (16 * HEAD_DIM_QK * sizeof(DTypeQ_P) * 16) ? 2 : 1;
    const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

    const uint32_t max_num_mma_kv_reg =
        (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
         !USE_FP16_QK_REDUCTION)
            ? 2
            : (8 / NUM_MMA_Q);
    // TODO(Zihao): fix the following computation
    const uint32_t max_num_mma_kv_smem =
        (max_smem_per_threadblock / (16 * HEAD_DIM_QK * sizeof(DTypeQ_P)) - NUM_MMA_Q * NUM_WARPS_Q) /
        (2 * NUM_WARPS_KV);

    // control NUM_MMA_KV for maximum warp occupancy
    DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
      //constexpr size_t NUM_MMA_KV = 2;
      if constexpr (is_invalid_configuration<POS_ENCODING_MODE, DTypeKV_P, DTypeQKAccum>(
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
        // Decode stuff
        //DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
          constexpr size_t GROUP_SIZE = 1;
          constexpr uint32_t bdy = GROUP_SIZE;
          constexpr uint32_t num_threads_d = std::max(128U, bdx * bdy);
          constexpr uint32_t bdz = num_threads_d / (bdx * bdy);
          constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV_D) == 1 ? 2U : 4U) : 1U;
          //DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
            constexpr uint32_t NUM_STAGES_SMEM = 2;

            // End decode stuff
            constexpr uint32_t num_threads_p = (NUM_WARPS_Q * NUM_WARPS_KV) * WARP_SIZE;
            constexpr uint32_t num_rows_per_cta = NUM_MMA_Q * NUM_WARPS_Q * 16;

            // TODO(Zihao): fix the following computation
            size_t smem_size_p =
                max(SmemSizeThreadBlockAttnSync<NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_Q, HEAD_DIM_VO,
                  DTypeO_P>(),
                    NUM_MMA_Q * NUM_WARPS_Q * 16 * HEAD_DIM_QK * sizeof(DTypeQ_P) +
                        NUM_MMA_KV * NUM_WARPS_KV * 16 * (HEAD_DIM_QK + HEAD_DIM_VO) * sizeof(DTypeKV_P));
            
            size_t smem_size_d =
              2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM_QK * sizeof(DTypeKV_D) +
              std::max(tile_size_per_bdx * num_threads_d * sizeof(DTypeKV_D*),
              2 * bdy * bdz * sizeof(float));

            auto kernel =
                PODWithKVCacheKernel<MASK_MODE, POS_ENCODING_MODE, NUM_MMA_Q, NUM_MMA_D_QK,
                                              NUM_MMA_D_VO, NUM_MMA_KV, NUM_WARPS_Q, 
                                              NUM_WARPS_KV, DTypeQKAccum, PrefillAttentionVariant, 
                                              NUM_STAGES_SMEM, tile_size_per_bdx, vec_size,
                                              bdx, bdy, bdz, DecodeAttentionVariant,
                                              PrefillParams, DecodeParams>;

            //auto kernel_d =
            //    BatchDecodeWithPagedKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
            //                                      vec_size, bdx, bdy, bdz, DecodeAttentionVariant, DecodeParams>;
            //FLASHINFER_CUDA_CALL(
            //    cudaFuncSetAttribute(kernel_d, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_d));
            int num_blocks_per_sm = 0;
            int num_sm = 0;
            FLASHINFER_CUDA_CALL(
                cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
            FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &num_blocks_per_sm, kernel, num_threads_p, smem_size_p));
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
            
            // Setup new prefill params if (not) split
            auto o_p = prefill_params.o;
            auto lse_p = prefill_params.lse;
            float* tmp_lse = (float*)(tmp_p + num_chunks * qo_len * num_qo_heads * HEAD_DIM_VO);
            if (num_chunks <= 1 || tmp_p == nullptr) {
              // Enough parallelism, do not split-kv
              prefill_params.partition_kv = 0;
            } else {
              // Use cooperative groups to increase occupancy
              prefill_params.partition_kv = num_chunks;
              prefill_params.o = tmp_p;
              prefill_params.lse = tmp_lse;
            }

            // Setup new decode params if (not) split
            auto o_d = decode_params.o;
            auto lse_d = decode_params.lse;
            if (tmp_v == nullptr) {
              // do not use partition-kv kernel
              decode_params.partition_kv = false;
            } else {
              // use partition-kv kernel
              decode_params.partition_kv = true;
              decode_params.o = tmp_v;
              decode_params.lse = tmp_s;
            }

            int nblks_d(padded_batch_size * num_kv_heads);
            int nthrs_d(bdx * bdy * bdz);
            //void* args_d[] = {(void*)&decode_params};
            //FLASHINFER_CUDA_CALL(
            //    cudaLaunchKernel((void*)kernel_d, nblks_d, nthrs_d, args_d, smem_size_d, stream));

            int nblks_p(ceil_div(qo_len * group_size, num_rows_per_cta) * 
              (prefill_params.partition_kv ? prefill_params.partition_kv : 1) * num_kv_heads);
            int nthrs_p(32 * NUM_WARPS_Q * NUM_WARPS_KV);

            //******* Select final combined sizes here *******/
            size_t smem_size = max(smem_size_p, smem_size_d);
            int nblks = nblks_p + nblks_d;
            int nthrs = max(nthrs_p, nthrs_d);

            printf("Smem: prefill %zu, decode %zu, total %zu\n", smem_size_p, smem_size_d, smem_size);
            printf("Blocks: prefill %d, decode %d, total %d\n", nblks_p, nblks_d, nblks);
            printf("Threads: prefill %d, decode %d, total %d\n", nthrs_p, nthrs_d, nthrs);
            //************************************************/

            // Setup kernel arguments
            void* args[] = {(void*)&group_size_fastdiv, (void*)&prefill_params, (void*)&decode_params};
            FLASHINFER_CUDA_CALL(
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            // Launch kernel
            FLASHINFER_CUDA_CALL(
                cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
            
            // Post-kernel stuff for split-kv prefill
            if (!(num_chunks <= 1 || tmp_p == nullptr)) {
              if constexpr (PrefillAttentionVariant::use_softmax) {
                FLASHINFER_CUDA_CALL(MergeStates(tmp_p, tmp_lse, o_p, lse_p, num_chunks, qo_len, num_qo_heads,
                                                HEAD_DIM_VO, stream));
              } else {
                FLASHINFER_CUDA_CALL(
                    AttentionSum(tmp_p, o_p, num_chunks, qo_len, num_qo_heads, HEAD_DIM_VO, stream));
              }
            }
            // Post-kernel stuff for split-kv decode
            if(tmp_v != nullptr) {
              if constexpr (DecodeAttentionVariant::use_softmax) {
                FLASHINFER_CUDA_CALL(VariableLengthMergeStates(tmp_v, tmp_s, decode_params.o_indptr, o_d, lse_d,
                                                              decode_params.paged_kv.batch_size, nullptr,
                                                              num_qo_heads, HEAD_DIM_QK, stream));
              } else {
                FLASHINFER_CUDA_CALL(VariableLengthAttentionSum(tmp_v, decode_params.o_indptr, o_d,
                                                                decode_params.paged_kv.batch_size, nullptr,
                                                                num_qo_heads, HEAD_DIM_QK, stream));
              }
            }
          //});
        //});
      }
    });
  //});

  /***********************************************************************/
  
  /***********************************************************************/
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_CUH_
