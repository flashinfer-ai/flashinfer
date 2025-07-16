#ifndef FLASHINFER_POD_CUH_
#define FLASHINFER_POD_CUH_

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
#include "decode.cuh"
#include "mask.cuh"
#include "prefill.cuh"
#include "variants.cuh"

namespace flashinfer {

namespace cg = cooperative_groups;
using cp_async::SharedMemFillMode;
using mma::MMAMode;

enum Operation {
  PREFILL = 0,
  DECODE = 1,
};

template <typename KTraits_P, typename KTraits_D, typename PrefillParams, typename DecodeParams>
__global__ __launch_bounds__(std::max(
    KTraits_P::NUM_THREADS,
    KTraits_D::NUM_THREADS)) void PODWithKVCacheTensorKernel(const __grid_constant__ PrefillParams
                                                                 prefill_params,
                                                             const __grid_constant__ DecodeParams
                                                                 decode_params,
                                                             int* tbAssign) {
  extern __shared__ uint8_t smem[];
  const uint32_t num_kv_heads = prefill_params.paged_kv.num_heads;
  // PREFILL VARS
  const uint32_t padded_bsize_p = prefill_params.padded_batch_size;

  // DECODE VARS
  const uint32_t padded_bsize_d = decode_params.padded_batch_size;

  // THREADBLOCKS
  const uint32_t prefill_blocks = padded_bsize_p * num_kv_heads;
  const uint32_t decode_blocks = padded_bsize_d * num_kv_heads;

  int op;
  int linear_bid;
  // SM-aware CTA scheduler
  if (threadIdx.x == 0) {
    // TODO_AK: If num_threads dont match, use virtual sub-CTAs.
    // Requires changing block-level sync in main prefill/decode kernels.
    constexpr int blk_factor_p = 1;
    constexpr int blk_factor_d = 1;

    // SM-aware threadblock scheduler code
    // Find out which SM this threadblock is scheduled on
    int num_SMs;
    // WARNING: nsmid has only been tested on A100/H100, and matches SM count
    // No guarantee this will work on other GPUs
    asm volatile("mov.u32 %0, %nsmid;" : "=r"(num_SMs));
    asm volatile("mov.u32 %0, %smid;" : "=r"(linear_bid));
    const int prefill_slots = (prefill_blocks + blk_factor_p - 1) / blk_factor_p;
    const int decode_slots = (decode_blocks + blk_factor_d - 1) / blk_factor_d;

    if (prefill_slots <= decode_slots) {
      // Total tags = (decode + prefill) / min(decode, prefill)
      // = 1 + decode / prefill; when prefill < decode
      const int total_tags = decode_slots / prefill_slots + 1;
      // For this SM, what's the next operation we want to run?
      op = (atomicAdd(&tbAssign[linear_bid], 1) % total_tags);
      if (op > 0) {
        op = 1;
      }
    } else {
      // Total tags = (decode + prefill) / min(decode, prefill)
      // = 1 + prefill / decode; when decode < prefill
      const int pref_tags = prefill_slots / decode_slots;

      // For this SM, what's the next operation we want to run?
      op = (atomicAdd(&tbAssign[linear_bid], 1) % (pref_tags + 1));
      if (op < pref_tags) {
        op = 0;
      } else {
        op = 1;
      }
    }

    // Get the next blockId for that operation
    linear_bid = atomicAdd(&tbAssign[num_SMs + op], 1);
    // If the blockId obtained exceeds the max blockIds for that op, switch to the other op
    if (op == 0 && linear_bid >= prefill_slots) {
      linear_bid = atomicAdd(&tbAssign[num_SMs + 1], 1);
      op = !op;
    } else if (op == 1 && linear_bid >= decode_slots) {
      op = !op;
      linear_bid = atomicAdd(&tbAssign[num_SMs + 0], 1);
    }
    // Write the global blockId and operation to shared memory
    ((int*)smem)[0] = linear_bid;
    ((int*)smem)[1] = op;
  }
  // Sync to wait for dynamic scheduler to write to smem
  __syncthreads();
  // Fetch from shared memory the assigned blockId and operation.
  linear_bid = ((int*)smem)[0];
  op = ((int*)smem)[1];
  // Sync to force all threads to wait
  // __syncthreads();

  if (op == PREFILL) {
    const uint32_t linear_tid = threadIdx.x;
    // Return if threadId exceeds number of threads for this op
    if (linear_tid >= 32 * KTraits_P::NUM_WARPS_Q * KTraits_P::NUM_WARPS_KV) return;
    if (linear_bid >= prefill_blocks) return;

    const dim3 tid = dim3(linear_tid % 32, (linear_tid / 32) % KTraits_P::NUM_WARPS_Q,
                          (linear_tid / 32) / KTraits_P::NUM_WARPS_Q);
    auto& smem_storage = reinterpret_cast<typename KTraits_P::SharedStorage&>(smem);
    const uint32_t bx = linear_bid % padded_bsize_p;
    const uint32_t kv_head_idx = linear_bid / padded_bsize_p;

    BatchPrefillWithPagedKVCacheDevice<KTraits_P>(prefill_params, smem_storage, tid, bx,
                                                  kv_head_idx, num_kv_heads);

  } else /* OP == DECODE */ {
    const uint32_t linear_tid = threadIdx.x;
    // Return if threadId exceeds number of threads for this op
    if (linear_tid >= 32 * KTraits_D::NUM_WARPS_Q * KTraits_D::NUM_WARPS_KV) return;
    if (linear_bid >= decode_blocks) return;

    const dim3 tid = dim3(linear_tid % 32, (linear_tid / 32) % KTraits_D::NUM_WARPS_Q,
                          (linear_tid / 32) / KTraits_D::NUM_WARPS_Q);
    auto& smem_storage = reinterpret_cast<typename KTraits_D::SharedStorage&>(smem);
    // dim3 nblks_d(padded_batch_size_d, 1, num_kv_heads);
    const uint32_t bx = linear_bid % padded_bsize_d;
    const uint32_t kv_head_idx = linear_bid / padded_bsize_d;

    // dim3 nthrs_d(32, NUM_WARPS_Q_D, NUM_WARPS_KV_D);

    // Decode is faster with tensor cores, which are usually not saturated by prefill
    BatchPrefillWithPagedKVCacheDevice<KTraits_D>(decode_params, smem_storage, tid, bx, kv_head_idx,
                                                  num_kv_heads);
  }
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE_P, uint32_t CTA_TILE_Q_P,
          uint32_t CTA_TILE_Q_D, MaskMode MASK_MODE_D, typename PrefillAttentionVariant,
          typename DecodeAttentionVariant, typename PrefillParams, typename DecodeParams>
cudaError_t PODWithKVCacheTensorDispatched(PrefillParams prefill_params, DecodeParams decode_params,
                                           typename DecodeParams::DTypeO* tmp_v, float* tmp_s,
                                           bool enable_pdl, cudaStream_t stream) {
  static_assert(std::is_same<typename PrefillParams::DTypeQ, typename DecodeParams::DTypeQ>::value);
  static_assert(
      std::is_same<typename PrefillParams::DTypeKV, typename DecodeParams::DTypeKV>::value);
  static_assert(std::is_same<typename PrefillParams::DTypeO, typename DecodeParams::DTypeO>::value);
  // Ensure heads match
  assert(prefill_params.paged_kv.num_heads == decode_params.paged_kv.num_heads);
  assert(prefill_params.num_qo_heads == decode_params.num_qo_heads);
  // Prefill variable setup
  using DTypeQ_P = typename PrefillParams::DTypeQ;
  using DTypeKV_P = typename PrefillParams::DTypeKV;
  using DTypeO_P = typename PrefillParams::DTypeO;
  const uint32_t num_qo_heads = prefill_params.num_qo_heads;
  const uint32_t num_kv_heads = prefill_params.paged_kv.num_heads;

  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;

  // Decode vars setup
  using DTypeQ_D = typename DecodeParams::DTypeQ;
  using DTypeKV_D = typename DecodeParams::DTypeKV;
  using DTypeO_D = typename DecodeParams::DTypeO;
  const uint32_t padded_batch_size_d = decode_params.padded_batch_size;
  constexpr uint32_t NUM_MMA_Q_D = get_num_mma_q(CTA_TILE_Q_D);
  constexpr uint32_t NUM_WARPS_Q_D = get_num_warps_q(CTA_TILE_Q_D);
  constexpr uint32_t NUM_WARPS_KV_D = get_num_warps_kv(CTA_TILE_Q_D);

  if (padded_batch_size_d == 0) {
    // No request, skip
    // this won't happen in CUDAGraph mode because we fixed the padded_batch_size
    return cudaSuccess;
  }

  // constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  // constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  using DTypeQKAccum_D =
      typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ_D, half>, half,
                                float>::type;

  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  int max_smem_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&max_smem_per_sm,
                                              cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
  // we expect each sm execute two threadblocks
  // TODO(Zihao): fix the following computation
  const int num_ctas_per_sm = max_smem_per_sm > (16 * HEAD_DIM_QK * sizeof(DTypeQ_D) * 16) ? 2 : 1;
  const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

  constexpr uint32_t max_num_mma_kv_reg_d =
      (HEAD_DIM_VO >= 128 && NUM_MMA_Q_D == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !USE_FP16_QK_REDUCTION)
          ? 2
          : (8 / NUM_MMA_Q_D);
  // TODO(Zihao): fix the following computation
  const uint32_t max_num_mma_kv_smem_d =
      (max_smem_per_threadblock / (16 * HEAD_DIM_QK * sizeof(DTypeQ_D)) -
       NUM_MMA_Q_D * NUM_WARPS_Q_D) /
      (2 * NUM_WARPS_KV_D);

  constexpr uint32_t NUM_WARPS_Q_P = get_num_warps_q(CTA_TILE_Q_P);
  constexpr uint32_t NUM_WARPS_KV_P = get_num_warps_kv(CTA_TILE_Q_P);
  constexpr uint32_t NUM_MMA_Q_P = get_num_mma_q(CTA_TILE_Q_P);

  using DTypeQKAccum_P =
      typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ_P, half>, half,
                                float>::type;

  // we expect each sm execute two threadblocks
  // TODO(Zihao): fix the following computation
  const int num_ctas_per_sm_p =
      max_smem_per_sm > (16 * HEAD_DIM_QK * sizeof(DTypeQ_P) * 16) ? 2 : 1;
  const int max_smem_per_threadblock_p = max_smem_per_sm / num_ctas_per_sm_p;

  constexpr uint32_t max_num_mma_kv_reg_p =
      (HEAD_DIM_VO >= 128 && NUM_MMA_Q_P == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !USE_FP16_QK_REDUCTION)
          ? 2
          : (8 / NUM_MMA_Q_P);
  // TODO(Zihao): fix the following computation
  const uint32_t max_num_mma_kv_smem_p =
      (max_smem_per_threadblock_p / (16 * HEAD_DIM_QK * sizeof(DTypeQ_P)) -
       NUM_MMA_Q_P * NUM_WARPS_Q_P) /
      (2 * NUM_WARPS_KV_P);

  // control NUM_MMA_KV for maximum warp occupancy
  uint32_t max_num_mma_kv_p = std::min(max_num_mma_kv_smem_p, max_num_mma_kv_reg_p);
  uint32_t max_num_mma_kv_d = std::min(max_num_mma_kv_smem_d, max_num_mma_kv_reg_d);

  DISPATCH_NUM_MMA_KV(max_num_mma_kv_p, NUM_MMA_KV_P, {
    using KTraits_P = KernelTraits<MASK_MODE_P, CTA_TILE_Q_P, NUM_MMA_Q_P, NUM_MMA_KV_P,
                                   NUM_MMA_D_QK, NUM_MMA_D_VO, NUM_WARPS_Q_P, NUM_WARPS_KV_P,
                                   POS_ENCODING_MODE, DTypeQ_P, DTypeKV_P, DTypeO_P, DTypeQKAccum_P,
                                   typename PrefillParams::IdType, PrefillAttentionVariant>;

    if constexpr (KTraits_P::IsInvalid()) {
      // Invalid configuration, skip
      std::ostringstream err_msg;
      err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=" << NUM_MMA_Q_P
              << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
              << " NUM_MMA_KV=" << NUM_MMA_KV_P << " NUM_WARPS_Q=" << NUM_WARPS_Q_P
              << " NUM_WARPS_KV=" << NUM_WARPS_KV_P
              << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                 " and report the issue to the developers.";
      FLASHINFER_ERROR(err_msg.str());
    } else {
      // Decode stuff
      // TODO: Is there a way to avoid this nested dispatch?
      DISPATCH_NUM_MMA_KV(max_num_mma_kv_d, NUM_MMA_KV_D, {
        using KTraits_D =
            KernelTraits<MASK_MODE_D, CTA_TILE_Q_D, NUM_MMA_Q_D, NUM_MMA_KV_D, NUM_MMA_D_QK,
                         NUM_MMA_D_VO, NUM_WARPS_Q_D, NUM_WARPS_KV_D, POS_ENCODING_MODE, DTypeQ_D,
                         DTypeKV_D, DTypeO_D, DTypeQKAccum_D, typename DecodeParams::IdType,
                         DecodeAttentionVariant>;
        if constexpr (KTraits_D::IsInvalid()) {
          // Invalid configuration, skip
          std::ostringstream err_msg;
          err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=" << NUM_MMA_Q_D
                  << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
                  << " NUM_MMA_KV=" << NUM_MMA_KV_D << " NUM_WARPS_Q=" << NUM_WARPS_Q_D
                  << " NUM_WARPS_KV=" << NUM_WARPS_KV_D
                  << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                     " and report the issue to the developers.";
          FLASHINFER_ERROR(err_msg.str());
        } else {
          // End decode stuff
          constexpr uint32_t num_threads_p = (NUM_WARPS_Q_P * NUM_WARPS_KV_P) * WARP_SIZE;
          size_t smem_size_p = sizeof(typename KTraits_P::SharedStorage);
          size_t smem_size_d = sizeof(typename KTraits_D::SharedStorage);

          auto kernel =
              PODWithKVCacheTensorKernel<KTraits_P, KTraits_D, PrefillParams, DecodeParams>;
          // Prefill: decide num_splits for split-kv
          int num_blocks_per_sm = 0;
          int num_sm = 0;
          FLASHINFER_CUDA_CALL(
              cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
          // FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          //     &num_blocks_per_sm, kernel, num_threads_p, smem_size_p));
          //  Above function returns 0 for some reason, so we use a workaround
          num_blocks_per_sm = std::max(
              1, std::min((int)(max_smem_per_sm / smem_size_p), (int)(256 / num_threads_p)));

          // Setup new prefill params if (not) split
          auto o = prefill_params.o;
          auto lse = prefill_params.lse;
          if (prefill_params.partition_kv) {
            // Use cooperative groups to increase occupancy
            assert(tmp_v != nullptr);
            prefill_params.o = tmp_v;
            prefill_params.lse = tmp_s;
          }

          // Setup new decode params if (not) split
          if (prefill_params.partition_kv) {
            assert(tmp_v != nullptr);
            decode_params.o = tmp_v;
            decode_params.lse = tmp_s;
          }

          uint32_t padded_batch_size_p = prefill_params.padded_batch_size;
          uint32_t padded_batch_size_d = decode_params.padded_batch_size;
          printf("Debug: launching prefill with padded_batch_size_p %d, num_kv_heads %d\n",
                 padded_batch_size_p, num_kv_heads);
          int nblks_p(padded_batch_size_p * num_kv_heads);
          int nthrs_p(32 * NUM_WARPS_Q_P * NUM_WARPS_KV_P);
          printf("Debug: launching decode with padded_batch_size_d %d, num_kv_heads %d\n",
                 padded_batch_size_d, num_kv_heads);
          int nblks_d(padded_batch_size_d * num_kv_heads);
          int nthrs_d(32 * NUM_WARPS_Q_D * NUM_WARPS_KV_D);

          // ******* Select final combined sizes here ******* /
          size_t smem_size = max(smem_size_p, smem_size_d);
          int nblks = nblks_p + nblks_d;
          int nthrs = max(nthrs_p, nthrs_d);

          // printf("Smem: prefill %zu, decode %zu, total %zu\n", smem_size_p, smem_size_d,
          // smem_size); printf("Blocks: prefill %d, decode %d, total %d\n", nblks_p, nblks_d,
          // nblks); printf("Threads: prefill %d, decode %d, total %d\n", nthrs_p, nthrs_d,
          // nthrs);
          //  ************************************************ /

          static int* tbAssign = nullptr;
          if (tbAssign == nullptr) cudaMalloc(&tbAssign, sizeof(int) * (num_sm + 2));
          cudaMemset(tbAssign, 0, sizeof(int) * (num_sm + 2));

          // Setup kernel arguments

          FLASHINFER_CUDA_CALL(
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

          // Launch kernel
          if (enable_pdl) {
            cudaLaunchAttribute attribute[1];
            cudaLaunchConfig_t config;
            attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute[0].val.programmaticStreamSerializationAllowed = 1;
            config.attrs = attribute;
            config.numAttrs = 1;
            config.gridDim = nblks;
            config.blockDim = nthrs;
            config.dynamicSmemBytes = smem_size;
            config.stream = stream;
            FLASHINFER_CUDA_CALL(
                cudaLaunchKernelEx(&config, kernel, prefill_params, decode_params, tbAssign));
          } else {
            void* args[] = {(void*)&prefill_params, (void*)&decode_params, (void*)&tbAssign};
            FLASHINFER_CUDA_CALL(
                cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
          }

          // Post-kernel stuff for split-kv
          if (tmp_v != nullptr) {
            if constexpr (DecodeAttentionVariant::use_softmax) {
              FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
                  tmp_v, tmp_s, decode_params.merge_indptr, o, lse,
                  decode_params.max_total_num_rows, decode_params.total_num_rows, num_qo_heads,
                  HEAD_DIM_VO, enable_pdl, stream));
            } else {
              FLASHINFER_CUDA_CALL(VariableLengthAttentionSum(
                  tmp_v, decode_params.merge_indptr, o, decode_params.max_total_num_rows,
                  decode_params.total_num_rows, num_qo_heads, HEAD_DIM_VO, enable_pdl, stream));
            }
          }
        }
      });
    }
  });
  return cudaSuccess;
}
}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_CUH_
