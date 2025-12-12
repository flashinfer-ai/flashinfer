#pragma once

#include "cute/tensor.hpp"
#include "flat/cute_ext.hpp"
#include "flat/math.hpp"

#define DEBUG_PIPE 0
#include "flat/debug.hpp"

#define CUDA_UNIFORM(expr) __shfl_sync(0xffffffff, (expr), 0)
#define CUDA_UNIFORM_LIKELY(x) __builtin_expect(!!CUDA_UNIFORM(x), 1)
#define CUDA_UNIFORM_UNLIKELY(x) __builtin_expect(!!CUDA_UNIFORM(x), 0)

namespace flat {

using namespace cute;

template <int NumThreads, int HeadSize, int BlockSize, typename TO, typename TQKV, typename TState>
struct LinearAttentionPrefillPipeline {
  struct Ldg;   // Q/K/V ldg configuration
  struct Smem;  // Q/K/V smem block configuration
  struct QK;    // QK MMA
  struct KV;    // KV MMA
  struct O;     // QK @ V + Q @ KV MMA

  __forceinline__ __device__ static void attention(
      TO* __restrict__ output,                 // ["packed_seq", Hqo, dq]
      TState* __restrict__ output_state,       // [num_seqs, Hkv, dv, dk], aka, KV, optional
      TQKV const* __restrict__ q,              // ["packed_seq", Hqo, dq]
      TQKV const* __restrict__ k,              // ["packed_seq", Hkv, dk]
      TQKV const* __restrict__ v,              // ["packed_seq", Hkv, dv]
      TState const* __restrict__ input_state,  // [num_seqs, Hkv, dv, dk], aka, KV initial value,
                                               // optional
      int64_t const* __restrict__ cu_seqlens,  // [num_seqs + 1], prefix scan of packed length of
                                               // sequences in the batch
      int32_t num_seqs, int32_t num_qo_heads, int32_t num_kv_heads,
      float scale,  // scaling on QK
      float decay, float const* per_head_decay, int32_t decay_exponent_offset) {
    int32_t const seq_idx = blockIdx.x / num_qo_heads;
    int32_t const qo_head_idx = blockIdx.x % num_qo_heads;
    int32_t const kv_head_idx = qo_head_idx / (num_qo_heads / num_kv_heads);

    int64_t const tok_offset = cu_seqlens[seq_idx];
    int64_t const seq_len = cu_seqlens[seq_idx + 1] - tok_offset;
    int64_t const num_blocks = ceil_div(seq_len, BlockSize);

    // #pragma region Partitioning
    // the last blocks of these tensors need special handling
    auto gO = make_tensor(
        make_gmem_ptr(output + tok_offset * (num_qo_heads * HeadSize)),
        make_layout(make_shape(Int<HeadSize>{}, num_qo_heads, Int<BlockSize>{}, num_blocks)))(
        _, qo_head_idx, _, _);  // (Dim, Tok, blk) -> idx, NOTE: Tok is token index in Block
    auto gQ = make_tensor(
        make_gmem_ptr(q + tok_offset * (num_qo_heads * HeadSize)),
        make_layout(make_shape(Int<HeadSize>{}, num_qo_heads, Int<BlockSize>{}, num_blocks)))(
        _, qo_head_idx, _, _);  // (Dim, Tok, blk) -> idx
    auto gK = make_tensor(
        make_gmem_ptr(k + tok_offset * (num_kv_heads * HeadSize)),
        make_layout(make_shape(Int<HeadSize>{}, num_kv_heads, Int<BlockSize>{}, num_blocks)))(
        _, kv_head_idx, _, _);  // (Dim, Tok, blk) -> idx
    auto gV = make_tensor(
        make_gmem_ptr(v + tok_offset * (num_kv_heads * HeadSize)),
        make_layout(make_shape(Int<HeadSize>{}, num_kv_heads, Int<BlockSize>{}, num_blocks)))(
        _, kv_head_idx, _, _);  // (Dim, Tok, blk) -> idx
    auto gKV = make_tensor(
        make_gmem_ptr(output_state),
        make_layout(make_shape(Int<HeadSize>{}, Int<HeadSize>{},
                               per_head_decay ? num_qo_heads : num_kv_heads, num_seqs)))(
        _, _, per_head_decay ? qo_head_idx : kv_head_idx, seq_idx);  // (KDim, VDim), K-contiguous

    if (per_head_decay != nullptr) {
      decay = per_head_decay[qo_head_idx];
    }

    extern __shared__ char dynamic_smem[];

    auto smem = new (&dynamic_smem[0]) Smem;
    auto sQ = make_tensor(make_smem_ptr(smem->q.data()), Smem::QLayout());  // (Dim, Tok, blk)
    auto sK = make_tensor(make_smem_ptr(smem->k.data()), Smem::KLayout());  // (Dim, Tok, blk)
    auto sV = make_tensor(make_smem_ptr(smem->v.data()), Smem::VLayout());  // (Dim, Tok, blk)
    auto sQK = make_tensor(make_smem_ptr(smem->qk.data()),
                           Smem::QKLayout());  // (KTok, QTok), K-contiguous

    auto sKV_opd = make_tensor(make_smem_ptr(smem->kv_opd.data()),
                               Smem::KVOpdLayout());  // (KDim, VDim), K-contiguous

    auto const cQ = make_identity_tensor(Shape<Int<HeadSize>, Int<BlockSize>>{});
    auto const& cO = cQ;
    auto const& cK = cQ;
    auto const cM0 = make_identity_tensor(Shape<Int<BlockSize>, Int<BlockSize>>{});  // (QTok, KTok)

    auto const qkv_thr_copy = typename Ldg::TiledCopy{}.get_thread_slice(threadIdx.x);
    auto const tQgQ = qkv_thr_copy.partition_S(gQ);  // ((Val), IterD, IterT, blk)
    auto const tKgK = qkv_thr_copy.partition_S(gK);  // ((Val), IterD, IterT, blk)
    auto const tVgV = qkv_thr_copy.partition_S(gV);  // ((Val), IterD, IterT, blk)
    auto const tQcQ = qkv_thr_copy.partition_S(cQ);  // ((Val), IterD, IterT) -> (Dim, Tok)
    auto tQsQ = qkv_thr_copy.partition_D(sQ);        // ((Val), IterD, IterT, Pipe)
    auto tKsK = qkv_thr_copy.partition_D(sK);        // ((Val), IterD, IterT, Pipe)
    auto tVsV = qkv_thr_copy.partition_D(sV);        // ((Val), IterD, IterT, Pipe)
#if 0
    if (thread0()) {
      printf("ValDT=[%dx1], ThrDT=[%dx%d]\n", QKV_ValD, QKV_ThrD, QKV_ThrT);
      print(QKV_TiledCopy{}), print("\n");
      print(gQ), print("\n");
      print(tQgQ), print("\n");
      print(tQsQ), print("\n");
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // MMA for related operations when result is of shape Q@K
    constexpr auto qk_tiled_mma = typename QK::TiledMMA{};
    auto const qk_thr_mma = qk_tiled_mma.get_thread_slice(threadIdx.x);

    auto tQKrQ_thr_copy =
        make_tiled_copy_A(typename QK::S2R{}, qk_tiled_mma).get_thread_slice(threadIdx.x);
    auto tQKrK_thr_copy =
        make_tiled_copy_B(typename QK::S2R{}, qk_tiled_mma).get_thread_slice(threadIdx.x);

    // A
    auto tQKrQ = qk_thr_mma.partition_fragment_A(select_tensor<1, 0>(sQ(_, _, _0{})));
    auto tQKrQ_cv = tQKrQ_thr_copy.retile_D(tQKrQ);  // copy view
    auto tQKsQ = tQKrQ_thr_copy.partition_S(select_tensor<1, 0, 2>(sQ));
    // B
    auto tQKrK = qk_thr_mma.partition_fragment_B(select_tensor<1, 0>(sK(_, _, _0{})));
    auto tQKrK_cv = tQKrK_thr_copy.retile_D(tQKrK);  // copy view
    auto tQKsK = tQKrK_thr_copy.partition_S(select_tensor<1, 0, 2>(sK));
    // C
    auto tQKsQK = qk_thr_mma.partition_C(select_tensor<1, 0>(sQK));
    auto tQKrQK = qk_thr_mma.partition_fragment_C(select_tensor<1, 0>(sQK));
    auto tQKcM0 = qk_thr_mma.partition_C(cM0);  // (idx) -> (tok_q, tok_k)
#if 0
    if (thread0()) {
      print("------\n");
      print("tQKsQ : "), print(tQKsQ), print("\n");
      print("tQKrQ : "), print(tQKrQ), print(", nreg="), print(size(tQKrQ) / 2), print("\n");
      print("tQKsK : "), print(tQKsK), print("\n");
      print("tQKrK : "), print(tQKrK), print(", nreg="), print(size(tQKrK) / 2), print("\n");
      print("sQK   : "), print(sQK), print("\n");
      print("tQKsQK: "), print(tQKsQK), print("\n");
      print("tQKrQK: "), print(tQKrQK), print(", nreg="), print(size(tQKrQK) / 2), print("\n");
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // MMA for related operations when result is of shape K@V
    constexpr auto kv_tiled_mma = typename KV::TiledMMA{};
    auto const kv_thr_mma = kv_tiled_mma.get_thread_slice(threadIdx.x);

    auto tKVrV_thr_copy =
        make_tiled_copy_A(typename KV::S2R{}, kv_tiled_mma).get_thread_slice(threadIdx.x);
    auto tKVrK_thr_copy =
        make_tiled_copy_B(typename KV::S2R{}, kv_tiled_mma).get_thread_slice(threadIdx.x);

    auto tKVrKV_opd_thr_r2s =
        make_tiled_copy_C(typename KV::R2S_opd{}, kv_tiled_mma).get_thread_slice(threadIdx.x);

    // A
    auto tKVrV = kv_thr_mma.partition_fragment_A(sV(_, _, _0{}));
    auto tKVrV_cv = tKVrV_thr_copy.retile_D(tKVrV);
    auto tKVsV = tKVrV_thr_copy.partition_S(sV);
    // B
    auto tKVcK = kv_thr_mma.partition_B(cK);
    auto tKVrK = kv_thr_mma.partition_fragment_B(sK(_, _, _0{}));
    auto tKVrK_cv = tKVrK_thr_copy.retile_D(tKVrK);
    auto tKVsK = tKVrK_thr_copy.partition_S(sK);
    // C
    auto tKVgKV = kv_thr_mma.partition_C(select_tensor<1, 0>(gKV));

    auto tKVsKV_cvt = tKVrKV_opd_thr_r2s.partition_D(select_tensor<1, 0>(sKV_opd));
    auto tKVrKV = make_fragment_like<TState>(tKVsKV_cvt);
#if 0
    if (thread0()) {
      print("------\n");
      print("tKVsK : "), print(tKVsK), print("\n");
      print("tKVrK : "), print(tKVrK), print("\n");
      print("tKVsV : "), print(tKVsV), print("\n");
      print("tKVrV : "), print(tKVrV), print("\n");
      print("tKVrKV: "), print(tKVrKV), print("\n");
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // MMA for related operations when result is of shape Q@K@V, aka, Output
    constexpr auto o_tiled_mma = typename O::TiledMMA{};
    auto const o_thr_mma = o_tiled_mma.get_thread_slice(threadIdx.x);

    auto tOrA_thr_copy =
        make_tiled_copy_A(typename O::S2R{}, o_tiled_mma).get_thread_slice(threadIdx.x);
    auto tOrB_thr_copy =
        make_tiled_copy_B(typename O::S2R{}, o_tiled_mma).get_thread_slice(threadIdx.x);
    auto tOrV_thr_copy = make_tiled_copy_B(typename O::S2R_V{}, o_tiled_mma)
                             .get_thread_slice(threadIdx.x);  // V needs transpose

    // A1
    auto tOrQ = o_thr_mma.partition_fragment_A(select_tensor<1, 0>(sQ(_, _, _0{})));
    auto tOrQ_cv = tOrA_thr_copy.retile_D(tOrQ);
    auto tOsQ = tOrA_thr_copy.partition_S(select_tensor<1, 0, 2>(sQ));
    // B1
    auto tOrKV = o_thr_mma.partition_fragment_B(select_tensor<1, 0>(sKV_opd));
    auto tOrKV_cv = tOrB_thr_copy.retile_D(tOrKV);
    auto tOsKV = tOrB_thr_copy.partition_S(select_tensor<1, 0>(sKV_opd));
    // A2
    auto tOrQK = o_thr_mma.partition_fragment_A(select_tensor<1, 0>(sQK));
    auto tOrQK_cv = tOrA_thr_copy.retile_D(tOrQK);
    auto tOsQK = tOrA_thr_copy.partition_S(select_tensor<1, 0>(sQK));
    // B2
    auto tOrV = o_thr_mma.partition_fragment_B(sV(_, _, _0{}));
    auto tOrV_cv = tOrV_thr_copy.retile_D(tOrV);
    auto tOsV = tOrV_thr_copy.partition_S(sV);
    // C = mma(A1, B1) + mma(A2, B2)
    auto tOgO = o_thr_mma.partition_C(select_tensor<1, 0, 2>(gO));
    auto tOrO = o_thr_mma.partition_fragment_C(select_tensor<1, 0>(gO(_, _, _0{})));
    auto tOcO = o_thr_mma.partition_C(select_tensor<1, 0>(cO));
#if 0
    if (thread0()) {
      print("------\n");
      print("tOsQ : "), print(tOsQ), print("\n");
      print("tOrQ : "), print(tOrQ), print("\n");
      print("tOsKV: "), print(tOsKV), print("\n");
      print("tOrKV: "), print(tOrKV), print("\n");
      print("tOsQK: "), print(tOsQK), print("\n");
      print("tOrQK: "), print(tOrQK), print("\n");
      print("tOsV : "), print(tOsV), print("\n");
      print("tOrV : "), print(tOrV), print("\n");
      print("tOrO : "), print(tOrO), print("\n");
    }
#endif
    // #pragma endregion Partitioning

    bool const needs_decay = decay != 1.0f;
    float block_decay = 1.0f;
    if (needs_decay) {
      precompute_decay(decay, smem->decay.data());
      __syncthreads();
      block_decay = smem->decay[BlockSize];
    }

    auto remaining_seq_len = [&](int blk) { return seq_len - blk * BlockSize; };

    auto compute_loop_body = [&](int blk, int pipe, auto is_final_block_) {
      PIPE_DEBUG_PRINTF("[%d,%d,%d]|| working on blk=%d, pipe=%d\n", seq_idx, qo_head_idx,
                        kv_head_idx, blk, pipe);
      constexpr bool is_final_block = decltype(is_final_block_)::value;

      if constexpr (is_final_block) {
        if (needs_decay) {
          block_decay = smem->decay[remaining_seq_len(blk)];
        }
      }

      /////////////////////////////////////////////////////////////////////////
      // 1. compute masked_qk = mask(QK)
      // 1.1 QK
      PIPE_DEBUG_PRINTF("[%d,%d,%d]** compute QK\n", seq_idx, qo_head_idx, kv_head_idx);
      clear(tQKrQK);
      copy(typename QK::S2R{}, tQKsQ(_, _, _, pipe), tQKrQ_cv);
      copy(typename QK::S2R{}, tQKsK(_, _, _, pipe), tQKrK_cv);
      gemm(qk_tiled_mma, tQKrQ, tQKrK, tQKrQK);

      // 1.2 apply mask on QK
      PIPE_DEBUG_PRINTF("[%d,%d,%d]** compute masked_QK\n", seq_idx, qo_head_idx, kv_head_idx);
      transform(tQKrQK, tQKcM0, tQKrQK, [&](auto val, auto coord) {
        auto [s, t] = coord;
        auto scaled = s >= t ? scale * val : decltype(val){};  // also masked
        if (needs_decay) {
          auto Lambda = s >= t ? smem->decay[s - t] : 0.0f;
          scaled *= Lambda;
        }
        return scaled;
      });

      copy(tQKrQK, tQKsQK);  // write masked_qk to shared memory for kv part

      /////////////////////////////////////////////////////////////////////////
      // 2. compute qkv
      // 2.1 Q @ KV, NOTE: use old KV here
      PIPE_DEBUG_PRINTF("[%d,%d,%d]** compute Q @ KV\n", seq_idx, qo_head_idx, kv_head_idx);
      clear(tOrO);
      if (blk != 0) {
        copy(typename O::S2R{}, tOsQ(_, _, _, pipe), tOrQ_cv);
        copy(typename O::S2R{}, tOsKV, tOrKV_cv);
        gemm(o_thr_mma, tOrQ, tOrKV, tOrO);
        if (needs_decay) {
          transform(tOrO, tOcO, tOrO, [&](auto val, auto coord) {
            auto [_, tok] = coord;
            return val * smem->decay[tok + decay_exponent_offset];
          });
        }
      }

      // 2.2 QK @ V
      PIPE_DEBUG_PRINTF("[%d,%d,%d]** compute QK @ V\n", seq_idx, qo_head_idx, kv_head_idx);
      __syncthreads();  // ensure finished writing of QK buffer (finished KV buffer consumption as a
                        // side effect)
      copy(typename O::S2R_V{}, tOsV(_, _, _, pipe), tOrV_cv);
      copy(typename O::S2R{}, tOsQK, tOrQK_cv);
      gemm(o_thr_mma, tOrQK, tOrV, tOrO);
      if constexpr (!is_final_block) {
        PIPE_DEBUG_PRINTF("[%d,%d,%d]>> save tOrO -> tOgO, blk=%d\n", seq_idx, qo_head_idx,
                          kv_head_idx, blk);
        copy(tOrO, tOgO(_, _, _, blk));
      } else if (seq_len % BlockSize == 0) {
        PIPE_DEBUG_PRINTF("[%d,%d,%d]>> save tOrO -> tOgO, blk=%d (tail block, full)\n", seq_idx,
                          qo_head_idx, kv_head_idx, blk);
        copy(tOrO, tOgO(_, _, _, blk));
      } else {
        PIPE_DEBUG_PRINTF("[%d,%d,%d]>> save tOrO -> tOgO, blk=%d (tail block, non-full)\n",
                          seq_idx, qo_head_idx, kv_head_idx, blk);
        store_o_tail(tOrO, tOgO(_, _, _, blk), tOcO, remaining_seq_len(blk));
      }

      /////////////////////////////////////////////////////////////////////////
      // 3. update KV
      PIPE_DEBUG_PRINTF("[%d,%d,%d]** compute KV\n", seq_idx, qo_head_idx, kv_head_idx);
      copy(typename KV::S2R{}, tKVsV(_, _, _, pipe), tKVrV_cv);  // load A
      copy(typename KV::S2R{}, tKVsK(_, _, _, pipe), tKVrK_cv);  // load B
      if (needs_decay) {                                         // decay by Lambda * lambda^(-tok)
        int B = is_final_block ? remaining_seq_len(blk) : BlockSize;
        transform(tKVrK, tKVcK, tKVrK, [&](auto val, auto coord) {
          auto tok = get<1>(coord);
          float decay_k = [&] {
            if constexpr (!is_final_block) {
              return smem->decay[B - tok - decay_exponent_offset];
            } else {
              return tok < B ? smem->decay[B - tok - decay_exponent_offset] : 1.0f;
            }
          }();
          return decltype(val)(val * decay_k);
        });
      }
      auto tKVrKV_inc = make_tensor_like(tKVrKV);
      clear(tKVrKV_inc);
      gemm(kv_tiled_mma, tKVrV, tKVrK, tKVrKV_inc);
      if (scale != 1.0f) {
        transform(tKVrKV_inc, tKVrKV_inc, [&](auto val) { return val * scale; });
      }
      transform(tKVrKV, tKVrKV_inc, tKVrKV,
                [&](auto carried_kv, auto inc_kv) { return block_decay * carried_kv + inc_kv; });

      if constexpr (!is_final_block) {  // write sKV for next block
        PIPE_DEBUG_PRINTF("[%d,%d,%d]>> save tKVrKV -> tKVsKV\n", seq_idx, qo_head_idx,
                          kv_head_idx);
        copy(typename KV::R2S_opd{}, tKVrKV,
             tKVsKV_cvt);  // premature convert for next block iteration
      } else {             // write gKV for output
        bool is_lead_kv_head = (qo_head_idx % (num_qo_heads / num_kv_heads)) == 0;
        if (per_head_decay || is_lead_kv_head) {
          PIPE_DEBUG_PRINTF("[%d,%d,%d]>> save tKVrKV -> tKVgKV\n", seq_idx, qo_head_idx,
                            kv_head_idx);
          copy(tKVrKV, tKVgKV);
        }
      }
    };

    for_each(make_int_sequence<NumPipe>{}, [&](auto pipe) {
      auto blk = pipe();
      if (blk >= num_blocks) {
        return;
      }

      if (CUDA_UNIFORM_LIKELY(blk < num_blocks - 1 || seq_len % BlockSize == 0)) {
        PIPE_DEBUG_PRINTF("[%d,%d,%d]|| preloading from blk=%d to pipe=%d\n", seq_idx, qo_head_idx,
                          kv_head_idx, blk, pipe());
        load_qkv(tQgQ(_, _, _, blk), tQsQ(_, _, _, pipe), tKgK(_, _, _, blk), tKsK(_, _, _, pipe),
                 tVgV(_, _, _, blk), tVsV(_, _, _, pipe));
      } else {
        PIPE_DEBUG_PRINTF("[%d,%d,%d]|| preloading from blk=%d to pipe=%d (tail block)\n", seq_idx,
                          qo_head_idx, kv_head_idx, blk, pipe());
        load_qkv_tail(tQgQ(_, _, _, blk), tQsQ(_, _, _, pipe), tKgK(_, _, _, blk),
                      tKsK(_, _, _, pipe), tVgV(_, _, _, blk), tVsV(_, _, _, pipe), tQcQ,
                      remaining_seq_len(blk));
      }
    });

    __syncthreads();
    if (input_state == nullptr) {
      PIPE_DEBUG_PRINTF("[%d,%d,%d]>> zero init tKVrKV\n", seq_idx, qo_head_idx, kv_head_idx);
      clear(tKVrKV);
    } else {
      PIPE_DEBUG_PRINTF("[%d,%d,%d]>> init tKVrKV with input_state\n", seq_idx, qo_head_idx,
                        kv_head_idx);
      Tensor gKV_in = make_tensor(
          make_gmem_ptr(output_state),
          make_layout(make_shape(Int<HeadSize>{}, Int<HeadSize>{},
                                 per_head_decay ? num_qo_heads : num_kv_heads, num_seqs)))(
          _, _, per_head_decay ? qo_head_idx : kv_head_idx, seq_idx);  // (KDim, VDim), K-contiguous
      Tensor tKVgKV_in = kv_thr_mma.partition_C(select_tensor<1, 0>(gKV_in));
      copy(tKVgKV_in, tKVrKV);
    }

    int blk = 0;
    for (; blk < num_blocks - NumPipe; blk++) {
      int pipe = blk % NumPipe;
      wait_for_qkv();
      compute_loop_body(blk, pipe, /*is_final_block_=*/cute::false_type{});

      /////////////////////////////////////////////////////////////////////////
      // qkv blockwise buffer preload
      __syncthreads();  // all the threads have consumed the target pipe state, we can issue the
                        // load now

      // current pipe stage consumed, launch preloading operation to it
      int blk_to_load = blk + NumPipe;
      if (CUDA_UNIFORM_LIKELY(blk_to_load < num_blocks - 1 || seq_len % BlockSize == 0)) {
        PIPE_DEBUG_PRINTF("[%d,%d,%d]|| preloading from blk=%d to pipe=%d\n", seq_idx, qo_head_idx,
                          kv_head_idx, blk_to_load, pipe);
        load_qkv(tQgQ(_, _, _, blk_to_load), tQsQ(_, _, _, pipe), tKgK(_, _, _, blk_to_load),
                 tKsK(_, _, _, pipe), tVgV(_, _, _, blk_to_load), tVsV(_, _, _, pipe));
      } else {
        PIPE_DEBUG_PRINTF("[%d,%d,%d]|| preloading from blk=%d to pipe=%d (tail block)\n", seq_idx,
                          qo_head_idx, kv_head_idx, blk_to_load, pipe);
        load_qkv_tail(tQgQ(_, _, _, blk_to_load), tQsQ(_, _, _, pipe), tKgK(_, _, _, blk_to_load),
                      tKsK(_, _, _, pipe), tVgV(_, _, _, blk_to_load), tVsV(_, _, _, pipe), tQcQ,
                      remaining_seq_len(blk_to_load));
      }
    }

    auto num_invalid_pipe = cute::max(NumPipe - num_blocks, 0);
    for_each(make_int_rsequence<NumPipe>{}, [&](auto RemainingPipe) {
      auto iter_idx = NumPipe - RemainingPipe - 1;

      auto blk_idx = blk + iter_idx - num_invalid_pipe;
      if (blk_idx < 0) {
        return;
      }  // means continue here, as we are in a lambda of the for_each
      wait_for_qkv<RemainingPipe>();
      if constexpr (RemainingPipe != 0) {
        compute_loop_body(blk_idx, blk_idx % NumPipe, /*is_final_block_=*/cute::false_type{});
      } else {
        compute_loop_body(blk_idx, blk_idx % NumPipe, /*is_final_block_=*/cute::true_type{});
      }
    });
  }

 public:
  static constexpr int NumPipe = 2;

  struct Ldg {
    static constexpr int QKV_ValD = 8;  // target for LDG.128

    using CopyAtom = Copy_Atom<
        SM80_CP_ASYNC_CACHEALWAYS<cute::array_aligned<TQKV, QKV_ValD, QKV_ValD * sizeof(TQKV)>>,
        TQKV>;
    using CopyAtomZ = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<
                                    cute::array_aligned<TQKV, QKV_ValD, QKV_ValD * sizeof(TQKV)>>,
                                TQKV>;
    using TiledCopy = decltype(make_tiled_copy(
        Ldg::CopyAtom{}, tile_to_shape(Layout<Shape<_8, _4>, Stride<_1, _8>>{}, Shape<_16, _16>{}),
        Layout<Int<QKV_ValD>>{}));
  };

  struct Smem {
    using Swizzle_8x8_4x1 = decltype(composition(
        Swizzle<3, 3, 3>{}, Layout<Shape<Shape<_8, _4>, _8>, Stride<Stride<_1, _64>, _8>>{}));

    using QLayoutAtom = Swizzle_8x8_4x1;
    using QLayout = decltype(tile_to_shape(QLayoutAtom{},
                                           Shape<Int<HeadSize>, Int<BlockSize>, Int<NumPipe>>{}));

    using KLayoutAtom = Swizzle_8x8_4x1;
    using KLayout = decltype(tile_to_shape(KLayoutAtom{},
                                           Shape<Int<HeadSize>, Int<BlockSize>, Int<NumPipe>>{}));

    using VLayoutAtom = Swizzle_8x8_4x1;
    using VLayout = decltype(tile_to_shape(VLayoutAtom{},
                                           Shape<Int<HeadSize>, Int<BlockSize>, Int<NumPipe>>{}));

    using QKLayoutAtom = Layout<Shape<_8, _8>>;
    using QKLayout =
        decltype(tile_to_shape(QKLayoutAtom{}, Shape<Int<BlockSize>, Int<BlockSize>>{}));

    using KVOpdLayoutAtom = Layout<Shape<_8, _8>>;
    using KVOpdLayout =
        decltype(tile_to_shape(KVOpdLayoutAtom{}, Shape<Int<HeadSize>, Int<HeadSize>>{}));

    cute::array_aligned<TQKV, cosize_v<QLayout>> q;
    cute::array_aligned<TQKV, cosize_v<KLayout>> k;
    cute::array_aligned<TQKV, cosize_v<VLayout>> v;
    cute::array_aligned<TQKV, cosize_v<QKLayout>> qk;
    cute::array_aligned<TQKV, cosize_v<KVOpdLayout>> kv_opd;
    cute::array_aligned<float, BlockSize + 1> decay;
  };

  __forceinline__ __device__ static void precompute_decay(float decay_factor, float* decay) {
    constexpr int WarpSize = 32;
    int warp_id = threadIdx.x / WarpSize;
    int lane_id = threadIdx.x % WarpSize;

    constexpr int len = BlockSize + 1;

    if (warp_id != 0) {
      return;
    }

    float scale = decay_factor;

    for (int base = 0; base < len; base += WarpSize) {
      float prod = scale;
      for (int offset = 1; offset < WarpSize; offset *= 2) {
        auto v = __shfl_xor_sync(0xFFFFFFFF, prod, offset);
        if (threadIdx.x > offset) {
          prod *= v;
        }
      }

      int vidx = base + lane_id;  // take scale==2.0 as example
      if (vidx == base) {
        prod = 1.0f;
      };  // correct prod as 2^0, 2^1, ..., 2^tid, ... for first iter
      if (base != 0) {
        prod *= scale * decay[base - 1];
      }  // correct prod as 2^(tid+1) for remaining iters
      if (vidx < len) {
        decay[vidx] = prod;
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // keep the asynchronize pipeline state with the following helpers
  /////////////////////////////////////////////////////////////////////////////
  template <typename QSrc, typename QDst, typename KSrc, typename KDst, typename VSrc,
            typename VDst>
  __forceinline__ __device__ static void load_qkv(QSrc&& q_src, QDst&& q_dst, KSrc&& k_src,
                                                  KDst&& k_dst, VSrc&& v_src, VDst&& v_dst) {
#if !defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    static_assert(dependent_false<QSrc>, "cp.async is not available");
#endif
    // total 2 groups of cp.async per pipe
    auto const atom = typename Ldg::CopyAtom{};
    copy(atom, std::forward<QSrc>(q_src), std::forward<QDst>(q_dst));
    copy(atom, std::forward<KSrc>(k_src), std::forward<KDst>(k_dst));
    copy(atom, std::forward<VSrc>(v_src), std::forward<VDst>(v_dst));
    cp_async_fence();
  }

  template <typename QSrc, typename QDst, typename KSrc, typename KDst, typename VSrc,
            typename VDst, typename Coord, typename Index>
  __forceinline__ __device__ static void load_qkv_tail(QSrc&& q_src, QDst&& q_dst, KSrc&& k_src,
                                                       KDst&& k_dst, VSrc&& v_src, VDst&& v_dst,
                                                       Coord&& coord, Index&& remaining_seq_len) {
#if !defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    static_assert(dependent_false<QSrc>, "cp.async is not available");
#endif
    constexpr auto L = typename cute::remove_cvref_t<QDst>::layout_type{};
    constexpr auto R = rank(L);
    auto p = make_tensor<bool>(take<1, R>(shape(L)));

    CUTE_UNROLL
    for (int n = 0; n < size<1>(p); ++n) {
      CUTE_UNROLL
      for (int m = 0; m < size<0>(p); ++m) {
        auto [_, tok] = std::forward<Coord>(coord)(_0{}, m, n);
        p(m, n) = tok < remaining_seq_len;
      }
    }

    auto const zfill = typename Ldg::CopyAtomZ{};
    copy_if(zfill, p, std::forward<QSrc>(q_src), std::forward<QDst>(q_dst));
    copy_if(zfill, p, std::forward<KSrc>(k_src), std::forward<KDst>(k_dst));
    copy_if(zfill, p, std::forward<VSrc>(v_src), std::forward<VDst>(v_dst));
    cp_async_fence();
  }

  template <int NumRemainingPipe = NumPipe - 1>
  __forceinline__ __device__ static void wait_for_qkv() {
    cp_async_wait<NumRemainingPipe>();
    __syncthreads();
  }

  template <typename Src, typename Dst, typename Coord, typename Index>
  __forceinline__ __device__ static void store_o_tail(Src&& src, Dst&& dst, Coord&& coord,
                                                      Index&& remaining_seq_len) {
    auto p = make_tensor<bool>(shape(std::forward<Src>(src)));
    CUTE_UNROLL
    for (int i = 0; i < size(std::forward<Src>(src)); ++i) {
      auto [_, tok] = coord(i);
      p(i) = tok < remaining_seq_len;
    }
    copy_if(p, std::forward<Src>(src), std::forward<Dst>(dst));
  }

  using MMA = std::conditional_t<std::is_same_v<TQKV, cute::bfloat16_t>,
                                 SM80_16x8x16_F32BF16BF16F32_TN, SM80_16x8x16_F32F16F16F32_TN>;
  struct QK {
    using TiledMMA = decltype(make_tiled_mma(
        MMA{}, Layout<Shape<_2, _4>>{},  // 2x4 of atoms to fully cover the threads in a CTA
        Tile<_32, _32, _32>{}));
    static_assert(size_v<QK::TiledMMA> == NumThreads, "CTA Cooperative MMA Assumed!");

    using S2R = Copy_Atom<SM75_U32x4_LDSM_N, TQKV>;
  };

  struct KV {
    using TiledMMA =
        decltype(make_tiled_mma(MMA{}, Layout<Shape<_2, _4>>{}, Tile<_32, _32, _32>{}));
    static_assert(size_v<KV::TiledMMA> == NumThreads, "CTA Cooperative MMA Assumed!");
    using S2R = Copy_Atom<SM75_U16x8_LDSM_T, TQKV>;

    using S2R_acc = Copy_Atom<SM75_U32x4_LDSM_N, float>;
    using R2S_acc = Copy_Atom<AutoVectorizingCopy, float>;
    using R2S_opd = Copy_Atom<AutoVectorizingCopy, TQKV>;
  };

  struct O {
    using TiledMMA =
        decltype(make_tiled_mma(MMA{}, Layout<Shape<_2, _4>>{}, Tile<_32, _32, _32>{}));
    static_assert(size_v<O::TiledMMA> == NumThreads, "CTA Cooperative MMA Assumed!");

    using S2R = Copy_Atom<SM75_U32x4_LDSM_N, TQKV>;
    using S2R_V = Copy_Atom<SM75_U16x8_LDSM_T, TQKV>;  // specific to V, it needs transpose
  };
};

template <int NumThreads, int HeadSize, int BlockSize, typename TO, typename TQKV, typename TState>
__launch_bounds__(256, 1) __global__ void linear_attention_prefill_kernel(
    TO* __restrict__ output,                 // ["packed_seq", Hqo, dq]
    TState* __restrict__ output_state,       // [num_seqs, Hkv, dv, dk], aka, KV
    TQKV const* __restrict__ q,              // ["packed_seq", Hqo, dq]
    TQKV const* __restrict__ k,              // ["packed_seq", Hkv, dk]
    TQKV const* __restrict__ v,              // ["packed_seq", Hkv, dv]
    TState const* __restrict__ input_state,  // [num_seqs, Hkv, dv, dk], aka, KV initial value
    int64_t const* __restrict__ cu_seqlens,  // [num_seqs + 1], prefix scan of packed length of
                                             // sequences in the batch
    int32_t num_seqs, int32_t num_qo_heads, int32_t num_kv_heads, float scale, float decay,
    float const* per_head_decay, int32_t decay_exponent_offset) {
  using Pipeline =
      LinearAttentionPrefillPipeline<NumThreads, HeadSize, BlockSize, TO, TQKV, TState>;
  Pipeline::attention(output, output_state, q, k, v, input_state, cu_seqlens, num_seqs,
                      num_qo_heads, num_kv_heads, scale, decay, per_head_decay,
                      decay_exponent_offset);
}

template <int NumThreads, int HeadSize, int BlockSize, typename TO, typename TQKV, typename TState>
inline size_t linear_attention_prefill_kernel_smem_size() {
  using Pipeline =
      LinearAttentionPrefillPipeline<NumThreads, HeadSize, BlockSize, TO, TQKV, TState>;
  return sizeof(typename Pipeline::Smem);
}

}  // namespace flat
