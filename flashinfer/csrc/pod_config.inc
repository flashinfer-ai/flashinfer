#pragma once
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/utils.cuh>

#include "aot_default_additional_params.h"
#include "aot_extension_utils.h"

using namespace flashinfer;

#define DISPATCH_context(MASK_MODE_P, MASK_MODE_D, DTypeQ, DTypeKV, HEAD_DIM_QK,     \
            USE_SLIDING_WINDOW_P, USE_SLIDING_WINDOW_D, USE_LOGITS_SOFT_CAP, ...)    \
{                                                                                    \
  DISPATCH_mask_mode(mask_mode_p, MASK_MODE_P, [&] {                                 \
    return DISPATCH_mask_mode(mask_mode_d, MASK_MODE_D, [&] {                        \
      return DISPATCH_PYTORCH_QKV_DTYPE_TO_CTYPE(                                    \
        q_scalar_type, kv_scalar_type, DTypeQ, DTypeKV, [&] {                        \
          using DTypeO = DTypeQ;                                                     \
          constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;                 \
          constexpr bool USE_FP16_QK_REDUCTION = false;                              \
          return DISPATCH_head_dim(head_dim_qk, HEAD_DIM_QK, [&] {                   \
            [[maybe_unused]] constexpr int HEAD_DIM_VO = HEAD_DIM_QK;                \
            return DISPATCH_BOOL(window_left_p > -1, USE_SLIDING_WINDOW_P, [&] {     \
              return DISPATCH_BOOL(window_left_d > -1, USE_SLIDING_WINDOW_D, [&] {   \
                return DISPATCH_BOOL(false, USE_LOGITS_SOFT_CAP, [&] {               \
                  using IdType = int32_t;                                            \
                  using PrefillParams = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>;\
                  using DecodeParams = BatchPrefillPagedParams<DTypeQ,               \
                                              DTypeKV, DTypeO, IdType>;              \
                  __VA_ARGS__();                                                     \
                  return true;                                                       \
                });                                                                  \
              });                                                                    \
            });                                                                      \
          });                                                                        \
        });                                                                          \
    });                                                                              \
  });                                                                                \
}
