#pragma once
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/utils.cuh>

#include "aot_default_additional_params.h"
#include "aot_extension_utils.h"

using IdType = int32_t;

#define ADDITIONAL_FUNC_PARAMS BATCH_PREFILL_ADDITIONAL_FUNC_PARAMS
#define ADDITIONAL_PARAMS_SETTER BATCH_PREFILL_ADDITIONAL_PARAMS_SETTER

#define DISPATCH_context(DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO,    \
                         POS_ENCODING_MODE, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP,              \
                         USE_FP16_QK_REDUCTION, AttentionVariant, RaggedParams, PagedParams, ...) \
  {                                                                                               \
    DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {                                                \
      return DISPATCH_PYTORCH_QKV_DTYPE_TO_CTYPE(                                                 \
          q_scalar_type, kv_scalar_type, DTypeQ, DTypeKV, [&] {                                   \
            using DTypeO = DTypeQ;                                                                \
            using RaggedParams = BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeO, IdType>;       \
            using PagedParams = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;         \
            constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;                            \
            constexpr bool USE_FP16_QK_REDUCTION = false;                                         \
            constexpr bool use_custom_mask = MASK_MODE == MaskMode::kCustom;                      \
            return DISPATCH_head_dim(head_dim_qk, HEAD_DIM_QK, [&] {                              \
              [[maybe_unused]] constexpr int HEAD_DIM_VO = HEAD_DIM_QK;                           \
              return DISPATCH_BOOL(window_left > -1, USE_SLIDING_WINDOW, [&] {                    \
                return DISPATCH_BOOL(logits_soft_cap > 0.f, USE_LOGITS_SOFT_CAP, [&] {            \
                  using AttentionVariant =                                                        \
                      DefaultAttention</*use_custom_mask=*/use_custom_mask, USE_SLIDING_WINDOW,   \
                                       USE_LOGITS_SOFT_CAP, /*use_alibi_bias=*/false>;            \
                  __VA_ARGS__();                                                                  \
                  return true;                                                                    \
                });                                                                               \
              });                                                                                 \
            });                                                                                   \
          });                                                                                     \
    });                                                                                           \
  }
