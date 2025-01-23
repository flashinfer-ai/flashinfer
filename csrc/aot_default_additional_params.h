#define SINGLE_DECODE_ADDITIONAL_FUNC_PARAMS                                             \
  , std::optional<at::Tensor> maybe_alibi_slopes, float logits_soft_cap, float sm_scale, \
      float rope_rcp_scale, float rope_rcp_theta

#define SINGLE_DECODE_ADDITIONAL_PARAMS_SETTER                                            \
  params.maybe_alibi_slopes =                                                             \
      maybe_alibi_slopes ? static_cast<float*>(maybe_alibi_slopes->data_ptr()) : nullptr; \
  params.logits_soft_cap = logits_soft_cap;                                               \
  params.sm_scale = sm_scale;                                                             \
  params.rope_rcp_scale = rope_rcp_scale;                                                 \
  params.rope_rcp_theta = rope_rcp_theta;

#define SINGLE_PREFILL_ADDITIONAL_FUNC_PARAMS                                                  \
  , std::optional<at::Tensor> maybe_custom_mask, std::optional<at::Tensor> maybe_alibi_slopes, \
      float logits_soft_cap, float sm_scale, float rope_rcp_scale, float rope_rcp_theta

#define SINGLE_PREFILL_ADDITIONAL_PARAMS_SETTER                                           \
  params.maybe_custom_mask =                                                              \
      maybe_custom_mask ? static_cast<uint8_t*>(maybe_custom_mask->data_ptr()) : nullptr; \
  params.maybe_alibi_slopes =                                                             \
      maybe_alibi_slopes ? static_cast<float*>(maybe_alibi_slopes->data_ptr()) : nullptr; \
  params.logits_soft_cap = logits_soft_cap;                                               \
  params.sm_scale = sm_scale;                                                             \
  params.rope_rcp_scale = rope_rcp_scale;                                                 \
  params.rope_rcp_theta = rope_rcp_theta;

#define SINGLE_PREFILL_SM90_ADDITIONAL_FUNC_PARAMS , float logits_soft_cap, float sm_scale

#define SINGLE_PREFILL_SM90_ADDITIONAL_PARAMS_SETTER          \
  params.additional_params.logits_soft_cap = logits_soft_cap; \
  params.additional_params.sm_scale = sm_scale;

#define BATCH_DECODE_ADDITIONAL_FUNC_PARAMS                                              \
  , std::optional<at::Tensor> maybe_alibi_slopes, float logits_soft_cap, float sm_scale, \
      float rope_rcp_scale, float rope_rcp_theta

#define BATCH_DECODE_ADDITIONAL_PARAMS_SETTER                                             \
  params.maybe_alibi_slopes =                                                             \
      maybe_alibi_slopes ? static_cast<float*>(maybe_alibi_slopes->data_ptr()) : nullptr; \
  params.logits_soft_cap = logits_soft_cap;                                               \
  params.sm_scale = sm_scale;                                                             \
  params.rope_rcp_scale = rope_rcp_scale;                                                 \
  params.rope_rcp_theta = rope_rcp_theta;

#define BATCH_PREFILL_ADDITIONAL_FUNC_PARAMS                                                  \
  , std::optional<at::Tensor> maybe_custom_mask, std::optional<at::Tensor> maybe_mask_indptr, \
      std::optional<at::Tensor> maybe_alibi_slopes, float logits_soft_cap, float sm_scale,    \
      float rope_rcp_scale, float rope_rcp_theta

#define BATCH_PREFILL_ADDITIONAL_PARAMS_SETTER                                            \
  params.maybe_custom_mask =                                                              \
      maybe_custom_mask ? static_cast<uint8_t*>(maybe_custom_mask->data_ptr()) : nullptr; \
  params.maybe_mask_indptr =                                                              \
      maybe_mask_indptr ? static_cast<int32_t*>(maybe_mask_indptr->data_ptr()) : nullptr; \
  params.maybe_alibi_slopes =                                                             \
      maybe_alibi_slopes ? static_cast<float*>(maybe_alibi_slopes->data_ptr()) : nullptr; \
  params.logits_soft_cap = logits_soft_cap;                                               \
  params.sm_scale = sm_scale;                                                             \
  params.rope_rcp_scale = rope_rcp_scale;                                                 \
  params.rope_rcp_theta = rope_rcp_theta;

#define BATCH_PREFILL_SM90_ADDITIONAL_FUNC_PARAMS , float logits_soft_cap, float sm_scale

#define BATCH_PREFILL_SM90_ADDITIONAL_PARAMS_SETTER           \
  params.additional_params.logits_soft_cap = logits_soft_cap; \
  params.additional_params.sm_scale = sm_scale;
