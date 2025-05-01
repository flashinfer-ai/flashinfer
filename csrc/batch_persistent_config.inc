#pragma once
#include <flashinfer/page.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/fastdiv.cuh>
#include <flashinfer/attention/variant_helper.cuh>
#include <flashinfer/attention/mla_params.cuh>
#include <flashinfer/profiler.cuh>

using namespace flashinfer;

#ifdef FLASHINFER_ENABLE_PROFILER
#define ADDITIONAL_FUNC_PARAMS , at::Tensor profiler_buffer
#define ADDITIONAL_PARAMS_SETTER \
  params[0].profiler_buffer = static_cast<uint64_t*>(profiler_buffer.data_ptr());
#else
#define ADDITIONAL_FUNC_PARAMS
#define ADDITIONAL_PARAMS_SETTER
#endif

// using DTypeQ = {{ dtype_q }};
// using DTypeKV = {{ dtype_kv }};
// using DTypeO = {{ dtype_o }};
// using IdType = {{ dtype_idx }};

// constexpr int HEAD_DIM_QK = {{ head_dim_qk }};
// constexpr int HEAD_DIM_VO = {{ head_dim_vo }};
using DTypeQ = half;
using DTypeKV = half;
using DTypeO = half;
using IdType = int;

constexpr int HEAD_DIM_QK = 128;
constexpr int HEAD_DIM_VO = 128;

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct PersistentParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  DTypeQ* q;
  DTypeKV* k;
  DTypeKV* v;
  DTypeO* o;
  DTypeO* partial_o;
  float* partial_lse;
  DTypeO* final_o;
  float* final_lse;

  IdType* batch_indices;
  IdType* q_indptr;
  IdType* kv_indptr;
  IdType* partial_indptr;
  IdType* kv_indices;
  IdType* q_len;
  IdType* kv_len;
  IdType* q_start;
  IdType* kv_start;
  IdType* kv_end;
  IdType* kv_head_idx_arr;
  IdType* work_indptr;

  uint_fastdiv gqa_group_size;
  uint_fastdiv page_size;

  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t k_stride_page;
  uint32_t k_stride_h;
  uint32_t k_stride_n;
  uint32_t v_stride_page;
  uint32_t v_stride_h;
  uint32_t v_stride_n;
  uint32_t o_stride_n;
  uint32_t o_stride_h;

  float sm_scale;

  PROFILER_PARAMS_DECL
};

#define DISPATCH_context(DTypeQ, DTypeKV, DTypeO, IdType, Params, ...) \
  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, { \
    using Params = PersistentParams<DTypeQ, DTypeKV, DTypeO, IdType>; \
    __VA_ARGS__(); \
  })
