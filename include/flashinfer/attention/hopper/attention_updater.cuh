/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_ATTENTION_UPDATER_CUH_
#define FLASHINFER_ATTENTION_HOPPER_ATTENTION_UPDATER_CUH_

#include <cute/tensor.hpp>
#include <cutlass/detail/helper_macros.hpp>

#include "flashinfer/attention/hopper/utils.cuh"

namespace flashinfer {

using namespace cute;

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(T const& x, T const& y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
  // This is slightly faster
  __device__ __forceinline__ float operator()(float const& x, float const& y) { return max(x, y); }
};

template <typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(T const& x, T const& y) { return x + y; }
};

template <int THREADS>
struct Allreduce {
  static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator& op) {
    constexpr int OFFSET = THREADS / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return Allreduce<OFFSET>::run(x, op);
  }
};

template <>
struct Allreduce<2> {
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator& op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }
};

template <bool init, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const& tensor,
                                               Tensor<Engine1, Layout1>& summary, Operator& op) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); mi++) {
    summary(mi) = init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
#pragma unroll
    for (int ni = 1; ni < size<1>(tensor); ni++) {
      summary(mi) = op(summary(mi), tensor(mi, ni));
    }
  }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0>& dst,
                                                Tensor<Engine1, Layout1>& src, Operator& op) {
  CUTE_STATIC_ASSERT_V(size(dst) == size(src));
#pragma unroll
  for (int i = 0; i < size(dst); i++) {
    dst(i) = Allreduce<4>::run(src(i), op);
  }
}

template <bool init, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor,
                                        Tensor<Engine1, Layout1>& summary, Operator& op) {
  thread_reduce_<init>(tensor, summary, op);
  quad_allreduce_(summary, summary, op);
}

template <bool init, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor,
                                           Tensor<Engine1, Layout1>& max) {
  MaxOp<float> max_op;
  reduce_<init>(tensor, max, max_op);
}

template <bool init, bool warp_reduce = true, typename Engine0, typename Layout0, typename Engine1,
          typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor,
                                           Tensor<Engine1, Layout1>& sum) {
  SumOp<float> sum_op;
  thread_reduce_<init>(tensor, sum, sum_op);
  if constexpr (warp_reduce) {
    quad_allreduce_(sum, sum, sum_op);
  }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void apply_exp2(Tensor<Engine0, Layout0>& tensor,
                                           Tensor<Engine1, Layout1> const& max) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    auto row_max = max(mi);
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      tensor(mi, ni) = exp2f(tensor(mi, ni) - row_max);
    }
  }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0>& tensor,
                                                 Tensor<Engine1, Layout1> const& max,
                                                 const float scale) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    auto row_max = max(mi);
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      // row_max * scale is a constant for each row, so we can use fma here
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - row_max * scale);
    }
  }
}

template <int NUM_ROWS_PER_THREAD>
struct DefaultUpdater {
  using TensorT = decltype(make_tensor<float>(Shape<Int<NUM_ROWS_PER_THREAD>>{}));
  template <typename MainloopParams>
  CUTLASS_DEVICE DefaultUpdater(MainloopParams params) {};

  __forceinline__ __device__ TensorT get_lse() { return TensorT(); }

  template <bool init, typename Tensor0>
  __forceinline__ __device__ void update(Tensor0& acc_s) {
    // NOTE(Zihao): nothing to do here
  };

  template <typename Tensor1>
  __forceinline__ __device__ void finalize(Tensor1& acc_s) {
    // NOTE(Zihao): nothing to do here
  };

  template <typename Tensor1>
  __forceinline__ __device__ void rescale_o(Tensor1& acc_o) {
    // NOTE(Zihao): nothing to do here
  };
};

template <int NUM_ROWS_PER_THREAD, bool WITH_SCALE>
struct OnlineSoftmax {
  using TensorT = decltype(make_tensor<float>(Shape<Int<NUM_ROWS_PER_THREAD>>{}));
  TensorT row_max, row_sum, scores_scale;
  float sm_scale_log2;

  template <typename MainloopParams>
  CUTLASS_DEVICE OnlineSoftmax(MainloopParams params) {
    if constexpr (WITH_SCALE) {
      sm_scale_log2 = params.additional_params.sm_scale * math::log2e;
    } else {
      sm_scale_log2 = 0.f;
    }
    clear(scores_scale);
  };

  __forceinline__ __device__ TensorT get_lse() const { return row_sum; }

  template <bool init, typename Tensor0>
  __forceinline__ __device__ void update(Tensor0& acc_s) {
    // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));

    static_assert(decltype(size<0>(scores))::value == NUM_ROWS_PER_THREAD);
    if constexpr (init) {
      reduce_max</*init=*/true>(scores, row_max);
      if constexpr (WITH_SCALE) {
        scale_apply_exp2(scores, row_max, sm_scale_log2);
      } else {
        apply_exp2(scores, row_max);
      }
      reduce_sum</*init=*/true, /*warp_reduce=*/false>(scores, row_sum);
    } else {
      // update row_max
      Tensor scores_max_prev = make_fragment_like(row_max);
      cute::copy(row_max, scores_max_prev);
      reduce_max</*init=*/false>(scores, row_max);
      // update scores_scale and scale row_sum
#pragma unroll
      for (int mi = 0; mi < size(row_max); ++mi) {
        float scores_max_cur = row_max(mi);
        if constexpr (WITH_SCALE) {
          scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * sm_scale_log2);
        } else {
          scores_scale(mi) = exp2f(scores_max_prev(mi) - scores_max_cur);
        }
        row_sum(mi) *= scores_scale(mi);
      }
      // perform exp2 on scores
      if constexpr (WITH_SCALE) {
        scale_apply_exp2(scores, row_max, sm_scale_log2);
      } else {
        apply_exp2(scores, row_max);
      }
      // update row_sum
      reduce_sum</*init=*/false, /*warp_reduce=*/false>(scores, row_sum);
    }
  };

  template <typename Tensor0>
  __forceinline__ __device__ void finalize(Tensor0& acc_s) {
    // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(decltype(size<0>(scores))::value == NUM_ROWS_PER_THREAD);
    SumOp<float> sum_op;
    quad_allreduce_(row_sum, row_sum, sum_op);
#pragma unroll
    for (int mi = 0; mi < size(row_max); ++mi) {
      float sum = row_sum(mi);
      float inv_sum = 1.f / sum;
      scores_scale(mi) = inv_sum;
      if constexpr (WITH_SCALE) {
        row_sum(mi) = row_max(mi) * sm_scale_log2 + math::ptx_log2(sum);
      } else {
        row_sum(mi) = row_max(mi) + math::ptx_log2(sum);
      }
    }
  };

  template <typename Tensor1>
  __forceinline__ __device__ void rescale_o(Tensor1& acc_o) {
    // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
    static_assert(decltype(size<0>(acc_o_rowcol))::value == NUM_ROWS_PER_THREAD);
#pragma unroll
    for (int mi = 0; mi < size(row_max); ++mi) {
#pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= scores_scale(mi);
      }
    }
  };
};

template <int NUM_ROWS_PER_THREAD>
using OnlineSoftmaxWithScale = OnlineSoftmax<NUM_ROWS_PER_THREAD, /*WITH_SCALE=*/true>;

template <int NUM_ROWS_PER_THREAD>
using OnlineSoftmaxWithoutScale = OnlineSoftmax<NUM_ROWS_PER_THREAD, /*WITH_SCALE=*/false>;

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_ATTENTION_UPDATER_CUH_
