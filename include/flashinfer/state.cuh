#ifndef FLASHINFER_STATE_CUH_
#define FLASHINFER_STATE_CUH_

#include "math.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

/*!
 * \brief The flashattention state.
 * \tparam vec_size The size of the vector used in o.
 * \tparam norm_on_the_fly Whether to normalize the state on the fly. If true, the state will be
 *   normalized when merge() is called. If false, the state will be normalized when normalize() is
 *   called.
 */
template <size_t vec_size, bool norm_on_the_fly>
struct state_t {
  vec_t<float, vec_size> o; /* the weighted sum of v: exp(pre-softmax logit - m) * v / d  */
  float m;                  /* maximum value of pre-softmax logits */
  float d;                  /* sum of exp(pre-softmax logits - m) */

  __device__ __forceinline__ void init() {
    o.fill(0.f);
    m = -5e4;
    d = 0.f;
  }

  __device__ __forceinline__ state_t() { init(); }

  /*!
   * \brief Merge the state with another state.
   * \param other_m The maximum value of pre-softmax logits of the other state.
   * \param other_d The sum of exp(pre-softmax logits - m) of the other state.
   * \param other_o The weighted sum of v of the other state.
   */
  __device__ __forceinline__ void merge(const vec_t<float, vec_size> &other_o, float other_m,
                                        float other_d) {
    float m_prev = m, d_prev = d;
    m = max(m_prev, other_m);
    d = d_prev * math::ptx_exp2(m_prev - m) + other_d * math::ptx_exp2(other_m - m);
    if constexpr (norm_on_the_fly) {
#pragma unroll
      for (size_t i = 0; i < vec_size; ++i) {
        o[i] = o[i] * math::ptx_exp2(m_prev - m) * (d_prev / d) +
               other_o[i] * math::ptx_exp2(other_m - m) * (other_d / d);
      }
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size; ++i) {
        o[i] = o[i] * math::ptx_exp2(m_prev - m) + other_o[i] * math::ptx_exp2(other_m - m);
      }
    }
  }

  /*!
   * \brief Merge the state with another state.
   * \param other The other state.
   */
  __device__ __forceinline__ void merge(const state_t<vec_size, norm_on_the_fly> &other) {
    merge(other.o, other.m, other.d);
  }

  /*!
   * \brief Merge the state with a single pre-softmax logit and value vector.
   * \param x The pre-softmax logit.
   * \param v The value vector.
   */
  __device__ __forceinline__ void merge(const vec_t<float, vec_size> &other_o, float x) {
    float m_prev = m, d_prev = d;
    m = max(m_prev, x);
    d = d * math::ptx_exp2(m_prev - m) + math::ptx_exp2(x - m);
    if constexpr (norm_on_the_fly) {
#pragma unroll
      for (size_t i = 0; i < vec_size; ++i) {
        o[i] = o[i] * (math::ptx_exp2(m_prev - m) * d_prev / d) +
               other_o[i] * (math::ptx_exp2(x - m) / d);
      }
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size; ++i) {
        o[i] = o[i] * math::ptx_exp2(m_prev - m) + other_o[i] * math::ptx_exp2(x - m);
      }
    }
  }

  __device__ __forceinline__ void normalize() {
    if constexpr (!norm_on_the_fly) {
      // only normalize by d when not normalized on the fly
#pragma unroll
      for (size_t i = 0; i < vec_size; ++i) {
        o[i] = __fdividef(o[i], d);
      }
    }
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_STATE_CUH_