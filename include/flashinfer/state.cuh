#ifndef FLASHINFER_STATE_CUH_
#define FLASHINFER_STATE_CUH_

#include "vec_dtypes.cuh"

namespace flashinfer {

/*!
 * \brief The flashattention state.
 * \tparam vec_size The size of the vector used in o.
 */
template <size_t vec_size>
struct state_t {
  float m;                  /* maximum value of pre-softmax logits */
  float d;                  /* sum of exp(pre-softmax logits - m) */
  vec_t<float, vec_size> o; /* the weighted sum of v: exp(pre-softmax logit - m) * v / d  */

  __device__ __forceinline__ void init() {
    m = -5e4;
    d = 0.f;
    o.fill(0.f);
  }

  __device__ __forceinline__ state_t() { init(); }

  /*!
   * \brief Merge the state with another state.
   * \param other_m The maximum value of pre-softmax logits of the other state.
   * \param other_d The sum of exp(pre-softmax logits - m) of the other state.
   * \param other_o The weighted sum of v of the other state.
   */
  __device__ __forceinline__ void merge(float other_m, float other_d,
                                        const vec_t<float, vec_size> &other_o) {
    float m_prev = m, d_prev = d;
    m = max(m_prev, other_m);
    d = d_prev * __expf(m_prev - m) + other_d * __expf(other_m - m);
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] = o[i] * __expf(m_prev - m) * (d_prev / d) +
             other_o[i] * __expf(other_m - m) * (other_d / d);
    }
  }

  /*!
   * \brief Merge the state with another state.
   * \param other The other state.
   */
  __device__ __forceinline__ void merge(const state_t<vec_size> &other) {
    merge(other.m, other.d, other.o);
  }

  /*!
   * \brief Merge the state with a single pre-softmax logit and value vector.
   * \param x The pre-softmax logit.
   * \param v The value vector.
   */
  __device__ __forceinline__ void merge(float x, const vec_t<float, vec_size> &v) {
    float m_prev = m, d_prev = d;
    m = max(m_prev, x);
    d = d * __expf(m_prev - m) + __expf(x - m);
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] = o[i] * (__expf(m_prev - m) * d_prev / d) + v[i] * (__expf(x - m) / d);
    }
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_STATE_CUH_