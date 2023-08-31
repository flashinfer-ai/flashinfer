#ifndef FLASHINFER_STATE_CUH_
#define FLASHINFER_STATE_CUH_

#include "vec_dtypes.cuh"

namespace flashinfer {

template <size_t vec_size>
struct state_t {
  float m, d;
  vec_t<float, vec_size> o;

  __device__ __forceinline__ void init() {
    m = -1e5;
    d = 0.f;
    o.fill(0.f);
  }

  __device__ __forceinline__ state_t() { init(); }

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

  __device__ __forceinline__ void merge(const state_t<vec_size> &other) {
    merge(other.m, other.d, other.o);
  }

  __device__ __forceinline__ void merge(float x, const vec_t<float, vec_size> &v) {
    float m_prev = m, d_prev = d;
    m = max(m, x);
    d = d * __expf(m_prev - m) + __expf(x - m);
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] = o[i] * (__expf(m_prev - m) * d_prev / d) + v[i] * (__expf(x - m) / d);
    }
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_STATE_CUH_