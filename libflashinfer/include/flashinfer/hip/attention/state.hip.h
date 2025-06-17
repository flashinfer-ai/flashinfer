// SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#pragma once
#ifndef FLASHINFER_STATE_CUH_
#define FLASHINFER_STATE_CUH_

#include "../math.hip.h"
#include "../vec_dtypes.hip.h"

namespace flashinfer
{

/*!
 * \brief The flashattention state.
 * \tparam vec_size The size of the vector used in o.
 */
template <size_t vec_size> struct state_t
{
    /* the weighted sum of v: exp(pre-softmax logit - m) * v / d  */
    vec_t<float, vec_size> o;
    /* maximum value of pre-softmax logits */
    float m;
    /* sum of exp(pre-softmax logits - m) */
    float d;

    __device__ __forceinline__ void init()
    {
        o.fill(0.f);
        m = -math::inf;
        d = 1.f;
    }

    __device__ __forceinline__ state_t() { init(); }

    __device__ __forceinline__ float get_lse() const
    {
        return m + math::ptx_log2(d);
    }

    /*!
     * \brief Merge the state with another state.
     * \param other_m The maximum value of pre-softmax logits of the other
     * state.
     * \param other_d The sum of exp(pre-softmax logits - m) of the other state.
     * \param other_o The weighted sum of v of the other state.
     */
    __device__ __forceinline__ void
    merge(const vec_t<float, vec_size> &other_o, float other_m, float other_d)
    {
        float m_prev = m, d_prev = d;
        m = max(m_prev, other_m);
        d = d_prev * math::ptx_exp2(m_prev - m) +
            other_d * math::ptx_exp2(other_m - m);
#pragma unroll
        for (size_t i = 0; i < vec_size; ++i) {
            o[i] = o[i] * math::ptx_exp2(m_prev - m) +
                   other_o[i] * math::ptx_exp2(other_m - m);
        }
    }

    /*!
     * \brief Merge the state with another state.
     * \param other The other state.
     */
    __device__ __forceinline__ void merge(const state_t<vec_size> &other)
    {
        merge(other.o, other.m, other.d);
    }

    __device__ __forceinline__ void normalize()
    {
        // only normalize by d when not normalized on the fly
#pragma unroll
        for (size_t i = 0; i < vec_size; ++i) {
            o[i] = __fdividef(o[i], d);
        }
    }
};

} // namespace flashinfer

#endif // FLASHINFER_STATE_CUH_
