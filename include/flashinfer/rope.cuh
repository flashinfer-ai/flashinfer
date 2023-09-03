#ifndef FLASHINFER_ROPE_CUH_
#define FLASHINFER_ROPE_CUH_

#include <string>

#include "vec_dtypes.cuh"

namespace flashinfer {

/*!
 * \brief An enumeration class that defines different modes for applying RoPE
 * (Rotary Positional Embeddings).
 */
enum class RotaryMode {
  // No rotary positional embeddings
  kNone = 0U,
  // Apply rotary positional embeddings to q and all rows in k matrix of
  // kv-cache.
  kApplyRotary = 1U,
};

/*!
 * \brief Convert RotaryMode to string
 * \param rotary_mode A RotaryMode value
 */
inline std::string RotaryModeToString(const RotaryMode &rotary_mode) {
  switch (rotary_mode) {
    case RotaryMode::kNone:
      return "None";
    case RotaryMode::kApplyRotary:
      return "ApplyRotary";
    default:
      return "Unknown";
  }
}

/*!
 * \brief Apply RoPE (Rotary Positional Embeddings) to input[0: head_dim],
 *   return thread-local vector
 * \tparam vec_size A template integer indicates the vector size used
 *   in the kernel
 * \tparam T A template type indicates the input data type
 * \param input A pointer to the start of input data
 * \param rotary_emb A vector of float indicates the thread-local rotary embedding
 * \param offset A integer indicates the offset of the position in RoPE
 */
template <size_t head_dim, size_t vec_size, typename T>
__device__ __forceinline__ vec_t<float, vec_size> apply_rotary(
    const T *input, const vec_t<float, vec_size> &rotary_emb, size_t offset) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(input + threadIdx.x * vec_size);
  permuted_vec.cast_load(input + ((threadIdx.x * vec_size < head_dim / 2)
                                      ? threadIdx.x * vec_size + head_dim / 2
                                      : threadIdx.x * vec_size - head_dim / 2));

#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    float embed = float(offset) * rotary_emb[i];
    float cos, sin;
    __sincosf(embed, &sin, &cos);
    vec[i] = vec[i] * cos +
             ((threadIdx.x * vec_size < head_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin;
  }
  return vec;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ROPE_CUH_