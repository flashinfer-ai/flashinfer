#ifndef FLASHINFER_ROPE_CUH_
#define FLASHINFER_ROPE_CUH_

#include <string>

namespace flashinfer {

/*!
 * \brief An enumeration class that defines different modes for applying RoPE
 * (Rotary Positional Embeddings).
 */
enum class RotaryMode {
  // No rotary positional embeddings
  kNone = 0U,
  // Apply Llama-style rope.
  kLlama = 1U,
};

/*!
 * \brief Convert RotaryMode to string
 * \param rotary_mode A RotaryMode value
 */
inline std::string RotaryModeToString(const RotaryMode &rotary_mode) {
  switch (rotary_mode) {
    case RotaryMode::kNone:
      return "None";
    case RotaryMode::kLlama:
      return "Llama";
    default:
      return "Unknown";
  }
}

}  // namespace flashinfer

#endif  // FLASHINFER_ROPE_CUH_