#ifndef FLASHINFER_UTILS_CUH_
#define FLASHINFER_UTILS_CUH_
#include <cuda_runtime.h>

#include "layout.cuh"
#include "rope.cuh"

#define FLASHINFER_CUDA_CALL(func, ...) \
  {                                     \
    cudaError_t e = (func);             \
    if (e != cudaSuccess) {             \
      return e;                         \
    }                                   \
  }

#define SWITCH_LAYOUT(layout, LAYOUT, ...)                                 \
  switch (layout) {                                                        \
    case QKVLayout::kNHD: {                                                \
      constexpr QKVLayout LAYOUT = QKVLayout::kNHD;                        \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case QKVLayout::kHND: {                                                \
      constexpr QKVLayout LAYOUT = QKVLayout::kHND;                        \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    default: {                                                             \
      std::cerr << "Unsupported qkv_layout: " << int(layout) << std::endl; \
      abort();                                                             \
    }                                                                      \
  }

#define SWITCH_HEAD_DIM(head_dim, HEAD_DIM, ...)                      \
  switch (head_dim) {                                                 \
    case 64: {                                                        \
      constexpr size_t HEAD_DIM = 64;                                 \
      __VA_ARGS__                                                     \
      break;                                                          \
    }                                                                 \
    case 128: {                                                       \
      constexpr size_t HEAD_DIM = 128;                                \
      __VA_ARGS__                                                     \
      break;                                                          \
    }                                                                 \
    case 256: {                                                       \
      constexpr size_t HEAD_DIM = 256;                                \
      __VA_ARGS__                                                     \
      break;                                                          \
    }                                                                 \
    default: {                                                        \
      std::cerr << "Unsupported head_dim: " << head_dim << std::endl; \
      abort();                                                        \
    }                                                                 \
  }

#define SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, ...)                        \
  switch (rotary_mode) {                                                         \
    case RotaryMode::kNone: {                                                    \
      constexpr RotaryMode ROTARY_MODE = RotaryMode::kNone;                      \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    case RotaryMode::kLlama: {                                                   \
      constexpr RotaryMode ROTARY_MODE = RotaryMode::kLlama;                     \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    default: {                                                                   \
      std::cerr << "Unsupported rotary_mode: " << int(rotary_mode) << std::endl; \
      abort();                                                                   \
    }                                                                            \
  }

#endif  // FLASHINFER_UTILS_CUH_