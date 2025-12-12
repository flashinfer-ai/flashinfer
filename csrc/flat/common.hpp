#pragma once

#include <cstdio>
#include <string>
#include <stdexcept>

#include "debug.hpp"

#define FLAT_UNUSED_PARAMETER(x) (void)x

#define CHECK(expr, msg)                                      \
  do {                                                        \
    if (!(expr)) {                                            \
      std::string buffer(1024, '\0');                         \
      sprintf(                                                \
          buffer.data(), "Failed to check %s, %s at %s:%d\n", \
          ##expr, msg __FILE__, __LINE__                      \
      );                                                      \
      throw std::runtime_error(buffer.c_str());               \
    }                                                         \
  } while (0)

#define CUDA_CHECK(expr)                                        \
  do {                                                          \
    cudaError_t err = (expr);                                   \
    if (err != cudaSuccess) {                                   \
      std::string buffer(1024, '\0');                           \
      sprintf(                                                  \
          buffer.data(), "CUDA Error: %s, Code: %d at %s:%d\n", \
          cudaGetErrorName(err), err, __FILE__, __LINE__        \
      );                                                        \
      throw std::runtime_error(buffer.c_str());                 \
    }                                                           \
  } while (0)
