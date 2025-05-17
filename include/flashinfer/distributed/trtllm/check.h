// adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/common/assert.h

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <array>
#include <cstdarg>
#include <cstddef>
#include <memory>   // std::make_unique
#include <sstream>  // std::stringstream
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// tentative: to be compatible with trtllm's assertion, we need to do a paper cut of trtllm's check
// target macro: TLLM_CUDA_CHECK, TLLM_CHECK, TLLM_CHECK_WITH_INFO

#define NEW_TLLM_EXCEPTION(...) \
  trtllm::TllmException(__FILE__, __LINE__, trtllm::fmtstr(__VA_ARGS__).c_str())

namespace trtllm {

// string utils
inline std::string fmtstr(std::string const& s) { return s; }

inline std::string fmtstr(std::string&& s) { return s; }

typedef char* (*fmtstr_allocator)(void* target, size_t count);
void fmtstr_(char const* format, fmtstr_allocator alloc, void* target, va_list args);

#if defined(_MSC_VER)
inline std::string fmtstr(char const* format, ...);
#else
inline std::string fmtstr(char const* format, ...) __attribute__((format(printf, 1, 2)));
#endif

inline std::string fmtstr(char const* format, ...) {
  std::string result;

  va_list args;
  va_start(args, format);
  fmtstr_(
      format,
      [](void* target, size_t count) -> char* {
        if (count <= 0) {
          return nullptr;
        }

        const auto str = static_cast<std::string*>(target);
        str->resize(count);
        return str->data();
      },
      &result, args);
  va_end(args);

  return result;
}

class TllmException : public std::runtime_error {
 public:
  static auto constexpr MAX_FRAMES = 128;

  explicit TllmException(char const* file, std::size_t line, char const* msg);

  ~TllmException() noexcept override;

  [[nodiscard]] std::string getTrace() const;

  static std::string demangle(char const* name);

 private:
  std::array<void*, MAX_FRAMES> mCallstack{};
  int mNbFrames;
};

template <typename T>
void check(T ptr, char const* const func, char const* const file, int const line) {
  if (ptr) {
    throw TllmException(
        file, line,
        fmtstr("[TensorRT-LLM][ERROR] CUDA runtime error in %s: %s", func, _cudaGetErrorEnum(ptr))
            .c_str());
  }
}

[[noreturn]] inline void throwRuntimeError(char const* const file, int const line,
                                           std::string const& info = "") {
  throw TllmException(file, line,
                      fmtstr("[TensorRT-LLM][ERROR] Assertion failed: %s", info.c_str()).c_str());
}

/*
 * Macros compliant with TensorRT coding conventions
 */
#define TLLM_CUDA_CHECK(stat)                 \
  do {                                        \
    check((stat), #stat, __FILE__, __LINE__); \
  } while (0)

#if defined(_WIN32)
#define TLLM_LIKELY(x) (__assume((x) == 1), (x))
#define TLLM_UNLIKELY(x) (__assume((x) == 0), (x))
#else
#define TLLM_LIKELY(x) __builtin_expect((x), 1)
#define TLLM_UNLIKELY(x) __builtin_expect((x), 0)
#endif

#define TLLM_CHECK(val)                                                                            \
  do {                                                                                             \
    TLLM_LIKELY(static_cast<bool>(val)) ? ((void)0) : throwRuntimeError(__FILE__, __LINE__, #val); \
  } while (0)

#define TLLM_CHECK_WITH_INFO(val, info, ...)                                          \
  do {                                                                                \
    TLLM_LIKELY(static_cast<bool>(val))                                               \
    ? ((void)0) : throwRuntimeError(__FILE__, __LINE__, fmtstr(info, ##__VA_ARGS__)); \
  } while (0)

}  // namespace trtllm
