/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_STATUS_H_
#define FLASHINFER_STATUS_H_

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>

#include <cstdint>
#include <type_traits>

namespace flashinfer {
enum class ErrorDomain : std::uint8_t { kNone, kCuda, kCutlass };

struct [[nodiscard]] Status {
  ErrorDomain domain{ErrorDomain::kNone};
  std::int32_t code{0};  ///< Raw numeric value.

  // Success constructor (default)
  constexpr Status() noexcept = default;

  //---------------------------------
  // Implicit constructors per domain
  //---------------------------------
  constexpr Status(cudaError_t e) noexcept
      : domain(e == cudaSuccess ? ErrorDomain::kNone : ErrorDomain::kCuda),
        code(static_cast<std::int32_t>(e)) {}

  constexpr Status(cutlass::Status s) noexcept
      : domain(s == cutlass::Status::kSuccess ? ErrorDomain::kNone : ErrorDomain::kCutlass),
        code(static_cast<std::int32_t>(s)) {}

  bool success() const noexcept { return domain == ErrorDomain::kNone; }

  static Status Success() noexcept { return Status{}; }

  std::string error_message() const noexcept { return std::string(error_string(*this)); }
};

// Primary template – no definition.  Specialise for each domain.
template <ErrorDomain>
struct DomainTraits;

// CUDA runtime -------------------------------------------------------------
template <>
struct DomainTraits<ErrorDomain::kCuda> {
  using CodeT = cudaError_t;
  static const char* to_string(CodeT c) noexcept { return cudaGetErrorString(c); }
};

// CUTLASS ------------------------------------------------------------------
template <>
struct DomainTraits<ErrorDomain::kCutlass> {
  using CodeT = cutlass::Status;
  static const char* to_string(CodeT s) noexcept { return cutlassGetStatusString(s); }
};

/// Return human‑readable string for any Status.
inline const char* error_string(Status st) noexcept {
  switch (st.domain) {
    case ErrorDomain::kNone:
      return "Success";
    case ErrorDomain::kCuda: {
      static std::string cuda_error =
          "CUDA Error: " + std::string(DomainTraits<ErrorDomain::kCuda>::to_string(
                               static_cast<DomainTraits<ErrorDomain::kCuda>::CodeT>(st.code)));
      return cuda_error.c_str();
    }
    case ErrorDomain::kCutlass: {
      static std::string cutlass_error =
          "CUTLASS Error: " +
          std::string(DomainTraits<ErrorDomain::kCutlass>::to_string(
              static_cast<DomainTraits<ErrorDomain::kCutlass>::CodeT>(st.code)));
      return cutlass_error.c_str();
    }
    default:
      return "Unknown error domain";
  }
}

};  // namespace flashinfer

#endif  // FLASHINFER_STATUS_H_
