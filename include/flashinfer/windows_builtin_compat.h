/*
 * Copyright (c) 2026 by FlashInfer team.
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
#pragma once

// Pre-included by the JIT on native Windows builds (nvcc -include). The generated Cake host
// bindings use the GNU checked-arithmetic builtins, which a Visual Studio host compiler does not
// provide. Define them here, for that host only, with the same semantics: store the wrapped result
// in *r and return true on overflow. Operands and result must share one integer type; any other
// combination fails to compile instead of silently changing the result.
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__GNUC__) && defined(__cplusplus)

#include <type_traits>

namespace flashinfer_windows_compat {

template <typename T>
constexpr bool kCheckedArithmeticType = std::is_integral_v<T> && !std::is_same_v<T, bool>;

template <typename A, typename B, typename R>
constexpr void RequireSameIntegerType() {
  static_assert(std::is_same_v<A, R> && std::is_same_v<B, R> && kCheckedArithmeticType<R>,
                "__builtin_*_overflow shim: operands and result must be the same integer type");
}

template <typename A, typename B, typename R>
inline bool AddOverflow(A a, B b, R* r) {
  RequireSameIntegerType<A, B, R>();
  using U = std::make_unsigned_t<R>;
  const U wrapped = static_cast<U>(static_cast<U>(a) + static_cast<U>(b));
  *r = static_cast<R>(wrapped);
  if constexpr (std::is_signed_v<R>) {
    return ((a ^ *r) & (b ^ *r)) < 0;
  } else {
    return wrapped < static_cast<U>(a);
  }
}

template <typename A, typename B, typename R>
inline bool SubOverflow(A a, B b, R* r) {
  RequireSameIntegerType<A, B, R>();
  using U = std::make_unsigned_t<R>;
  const U wrapped = static_cast<U>(static_cast<U>(a) - static_cast<U>(b));
  *r = static_cast<R>(wrapped);
  if constexpr (std::is_signed_v<R>) {
    return ((a ^ b) & (a ^ *r)) < 0;
  } else {
    return static_cast<U>(a) < static_cast<U>(b);
  }
}

template <typename A, typename B, typename R>
inline bool MulOverflow(A a, B b, R* r) {
  RequireSameIntegerType<A, B, R>();
  using U = std::make_unsigned_t<R>;
  const U wrapped = static_cast<U>(static_cast<U>(a) * static_cast<U>(b));
  *r = static_cast<R>(wrapped);
  if (a == 0 || b == 0) return false;
  if constexpr (std::is_signed_v<R>) {
    // min * -1 wraps back to min, and min / -1 below would itself overflow: decide it here.
    constexpr R kMin = static_cast<R>(U{1} << (sizeof(R) * 8 - 1));
    if ((a == R(-1) && b == kMin) || (b == R(-1) && a == kMin)) return true;
  }
  return static_cast<R>(*r / a) != b;
}

}  // namespace flashinfer_windows_compat

#define __builtin_add_overflow(a, b, r) ::flashinfer_windows_compat::AddOverflow((a), (b), (r))
#define __builtin_sub_overflow(a, b, r) ::flashinfer_windows_compat::SubOverflow((a), (b), (r))
#define __builtin_mul_overflow(a, b, r) ::flashinfer_windows_compat::MulOverflow((a), (b), (r))

#endif  // _MSC_VER && !__clang__ && !__GNUC__ && __cplusplus
