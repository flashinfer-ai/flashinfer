/*
 * Copyright (c) 2025 by SageAttention team.
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

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define PREC_SWITCH(PRECTYPE, ...)             \
  [&] {                                        \
    if (PRECTYPE == 1) {                       \
      using kPrecType = cutlass::half_t;       \
      constexpr static bool kSoftFp16 = false; \
      constexpr static bool kHybrid = false;   \
      return __VA_ARGS__();                    \
    } else if (PRECTYPE == 2) {                \
      using kPrecType = cutlass::float_e4m3_t; \
      constexpr static bool kSoftFp16 = false; \
      constexpr static bool kHybrid = false;   \
      return __VA_ARGS__();                    \
    } else if (PRECTYPE == 3) {                \
      using kPrecType = cutlass::float_e4m3_t; \
      constexpr static bool kSoftFp16 = false; \
      constexpr static bool kHybrid = true;    \
      return __VA_ARGS__();                    \
    } else if (PRECTYPE == 4) {                \
      using kPrecType = cutlass::float_e4m3_t; \
      constexpr static bool kSoftFp16 = true;  \
      constexpr static bool kHybrid = false;   \
      return __VA_ARGS__();                    \
    } else {                                   \
      __builtin_unreachable();                 \
    }                                          \
  }()

#define HEADDIM_SWITCH(HEADDIM, ...)        \
  [&] {                                     \
    if (HEADDIM == 64) {                    \
      constexpr static int kHeadSize = 64;  \
      return __VA_ARGS__();                 \
    } else if (HEADDIM == 128) {            \
      constexpr static int kHeadSize = 128; \
      return __VA_ARGS__();                 \
    } else if (HEADDIM == 256) {            \
      constexpr static int kHeadSize = 256; \
      return __VA_ARGS__();                 \
    } else {                                \
      __builtin_unreachable();              \
    }                                       \
  }()

#define SEQLEN_SWITCH(USE_VAR_SEQ_LEN, SEQ_LEN_OUT_OF_BOUND_CHECK, ...) \
  [&] {                                                                 \
    if (!USE_VAR_SEQ_LEN) {                                             \
      if (SEQ_LEN_OUT_OF_BOUND_CHECK) {                                 \
        using kSeqLenTraitsType = FixedSeqLenTraits<true>;              \
        return __VA_ARGS__();                                           \
      } else {                                                          \
        using kSeqLenTraitsType = FixedSeqLenTraits<false>;             \
        return __VA_ARGS__();                                           \
      }                                                                 \
    } else {                                                            \
      using kSeqLenTraitsType = VarSeqLenTraits;                        \
      return __VA_ARGS__();                                             \
    }                                                                   \
  }()
