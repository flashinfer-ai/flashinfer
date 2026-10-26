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

#ifndef CAKE_SM110_XQA_KV_CACHE_LIST_CONTIG_H
#define CAKE_SM110_XQA_KV_CACHE_LIST_CONTIG_H

// KVCacheList record of the `contig` KV-cache layout, shared by each tree kernel TU of this
// package and its tvm-ffi host binding; hand-folded out of the generated TUs pending exporter
// regeneration.

#if !defined(__CUDACC_RTC__)
#include <stddef.h>
#endif
struct __align__(8) KVCacheList {
    void* data;
    const int* sequence_lengths;
    unsigned int capacity;
};
static_assert(sizeof(KVCacheList) == 24, "KVCacheList size");
static_assert(__alignof__(KVCacheList) == 8, "KVCacheList alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(KVCacheList, data) == 0, "KVCacheList.data offset");
static_assert(offsetof(KVCacheList, sequence_lengths) == 8, "KVCacheList.sequence_lengths offset");
static_assert(offsetof(KVCacheList, capacity) == 16, "KVCacheList.capacity offset");
#endif

#endif  // CAKE_SM110_XQA_KV_CACHE_LIST_CONTIG_H
