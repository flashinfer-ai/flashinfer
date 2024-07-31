/*
 * Copyright (c) 2023 by FlashInfer team.
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
#ifndef FLASHINFER_ALLOCATOR_H_
#define FLASHINFER_ALLOCATOR_H_

#include <memory>
#include <sstream>
#include <stdexcept>

namespace flashinfer {

struct AlignedAllocator {
  void* ptr;
  size_t space;
  AlignedAllocator(void* buf, size_t space) : ptr(buf), space(space) {}
  template <typename T>
  T* aligned_alloc(size_t size, size_t alignment, std::string name) {
    if (std::align(alignment, size, ptr, space)) {
      T* result = reinterpret_cast<T*>(ptr);
      ptr = (char*)ptr + size;
      space -= size;
      return result;
    } else {
      std::ostringstream oss;
      oss << "Failed to allocate memory for " << name << " with size " << size << " and alignment "
          << alignment << " in AlignedAllocator";
      throw std::runtime_error(oss.str());
    }
    return nullptr;
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ALLOCATOR_H_
