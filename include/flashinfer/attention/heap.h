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
#ifndef FLASHINFER_ATTENTION_HEAP_H
#define FLASHINFER_ATTENTION_HEAP_H

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace flashinfer {

/*!
 * \brief Heap data structure for (index, value) pairs
 * \note minimal element on top
 */
class CTACostHeap {
 public:
  // first: index, second: value
  using Element = std::pair<int, float>;

  CTACostHeap(int capacity) : full_(true), heap_(capacity) {
    for (int i = 0; i < capacity; ++i) {
      heap_[i] = std::make_pair(i, 0.f);
    }
  }

  void insert(const Element& element) {
    if (full_) {
      throw std::runtime_error("Heap is full, cannot insert more elements");
    }
    heap_.back() = element;
    std::push_heap(heap_.begin(), heap_.end(), compare);
    full_ = true;
  }

  Element pop() {
    if (!full_) {
      throw std::runtime_error("Heap is not full, cannot pop element");
    }
    std::pop_heap(heap_.begin(), heap_.end(), compare);
    Element minElement = heap_.back();
    full_ = false;
    return minElement;
  }

 private:
  bool full_;
  // Custom comparator for the min-heap: compare based on 'val' in the pair
  static bool compare(const Element& a, const Element& b) {
    return a.second > b.second;  // create a min-heap based on val
  }

  std::vector<Element> heap_;
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HEAP_H