// SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#ifndef FLASHINFER_ATTENTION_HEAP_H
#define FLASHINFER_ATTENTION_HEAP_H

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace flashinfer
{

/*!
 * \brief Heap data structure for (index, value) pairs
 * \note minimal element on top
 */
class MinHeap
{
public:
    // first: index, second: cost
    using Element = std::pair<int, float>;

    MinHeap(int capacity) : heap_(capacity)
    {
        for (int i = 0; i < capacity; ++i) {
            heap_[i] = std::make_pair(i, 0.f);
        }
    }

    void insert(const Element &element)
    {
        heap_.push_back(element);
        std::push_heap(heap_.begin(), heap_.end(), compare);
    }

    Element pop()
    {
        std::pop_heap(heap_.begin(), heap_.end(), compare);
        Element minElement = heap_.back();
        heap_.pop_back();
        return minElement;
    }

    std::vector<Element> getHeap() const { return heap_; }

private:
    // Custom comparator for the min-heap: compare based on 'val' in the pair
    static bool compare(const Element &a, const Element &b)
    {
        return a.second > b.second; // create a min-heap based on val
    }

    std::vector<Element> heap_;
};

} // namespace flashinfer

#endif // FLASHINFER_ATTENTION_HEAP_H
