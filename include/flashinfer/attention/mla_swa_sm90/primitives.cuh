#pragma once
namespace mla_swa {
// Identical physical-page traversal for every role and configuration. The
// iterator deliberately returns an int, preserving the previous scalar loop.
template <class Config>
struct PageRange {
  struct Iterator {
    int page;
    __host__ __device__ __forceinline__ int operator*() const { return page; }
    __host__ __device__ __forceinline__ void operator++() { ++page; }
    __host__ __device__ __forceinline__ bool operator!=(Iterator other) const {
      // This is a forward range/sentinel, not a general bidirectional
      // iterator. Preserve the old loop's < bound, including empty or
      // inverted ranges, and avoid silently changing its predicate.
      return page < other.page;
    }
  };
  int first, last;
  __host__ __device__ __forceinline__ Iterator begin() const { return {first}; }
  __host__ __device__ __forceinline__ Iterator end() const { return {last}; }
};
}  // namespace mla_swa
