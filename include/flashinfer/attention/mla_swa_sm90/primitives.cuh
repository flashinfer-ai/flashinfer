#pragma once
namespace mla_swa {
// Shared physical-page traversal. W513 and one-query CTA bounds give at
// most nine pages, so S9 has zero or one page, never a steady-state loop.
template <class Config>
struct PageRange {
  struct Iterator {
    int page, last;
    __host__ __device__ __forceinline__ int operator*() const { return page; }
    __host__ __device__ __forceinline__ void operator++() {
      if constexpr (Config::S == 9)
        page = last;
      else
        ++page;
    }
    __host__ __device__ __forceinline__ bool operator!=(Iterator other) const {
      // This is a forward range/sentinel, not a general bidirectional
      // iterator. Preserve the old loop's < bound, including empty or
      // inverted ranges, and avoid silently changing its predicate.
      return page < other.page;
    }
  };
  int first, last;
  __host__ __device__ __forceinline__ Iterator begin() const { return {first, last}; }
  __host__ __device__ __forceinline__ Iterator end() const { return {last, last}; }
};
}  // namespace mla_swa
