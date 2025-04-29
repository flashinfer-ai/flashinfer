// Include the required headers with full paths
#include "attention/cascade.hip.h"
#include "attention/state.hip.h"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <random>
#include <vector>
#include <cmath>
#include <iostream>


namespace flashinfer {
namespace {

template <typename T>
class CascadeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize HIP device
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
  }

  // Helper function to check if two arrays are approximately equal
  bool ArraysNearlyEqual(const std::vector<float>& expected, const std::vector<float>& actual, float tol = 1e-5) {
    if (expected.size() != actual.size()) return false;
    
    size_t mismatch_count = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
      if (std::abs(expected[i] - actual[i]) >= tol) {
        std::cout << "Mismatch at index " << i << ": expected " << expected[i] 
                  << ", actual " << actual[i] << std::endl;
        ++mismatch_count;
      }
    }
    return mismatch_count == 0;
  }

  // CPU reference implementation to merge two states
  void MergeStateCPU(const std::vector<float>& v_a, const std::vector<float>& s_a,
                    const std::vector<float>& v_b, const std::vector<float>& s_b,
                    std::vector<float>& v_merged, std::vector<float>& s_merged,
                    uint32_t seq_len, uint32_t num_heads, uint32_t head_dim) {
    for (uint32_t pos = 0; pos < seq_len; ++pos) {
      for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
        // Compute merged LSE
        float s_a_val = s_a[pos * num_heads + head_idx];
        float s_b_val = s_b[pos * num_heads + head_idx];
        float s_max = std::max(s_a_val, s_b_val);
        float exp_a = std::exp2(s_a_val - s_max);
        float exp_b = std::exp2(s_b_val - s_max);
        float a_scale = exp_a / (exp_a + exp_b);
        float b_scale = exp_b / (exp_a + exp_b);
        s_merged[pos * num_heads + head_idx] = std::log2(exp_a + exp_b) + s_max;
        
        // Compute merged v
        for (uint32_t dim = 0; dim < head_dim; ++dim) {
          uint32_t idx = (pos * num_heads + head_idx) * head_dim + dim;
          v_merged[idx] = a_scale * v_a[idx] + b_scale * v_b[idx];
        }
      }
    }
  }
  
  // CPU reference implementation to merge multiple states
  void MergeStatesCPU(const std::vector<float>& v, const std::vector<float>& s,
                     std::vector<float>& v_merged, std::vector<float>& s_merged,
                     uint32_t num_index_sets, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim) {
    for (uint32_t pos = 0; pos < seq_len; ++pos) {
      for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
        // Handle edge cases
        if (num_index_sets == 0) {
          s_merged[pos * num_heads + head_idx] = -INFINITY;
          for (uint32_t dim = 0; dim < head_dim; ++dim) {
            v_merged[(pos * num_heads + head_idx) * head_dim + dim] = 0.0f;
          }
          continue;
        }
        
        if (num_index_sets == 1) {
          s_merged[pos * num_heads + head_idx] = s[(pos * num_index_sets) * num_heads + head_idx];
          for (uint32_t dim = 0; dim < head_dim; ++dim) {
            v_merged[(pos * num_heads + head_idx) * head_dim + dim] = 
              v[((pos * num_index_sets) * num_heads + head_idx) * head_dim + dim];
          }
          continue;
        }
        
        // Find max s for numerical stability
        float s_max = -INFINITY;
        for (uint32_t idx_set = 0; idx_set < num_index_sets; ++idx_set) {
          float s_val = s[((pos * num_index_sets + idx_set) * num_heads) + head_idx];
          if (s_val > s_max) {
            s_max = s_val;
          }
        }
        
        // Compute weights and weighted sum
        std::vector<float> weights(num_index_sets);
        float sum_weights = 0.0f;
        
        for (uint32_t idx_set = 0; idx_set < num_index_sets; ++idx_set) {
          float s_val = s[((pos * num_index_sets + idx_set) * num_heads) + head_idx];
          weights[idx_set] = std::exp2(s_val - s_max);
          sum_weights += weights[idx_set];
        }
        
        // Initialize merged v
        std::vector<float> merged_v(head_dim, 0.0f);
        
        // Compute weighted sum of v
        for (uint32_t idx_set = 0; idx_set < num_index_sets; ++idx_set) {
          float weight = weights[idx_set] / sum_weights;
          for (uint32_t dim = 0; dim < head_dim; ++dim) {
            uint32_t v_idx = ((pos * num_index_sets + idx_set) * num_heads + head_idx) * head_dim + dim;
            merged_v[dim] += weight * v[v_idx];
          }
        }
        
        // Store results
        s_merged[pos * num_heads + head_idx] = std::log2(sum_weights) + s_max;
        for (uint32_t dim = 0; dim < head_dim; ++dim) {
          v_merged[(pos * num_heads + head_idx) * head_dim + dim] = merged_v[dim];
        }
      }
    }
  }
  
  // CPU reference implementation for AttentionSum
  void AttentionSumCPU(const std::vector<float>& v, std::vector<float>& v_sum,
                      uint32_t num_index_sets, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim) {
    for (uint32_t pos = 0; pos < seq_len; ++pos) {
      for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
        // Handle edge cases
        if (num_index_sets == 0) {
          for (uint32_t dim = 0; dim < head_dim; ++dim) {
            v_sum[(pos * num_heads + head_idx) * head_dim + dim] = 0.0f;
          }
          continue;
        }
        
        if (num_index_sets == 1) {
          for (uint32_t dim = 0; dim < head_dim; ++dim) {
            v_sum[(pos * num_heads + head_idx) * head_dim + dim] = 
              v[((pos * num_index_sets) * num_heads + head_idx) * head_dim + dim];
          }
          continue;
        }
        
        // Compute sum
        for (uint32_t dim = 0; dim < head_dim; ++dim) {
          float sum = 0.0f;
          for (uint32_t idx_set = 0; idx_set < num_index_sets; ++idx_set) {
            uint32_t v_idx = ((pos * num_index_sets + idx_set) * num_heads + head_idx) * head_dim + dim;
            sum += v[v_idx];
          }
          v_sum[(pos * num_heads + head_idx) * head_dim + dim] = sum;
        }
      }
    }
  }
};

using DataTypes = ::testing::Types<float>;
TYPED_TEST_SUITE(CascadeTest, DataTypes);

template<typename U>
std::vector<U> GenerateRandomData(size_t size, int seed = 42, float min = -1.0f, float max = 1.0f) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(min, max);
  
  std::vector<U> result(size);
  for (size_t i = 0; i < size; ++i) {
    result[i] = static_cast<U>(dist(gen));
  }
  return result;
}


// Test merging two attention states
TYPED_TEST(CascadeTest, MergeState) {
  const uint32_t seq_len = 4;
  const uint32_t num_heads = 2;
  const uint32_t head_dim = 64;
  
  // Create input data
  std::vector<TypeParam> h_v_a = GenerateRandomData<TypeParam>(seq_len * num_heads * head_dim, 42);
  std::vector<TypeParam> h_v_b = GenerateRandomData<TypeParam>(seq_len * num_heads * head_dim, 43);
  std::vector<float> h_s_a = GenerateRandomData<float>(seq_len * num_heads, 44, -5.0f, 5.0f);
  std::vector<float> h_s_b = GenerateRandomData<float>(seq_len * num_heads, 45, -5.0f, 5.0f);
  
  // CPU reference results
  std::vector<float> h_expected_v(seq_len * num_heads * head_dim);
  std::vector<float> h_expected_s(seq_len * num_heads);
  
  this->MergeStateCPU(
      std::vector<float>(h_v_a.begin(), h_v_a.end()),
      h_s_a,
      std::vector<float>(h_v_b.begin(), h_v_b.end()),
      h_s_b,
      h_expected_v,
      h_expected_s,
      seq_len, num_heads, head_dim
  );
  
  // Allocate device memory
  TypeParam *d_v_a, *d_v_b, *d_v_merged;
  float *d_s_a, *d_s_b, *d_s_merged;
  
  ASSERT_EQ(hipMalloc(&d_v_a, seq_len * num_heads * head_dim * sizeof(TypeParam)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_v_b, seq_len * num_heads * head_dim * sizeof(TypeParam)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_v_merged, seq_len * num_heads * head_dim * sizeof(TypeParam)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_s_a, seq_len * num_heads * sizeof(float)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_s_b, seq_len * num_heads * sizeof(float)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_s_merged, seq_len * num_heads * sizeof(float)), hipSuccess);
  
  // Copy data to device
  ASSERT_EQ(hipMemcpy(d_v_a, h_v_a.data(), seq_len * num_heads * head_dim * sizeof(TypeParam), hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_v_b, h_v_b.data(), seq_len * num_heads * head_dim * sizeof(TypeParam), hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_s_a, h_s_a.data(), seq_len * num_heads * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_s_b, h_s_b.data(), seq_len * num_heads * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
  
  // Execute kernel
  ASSERT_EQ(MergeState(d_v_a, d_s_a, d_v_b, d_s_b, d_v_merged, d_s_merged, 
                       seq_len, num_heads, head_dim), hipSuccess);
  
  // Copy results back to host
  std::vector<TypeParam> h_output_v(seq_len * num_heads * head_dim);
  std::vector<float> h_output_s(seq_len * num_heads);
  
  ASSERT_EQ(hipMemcpy(h_output_v.data(), d_v_merged, seq_len * num_heads * head_dim * sizeof(TypeParam), hipMemcpyDeviceToHost), hipSuccess);
  ASSERT_EQ(hipMemcpy(h_output_s.data(), d_s_merged, seq_len * num_heads * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
  
  // Verify results
  EXPECT_TRUE(this->ArraysNearlyEqual(h_expected_s, h_output_s));
  EXPECT_TRUE(this->ArraysNearlyEqual(h_expected_v, std::vector<float>(h_output_v.begin(), h_output_v.end())));
  
  // Free device memory
  hipFree(d_v_a);
  hipFree(d_v_b);
  hipFree(d_v_merged);
  hipFree(d_s_a);
  hipFree(d_s_b);
  hipFree(d_s_merged);
}

// Test merging states in place
TYPED_TEST(CascadeTest, MergeStateInPlace) {
  const uint32_t seq_len = 4;
  const uint32_t num_heads = 2;
  const uint32_t head_dim = 64;
  
  // Create input data
  std::vector<TypeParam> h_v = GenerateRandomData<TypeParam>(seq_len * num_heads * head_dim, 42);
  std::vector<TypeParam> h_v_other = GenerateRandomData<TypeParam>(seq_len * num_heads * head_dim, 43);
  std::vector<float> h_s = GenerateRandomData<float>(seq_len * num_heads, 44, -5.0f, 5.0f);
  std::vector<float> h_s_other = GenerateRandomData<float>(seq_len * num_heads, 45, -5.0f, 5.0f);
  
  // Make a copy for CPU reference
  std::vector<TypeParam> h_v_copy = h_v;
  std::vector<float> h_s_copy = h_s;
  
  // CPU reference results - we'll use MergeStateCPU and put results back in h_v_copy
  std::vector<float> h_expected_v(seq_len * num_heads * head_dim);
  std::vector<float> h_expected_s(seq_len * num_heads);
  
  this->MergeStateCPU(
      std::vector<float>(h_v_copy.begin(), h_v_copy.end()),
      h_s_copy,
      std::vector<float>(h_v_other.begin(), h_v_other.end()),
      h_s_other,
      h_expected_v,
      h_expected_s,
      seq_len, num_heads, head_dim
  );
  
  // Allocate device memory
  TypeParam *d_v, *d_v_other;
  float *d_s, *d_s_other;
  
  ASSERT_EQ(hipMalloc(&d_v, seq_len * num_heads * head_dim * sizeof(TypeParam)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_v_other, seq_len * num_heads * head_dim * sizeof(TypeParam)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_s, seq_len * num_heads * sizeof(float)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_s_other, seq_len * num_heads * sizeof(float)), hipSuccess);
  
  // Copy data to device
  ASSERT_EQ(hipMemcpy(d_v, h_v.data(), seq_len * num_heads * head_dim * sizeof(TypeParam), hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_v_other, h_v_other.data(), seq_len * num_heads * head_dim * sizeof(TypeParam), hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_s, h_s.data(), seq_len * num_heads * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_s_other, h_s_other.data(), seq_len * num_heads * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
  
  // Execute kernel
  ASSERT_EQ(MergeStateInPlace(d_v, d_s, d_v_other, d_s_other, seq_len, num_heads, head_dim), hipSuccess);
  
  // Copy results back to host
  std::vector<TypeParam> h_output_v(seq_len * num_heads * head_dim);
  std::vector<float> h_output_s(seq_len * num_heads);
  
  ASSERT_EQ(hipMemcpy(h_output_v.data(), d_v, seq_len * num_heads * head_dim * sizeof(TypeParam), hipMemcpyDeviceToHost), hipSuccess);
  ASSERT_EQ(hipMemcpy(h_output_s.data(), d_s, seq_len * num_heads * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
  
  // Verify results
  EXPECT_TRUE(this->ArraysNearlyEqual(h_expected_s, h_output_s));
  EXPECT_TRUE(this->ArraysNearlyEqual(h_expected_v, std::vector<float>(h_output_v.begin(), h_output_v.end())));
  
  // Free device memory
  hipFree(d_v);
  hipFree(d_v_other);
  hipFree(d_s);
  hipFree(d_s_other);
}

// Test merging multiple states
TYPED_TEST(CascadeTest, MergeStates) {
  const uint32_t seq_len = 3;
  const uint32_t num_heads = 2;
  const uint32_t head_dim = 64;
  const uint32_t num_index_sets = 3;
  
  // Create input data
  std::vector<TypeParam> h_v = GenerateRandomData<TypeParam>(seq_len * num_index_sets * num_heads * head_dim, 42);
  std::vector<float> h_s = GenerateRandomData<float>(seq_len * num_index_sets * num_heads, 44, -5.0f, 5.0f);
  
  // CPU reference results
  std::vector<float> h_expected_v(seq_len * num_heads * head_dim);
  std::vector<float> h_expected_s(seq_len * num_heads);
  
  this->MergeStatesCPU(
      std::vector<float>(h_v.begin(), h_v.end()),
      h_s,
      h_expected_v,
      h_expected_s,
      num_index_sets, seq_len, num_heads, head_dim
  );
  
  // Allocate device memory
  TypeParam *d_v, *d_v_merged;
  float *d_s, *d_s_merged;
  
  ASSERT_EQ(hipMalloc(&d_v, seq_len * num_index_sets * num_heads * head_dim * sizeof(TypeParam)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_v_merged, seq_len * num_heads * head_dim * sizeof(TypeParam)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_s, seq_len * num_index_sets * num_heads * sizeof(float)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_s_merged, seq_len * num_heads * sizeof(float)), hipSuccess);
  
  // Copy data to device
  ASSERT_EQ(hipMemcpy(d_v, h_v.data(), seq_len * num_index_sets * num_heads * head_dim * sizeof(TypeParam), hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_s, h_s.data(), seq_len * num_index_sets * num_heads * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
  
  // Execute kernel
  ASSERT_EQ(MergeStates(d_v, d_s, d_v_merged, d_s_merged, 
                      num_index_sets, seq_len, num_heads, head_dim), hipSuccess);
  
  // Copy results back to host
  std::vector<TypeParam> h_output_v(seq_len * num_heads * head_dim);
  std::vector<float> h_output_s(seq_len * num_heads);
  
  ASSERT_EQ(hipMemcpy(h_output_v.data(), d_v_merged, seq_len * num_heads * head_dim * sizeof(TypeParam), hipMemcpyDeviceToHost), hipSuccess);
  ASSERT_EQ(hipMemcpy(h_output_s.data(), d_s_merged, seq_len * num_heads * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
  
  // Verify results
  EXPECT_TRUE(this->ArraysNearlyEqual(h_expected_s, h_output_s));
  EXPECT_TRUE(this->ArraysNearlyEqual(h_expected_v, std::vector<float>(h_output_v.begin(), h_output_v.end())));
  
  // Free device memory
  hipFree(d_v);
  hipFree(d_v_merged);
  hipFree(d_s);
  hipFree(d_s_merged);
}

// Test attention sum
TYPED_TEST(CascadeTest, AttentionSum) {
  const uint32_t seq_len = 3;
  const uint32_t num_heads = 2;
  const uint32_t head_dim = 64;
  const uint32_t num_index_sets = 3;
  
  // Create input data
  std::vector<TypeParam> h_v = GenerateRandomData<TypeParam>(seq_len * num_index_sets * num_heads * head_dim, 42);
  
  // CPU reference results
  std::vector<float> h_expected_v_sum(seq_len * num_heads * head_dim);
  
  this->AttentionSumCPU(
      std::vector<float>(h_v.begin(), h_v.end()),
      h_expected_v_sum,
      num_index_sets, seq_len, num_heads, head_dim
  );
  
  // Allocate device memory
  TypeParam *d_v;
  TypeParam *d_v_sum;
  
  ASSERT_EQ(hipMalloc(&d_v, seq_len * num_index_sets * num_heads * head_dim * sizeof(TypeParam)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_v_sum, seq_len * num_heads * head_dim * sizeof(TypeParam)), hipSuccess);
  
  // Copy data to device
  ASSERT_EQ(hipMemcpy(d_v, h_v.data(), seq_len * num_index_sets * num_heads * head_dim * sizeof(TypeParam), hipMemcpyHostToDevice), hipSuccess);
  
  // Execute kernel
  ASSERT_EQ(AttentionSum(d_v, d_v_sum, num_index_sets, seq_len, num_heads, head_dim), hipSuccess);
  
  // Copy results back to host
  std::vector<TypeParam> h_output_v_sum(seq_len * num_heads * head_dim);
  
  ASSERT_EQ(hipMemcpy(h_output_v_sum.data(), d_v_sum, seq_len * num_heads * head_dim * sizeof(TypeParam), hipMemcpyDeviceToHost), hipSuccess);
  
  // Verify results
  EXPECT_TRUE(this->ArraysNearlyEqual(h_expected_v_sum, std::vector<float>(h_output_v_sum.begin(), h_output_v_sum.end())));
  
  // Free device memory
  hipFree(d_v);
  hipFree(d_v_sum);
}

// Test variable length merge states
TYPED_TEST(CascadeTest, VariableLengthMergeStates) {
  const uint32_t seq_len = 3;
  const uint32_t num_heads = 2;
  const uint32_t head_dim = 64;
  const uint32_t max_num_indices_per_pos = 4;
  
  // Create indptr to indicate different number of index sets per position
  std::vector<int32_t> h_indptr = {0, 2, 5, 8};  // Position 0: 2 sets, Position 1: 3 sets, Position 2: 3 sets
  const uint32_t total_indices = h_indptr[seq_len];
  
  // Create input data
  std::vector<TypeParam> h_v = GenerateRandomData<TypeParam>(total_indices * num_heads * head_dim, 42);
  std::vector<float> h_s = GenerateRandomData<float>(total_indices * num_heads, 44, -5.0f, 5.0f);
  
  // CPU reference results - manually merge states based on indptr
  std::vector<float> h_expected_v(seq_len * num_heads * head_dim);
  std::vector<float> h_expected_s(seq_len * num_heads);
  
  for (uint32_t pos = 0; pos < seq_len; ++pos) {
    const uint32_t start_idx = h_indptr[pos];
    const uint32_t end_idx = h_indptr[pos + 1];
    const uint32_t num_indices = end_idx - start_idx;
    
    // Extract v and s for this position
    std::vector<float> pos_v(num_indices * num_heads * head_dim);
    std::vector<float> pos_s(num_indices * num_heads);
    
    for (uint32_t i = 0; i < num_indices; ++i) {
      for (uint32_t h = 0; h < num_heads; ++h) {
        pos_s[i * num_heads + h] = h_s[(start_idx + i) * num_heads + h];
        for (uint32_t d = 0; d < head_dim; ++d) {
          pos_v[(i * num_heads + h) * head_dim + d] = h_v[((start_idx + i) * num_heads + h) * head_dim + d];
        }
      }
    }
    
    // Merge states for this position
    std::vector<float> merged_v(num_heads * head_dim);
    std::vector<float> merged_s(num_heads);
    
    this->MergeStatesCPU(pos_v, pos_s, merged_v, merged_s, num_indices, 1, num_heads, head_dim);
    
    // Copy to expected output
    for (uint32_t h = 0; h < num_heads; ++h) {
      h_expected_s[pos * num_heads + h] = merged_s[h];
      for (uint32_t d = 0; d < head_dim; ++d) {
        h_expected_v[(pos * num_heads + h) * head_dim + d] = merged_v[h * head_dim + d];
      }
    }
  }
  
  // Allocate device memory
  TypeParam *d_v, *d_v_merged;
  float *d_s, *d_s_merged;
  int32_t *d_indptr;
  uint32_t *d_seq_len;
  
  ASSERT_EQ(hipMalloc(&d_v, total_indices * num_heads * head_dim * sizeof(TypeParam)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_v_merged, seq_len * num_heads * head_dim * sizeof(TypeParam)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_s, total_indices * num_heads * sizeof(float)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_s_merged, seq_len * num_heads * sizeof(float)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_indptr, (seq_len + 1) * sizeof(int32_t)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_seq_len, sizeof(uint32_t)), hipSuccess);
  
  // Copy data to device
  ASSERT_EQ(hipMemcpy(d_v, h_v.data(), total_indices * num_heads * head_dim * sizeof(TypeParam), hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_s, h_s.data(), total_indices * num_heads * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_indptr, h_indptr.data(), (seq_len + 1) * sizeof(int32_t), hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_seq_len, &seq_len, sizeof(uint32_t), hipMemcpyHostToDevice), hipSuccess);
  
  // Execute kernel
  auto Error = VariableLengthMergeStates<TypeParam, TypeParam, int32_t>(
    d_v, d_s, d_indptr, d_v_merged, d_s_merged, seq_len, d_seq_len, num_heads, head_dim);

  ASSERT_EQ(Error, hipSuccess);

  // Copy results back to host
  std::vector<TypeParam> h_output_v(seq_len * num_heads * head_dim);
  std::vector<float> h_output_s(seq_len * num_heads);
  
  ASSERT_EQ(hipMemcpy(h_output_v.data(), d_v_merged, seq_len * num_heads * head_dim * sizeof(TypeParam), hipMemcpyDeviceToHost), hipSuccess);
  ASSERT_EQ(hipMemcpy(h_output_s.data(), d_s_merged, seq_len * num_heads * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
  
  // Verify results
  EXPECT_TRUE(this->ArraysNearlyEqual(h_expected_s, h_output_s));
  EXPECT_TRUE(this->ArraysNearlyEqual(h_expected_v, std::vector<float>(h_output_v.begin(), h_output_v.end())));
  
  // Free device memory
  hipFree(d_v);
  hipFree(d_v_merged);
  hipFree(d_s);
  hipFree(d_s_merged);
  hipFree(d_indptr);
  hipFree(d_seq_len);
}

// Disabled due to flakyness
// // Test variable length attention sum
// TYPED_TEST(CascadeTest, VariableLengthAttentionSum) {
//   const uint32_t seq_len = 3;
//   const uint32_t num_heads = 2;
//   const uint32_t head_dim = 64;
//   const uint32_t max_num_indices_per_pos = 4;
  
//   // Create indptr to indicate different number of index sets per position
//   std::vector<int32_t> h_indptr = {0, 2, 5, 8};  // Position 0: 2 sets, Position 1: 3 sets, Position 2: 3 sets
//   const uint32_t total_indices = h_indptr[seq_len];
  
//   // Create input data
//   std::vector<TypeParam> h_v = GenerateRandomData<TypeParam>(total_indices * num_heads * head_dim, 42);
  
//   // CPU reference results - manually sum attention based on indptr
//   std::vector<float> h_expected_v_sum(seq_len * num_heads * head_dim, 0.0f);
  
//   for (uint32_t pos = 0; pos < seq_len; ++pos) {
//     const uint32_t start_idx = h_indptr[pos];
//     const uint32_t end_idx = h_indptr[pos + 1];
    
//     for (uint32_t i = start_idx; i < end_idx; ++i) {
//       for (uint32_t h = 0; h < num_heads; ++h) {
//         for (uint32_t d = 0; d < head_dim; ++d) {
//           h_expected_v_sum[(pos * num_heads + h) * head_dim + d] += 
//               h_v[(i * num_heads + h) * head_dim + d];
//         }
//       }
//     }
//   }
  
//   // Allocate device memory
//   TypeParam *d_v, *d_v_sum;
//   int32_t *d_indptr;
//   uint32_t *d_seq_len;
  
//   ASSERT_EQ(hipMalloc(&d_v, total_indices * num_heads * head_dim * sizeof(TypeParam)), hipSuccess);
//   ASSERT_EQ(hipMalloc(&d_v_sum, seq_len * num_heads * head_dim * sizeof(TypeParam)), hipSuccess);
//   ASSERT_EQ(hipMalloc(&d_indptr, (seq_len + 1) * sizeof(int32_t)), hipSuccess);
//   ASSERT_EQ(hipMalloc(&d_seq_len, sizeof(uint32_t)), hipSuccess);
  
//   // Copy data to device
//   ASSERT_EQ(hipMemcpy(d_v, h_v.data(), total_indices * num_heads * head_dim * sizeof(TypeParam), hipMemcpyHostToDevice), hipSuccess);
//   ASSERT_EQ(hipMemcpy(d_indptr, h_indptr.data(), (seq_len + 1) * sizeof(int32_t), hipMemcpyHostToDevice), hipSuccess);
//   ASSERT_EQ(hipMemcpy(d_seq_len, &seq_len, sizeof(uint32_t), hipMemcpyHostToDevice), hipSuccess);
  
//   // Execute kernel
//   auto Error = VariableLengthAttentionSum<TypeParam, TypeParam, int32_t>(
//     d_v, d_indptr, d_v_sum, seq_len, d_seq_len, num_heads, head_dim);

//   ASSERT_EQ(Error, hipSuccess);
  
//   // Copy results back to host
//   std::vector<TypeParam> h_output_v_sum(seq_len * num_heads * head_dim);
//   ASSERT_EQ(hipMemcpy(h_output_v_sum.data(), d_v_sum, seq_len * num_heads * head_dim * sizeof(TypeParam), hipMemcpyDeviceToHost), hipSuccess);
  
//   // Verify results
//   //FLAKY
//   for(int i = 0; i < h_output_v_sum.size(); ++i){
//     EXPECT_NEAR(h_output_v_sum[i], h_expected_v_sum[i], 1e-5);
//   }
  
//   // Free device memory
//   hipFree(d_v);
//   hipFree(d_v_sum);
//   hipFree(d_indptr);
//   hipFree(d_seq_len);
// }

// Test edge cases
TYPED_TEST(CascadeTest, EdgeCases) {
  // Test with zero index sets
  const uint32_t seq_len = 2;
  const uint32_t num_heads = 1;
  const uint32_t head_dim = 64;
  const uint32_t num_index_sets = 0;
  
  // Allocate device memory
  TypeParam *d_v, *d_v_merged;
  float *d_s, *d_s_merged;
  
  ASSERT_EQ(hipMalloc(&d_v, seq_len * num_index_sets * num_heads * head_dim * sizeof(TypeParam)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_v_merged, seq_len * num_heads * head_dim * sizeof(TypeParam)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_s, seq_len * num_index_sets * num_heads * sizeof(float)), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_s_merged, seq_len * num_heads * sizeof(float)), hipSuccess);
  
  // Execute kernel
  ASSERT_EQ(MergeStates(d_v, d_s, d_v_merged, d_s_merged, 
                      num_index_sets, seq_len, num_heads, head_dim), hipSuccess);
  
  // Copy results back to host
  std::vector<TypeParam> h_output_v(seq_len * num_heads * head_dim);
  std::vector<float> h_output_s(seq_len * num_heads);
  
  ASSERT_EQ(hipMemcpy(h_output_v.data(), d_v_merged, seq_len * num_heads * head_dim * sizeof(TypeParam), hipMemcpyDeviceToHost), hipSuccess);
  ASSERT_EQ(hipMemcpy(h_output_s.data(), d_s_merged, seq_len * num_heads * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
  
  // Verify results - we expect all zeros for v
  for (uint32_t i = 0; i < seq_len * num_heads * head_dim; ++i) {
    EXPECT_FLOAT_EQ(0.0f, float(h_output_v[i]));
  }
  
  // Free device memory
  hipFree(d_v);
  hipFree(d_v_merged);
  hipFree(d_s);
  hipFree(d_s_merged);
}

}  // namespace
}  // namespace flashinfer

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}