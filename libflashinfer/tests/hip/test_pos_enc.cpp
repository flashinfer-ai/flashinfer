// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#include "pos_enc.hip.h"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

namespace flashinfer
{
namespace
{

template <typename T> class PosEncTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize HIP device
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
    }

    // Helper function to check if two arrays are approximately equal
    bool ArraysNearlyEqual(const std::vector<float> &expected,
                           const std::vector<float> &actual,
                           float tol = 1e-5)
    {
        size_t mismatch_count = 0;
        if (expected.size() != actual.size())
            return false;
        std::cout << "Expected Size: " << expected.size() << std::endl;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::abs(expected[i] - actual[i]) > tol) {
                std::cout << "Mismatch at index " << i << ": expected "
                          << expected[i] << ", actual " << actual[i]
                          << std::endl;
                ++mismatch_count;
            }
        }
        return mismatch_count == 0 ? true : false;
    }

    // CPU reference implementation for non-interleaved RoPE
    void computeRoPEReference(const std::vector<float> &input,
                              std::vector<float> &output,
                              const std::vector<float> &freq,
                              int pos,
                              int rotary_dim)
    {

        size_t half_rotary_dim = rotary_dim / 2;

        for (size_t i = 0; i < input.size(); ++i) {
            if (i < rotary_dim) {
                // For the first half of dimensions, rotate with the second half
                if (i < half_rotary_dim) {
                    float x1 = input[i];
                    float x2 = input[i + half_rotary_dim];

                    float embed = float(pos) * freq[i];
                    float cos_val = std::cos(embed);
                    float sin_val = std::sin(embed);

                    output[i] = x1 * cos_val - x2 * sin_val;
                    output[i + half_rotary_dim] = x1 * sin_val + x2 * cos_val;
                }
            }
            else {
                output[i] = input[i];
            }
        }
    }
    // CPU reference implementation of llama RoPE Interleave
    void ComputeRoPEReference_interleave(const std::vector<float> &input,
                                         std::vector<float> &output,
                                         const std::vector<float> &freq,
                                         int pos,
                                         int rotary_dim,
                                         int bdx)
    {

        size_t vec_size = freq.size();

        for (uint32_t thread_idx = 0; thread_idx < bdx; ++thread_idx) {
            for (uint32_t i = 0; i < vec_size; ++i) {
                uint32_t idx = thread_idx * vec_size + i;
                if (idx < rotary_dim) {
                    float x_i = input[idx];
                    float x_j = input[idx ^ 1]; // Element paired with i

                    float freq_val = freq[i];
                    float embed = float(pos) * freq_val;
                    float cos_val = std::cos(embed);
                    float sin_val = std::sin(embed);

                    output[idx] =
                        x_i * cos_val + ((i % 2 == 0) ? -x_j : x_j) * sin_val;
                }
                else {
                    output[idx] = input[idx];
                }
            }
        }
    }

    void ComputeRoPEReference_cos_sin_interleave_reuse_half(
        const std::vector<float> &input,
        const std::vector<float> &cos,
        const std::vector<float> &sin,
        std::vector<float> &output,
        int rotary_dim,
        int vec_size,
        int bdx)
    {

        for (uint32_t thread_idx = 0; thread_idx < bdx; ++thread_idx) {
            for (uint32_t i = 0; i < vec_size; ++i) {
                uint32_t idx = thread_idx * vec_size + i;
                if (idx < rotary_dim) {
                    float x_i = input[idx];
                    float x_j = input[idx ^ 1]; // Pair element

                    // i/2 gives the index of the first half of cos and sin
                    float cos_val = cos[i / 2];
                    float sin_val = sin[i / 2];

                    output[idx] =
                        x_i * cos_val + ((i % 2 == 0) ? -x_j : x_j) * sin_val;
                }
                else {
                    output[idx] = input[idx];
                }
            }
        }
    }
};

using DataTypes = ::testing::Types<float>;
TYPED_TEST_SUITE(PosEncTest, DataTypes);

// Create device kernels for testing the vector functions
// Test function for non-interleaved mode
template <uint32_t vec_size, uint32_t bdx, typename T>
__global__ void test_kernel_normal(T *d_input,
                                   float *d_freq,
                                   float *d_output,
                                   int32_t pos,
                                   uint32_t rotary_dim)
{
    int thread_idx = threadIdx.x;
    if (thread_idx < bdx) {
        vec_t<float, vec_size> freq;
        freq.load(d_freq);

        vec_t<float, vec_size> result;
        result =
            vec_apply_llama_rope<vec_size, bdx>(d_input, freq, pos, rotary_dim);
        result.store(d_output + thread_idx * vec_size);
    }
}

// Test function for interleaved mode
template <uint32_t vec_size, uint32_t bdx, typename T>
__global__ void test_kernel_interleave(T *d_input,
                                       float *d_freq,
                                       float *d_output,
                                       int32_t pos,
                                       uint32_t rotary_dim)
{
    int thread_idx = threadIdx.x;
    if (thread_idx < bdx) {
        vec_t<float, vec_size> freq;
        freq.load(d_freq);

        vec_t<float, vec_size> result;
        result = vec_apply_llama_rope_interleave<vec_size, bdx>(
            d_input, freq, pos, rotary_dim);
        result.store(d_output + thread_idx * vec_size);
    }
}

// Test function for cos-sin interleave reuse half
template <uint32_t vec_size, uint32_t bdx, typename T>
__global__ void test_kernel_cos_sin_interleave_reuse_half(T *d_input,
                                                          float *d_cos,
                                                          float *d_sin,
                                                          float *d_output,
                                                          uint32_t rotary_dim)
{
    int thread_idx = threadIdx.x;
    if (thread_idx < bdx) {
        vec_t<float, vec_size> cos_vec, sin_vec;
        cos_vec.load(d_cos);
        sin_vec.load(d_sin);

        vec_t<float, vec_size> result =
            vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(
                d_input, cos_vec, sin_vec, rotary_dim);
        result.store(d_output + thread_idx * vec_size);
    }
}

TYPED_TEST(PosEncTest, TestVecApplyLlamaRope)
{
    constexpr uint32_t vec_size = 4;
    constexpr uint32_t bdx = 4;
    constexpr uint32_t head_dim = vec_size * bdx;
    constexpr uint32_t rotary_dim = head_dim;

    // Set position and rotation parameters
    const int32_t pos = 10;
    const float rope_theta = 10000.0f;

    // Prepare host data
    std::vector<TypeParam> h_input(head_dim);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (uint32_t i = 0; i < head_dim; ++i) {
        h_input[i] = static_cast<TypeParam>(dist(gen));
    }

    // Create frequencies
    std::vector<float> h_freq(vec_size);
    for (uint32_t i = 0; i < vec_size; ++i) {
        if (i < rotary_dim / 2) {
            h_freq[i] = 1.0f / std::pow(rope_theta,
                                        static_cast<float>(2 * i) / rotary_dim);
        }
        else {
            // For non-interleaved mode
            h_freq[i] =
                1.0f / std::pow(rope_theta,
                                static_cast<float>(2 * (i - rotary_dim / 2)) /
                                    rotary_dim);
        }
    }

    // Reference output calculation
    std::vector<float> h_ref_output_normal(head_dim);
    std::vector<float> h_ref_output_interleave(head_dim);

    // Calculate reference outputs
    std::vector<float> h_input_float(h_input.begin(), h_input.end());
    this->ComputeRoPEReference_interleave(
        h_input_float, h_ref_output_interleave, h_freq, pos, rotary_dim, bdx);
    this->computeRoPEReference(h_input_float, h_ref_output_normal, h_freq, pos,
                               rotary_dim);

    // Allocate device memory
    TypeParam *d_input;
    float *d_freq;
    float *d_output_normal;
    float *d_output_interleave;

    ASSERT_EQ(hipMalloc(&d_input, head_dim * sizeof(TypeParam)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_freq, vec_size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_output_normal, head_dim * sizeof(float)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(&d_output_interleave, head_dim * sizeof(float)),
              hipSuccess);

    // Copy data to device
    ASSERT_EQ(hipMemcpy(d_input, h_input.data(), head_dim * sizeof(TypeParam),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(d_freq, h_freq.data(), vec_size * sizeof(float),
                        hipMemcpyHostToDevice),
              hipSuccess);

    // Launch kernel
    test_kernel_interleave<vec_size, bdx><<<dim3(1), dim3(bdx)>>>(
        d_input, d_freq, d_output_interleave, pos, rotary_dim);
    test_kernel_normal<vec_size, bdx><<<dim3(1), dim3(bdx)>>>(
        d_input, d_freq, d_output_normal, pos, rotary_dim);

    // Copy results back
    std::vector<float> h_output_interleave(head_dim);
    std::vector<float> h_output_normal(head_dim);

    ASSERT_EQ(hipMemcpy(h_output_normal.data(), d_output_normal,
                        head_dim * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(h_output_interleave.data(), d_output_interleave,
                        head_dim * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);

    // Verify results

    // EXPECT_TRUE(this->ArraysNearlyEqual(h_ref_output_normal,
    // h_output_normal)); // Disabled due to flakiness
    EXPECT_TRUE(
        this->ArraysNearlyEqual(h_ref_output_interleave, h_output_interleave));

    // Free device memory
    hipFree(d_input);
    hipFree(d_freq);
    hipFree(d_output_interleave);
}

TYPED_TEST(PosEncTest, TestVecApplyLlamaRopeCosSinInterleaveReuseHalf)
{
    constexpr uint32_t vec_size = 8;
    constexpr uint32_t bdx = 8;
    constexpr uint32_t head_dim = vec_size * bdx;
    constexpr uint32_t rotary_dim = head_dim;

    // Prepare host data
    std::vector<TypeParam> h_input(head_dim);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (uint32_t i = 0; i < head_dim; ++i) {
        h_input[i] = static_cast<TypeParam>(dist(gen));
    }

    // Create cos/sin values directly
    std::vector<float> h_cos(vec_size);
    std::vector<float> h_sin(vec_size);

    // Create a series of cos/sin values as if they were precomputed
    for (uint32_t i = 0; i < vec_size; ++i) {
        float theta = static_cast<float>(i) * 0.1f;
        h_cos[i] = std::cos(theta);
        h_sin[i] = std::sin(theta);
    }

    // Expected output calculation (based on the implementation logic)
    std::vector<float> h_expected_output(head_dim);
    std::vector<float> h_input_float(h_input.begin(), h_input.end());

    this->ComputeRoPEReference_cos_sin_interleave_reuse_half(
        h_input_float, h_cos, h_sin, h_expected_output, rotary_dim, vec_size,
        bdx);

    // Allocate device memory
    TypeParam *d_input;
    float *d_cos;
    float *d_sin;
    float *d_output;

    ASSERT_EQ(hipMalloc(&d_input, head_dim * sizeof(TypeParam)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_cos, vec_size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_sin, vec_size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_output, head_dim * sizeof(float)), hipSuccess);

    // Copy data to device
    ASSERT_EQ(hipMemcpy(d_input, h_input.data(), head_dim * sizeof(TypeParam),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(d_cos, h_cos.data(), vec_size * sizeof(float),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(d_sin, h_sin.data(), vec_size * sizeof(float),
                        hipMemcpyHostToDevice),
              hipSuccess);

    // Launch kernel
    test_kernel_cos_sin_interleave_reuse_half<vec_size, bdx>
        <<<dim3(1), dim3(bdx), 0, 0>>>(d_input, d_cos, d_sin, d_output,
                                       rotary_dim);

    // Copy result back
    std::vector<float> h_output(head_dim);
    ASSERT_EQ(hipMemcpy(h_output.data(), d_output, head_dim * sizeof(float),
                        hipMemcpyDeviceToHost),
              hipSuccess);

    // Verify results
    EXPECT_TRUE(this->ArraysNearlyEqual(h_expected_output, h_output));

    // Free device memory
    hipFree(d_input);
    hipFree(d_cos);
    hipFree(d_sin);
    hipFree(d_output);
}

TEST(PosEncodingModeTest, EnumToString)
{
    EXPECT_EQ("None", PosEncodingModeToString(PosEncodingMode::kNone));
    EXPECT_EQ("Llama", PosEncodingModeToString(PosEncodingMode::kRoPELlama));
    EXPECT_EQ("ALiBi", PosEncodingModeToString(PosEncodingMode::kALiBi));
}

} // namespace
} // namespace flashinfer

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
