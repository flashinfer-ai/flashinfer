// SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#include "../../utils/utils_hip.h"
#include "flashinfer/attention/generic/cascade.cuh"
#include "gpu_iface/conversion_utils.h"
#include "gpu_iface/layout.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

using namespace flashinfer;
constexpr QKVLayout kv_layout = QKVLayout::kHND;

bool is_prime(int x)
{
    for (int i = 2; i < int(std::sqrt(x)); ++i) {
        if (x % i == 0)
            return false;
    }
    return true;
}

template <typename T>
void _TestVariableLengthMergeKernelCorrectness(size_t seq_len,
                                               size_t num_heads,
                                               size_t head_dim,
                                               bool sparse_s)
{
    const uint32_t max_num_index_sets = 512;
    std::vector<int32_t> lengths(seq_len);
    utils::vec_randint_(lengths, 1, max_num_index_sets);
    std::vector<int32_t> indptr{0};
    for (size_t i = 0; i < seq_len; ++i) {
        indptr.push_back(indptr.back() + lengths[i]);
    }
    std::vector<T> V_padded_host(seq_len * max_num_index_sets * num_heads *
                                 head_dim);
    std::vector<T> V_ragged_host(indptr.back() * num_heads * head_dim);
    std::vector<float> S_padded_host(seq_len * max_num_index_sets * num_heads);
    std::vector<float> S_ragged_host(indptr.back() * num_heads);

    utils::vec_normal_(V_ragged_host);
    for (uint32_t j = 0; j < seq_len; ++j) {
        std::copy(V_ragged_host.begin() + indptr[j] * num_heads * head_dim,
                  V_ragged_host.begin() + indptr[j + 1] * num_heads * head_dim,
                  V_padded_host.begin() +
                      j * max_num_index_sets * num_heads * head_dim);
    }
    if (sparse_s) {
        for (uint32_t i = 0; i < max_num_index_sets; ++i) {
            float fill_val = is_prime(i) ? 10 : -10;
            for (uint32_t j = 0; j < seq_len; ++j) {
                if (i < lengths[j]) {
                    std::fill(
                        S_ragged_host.begin() + (indptr[j] + i) * num_heads,
                        S_ragged_host.begin() + (indptr[j] + i + 1) * num_heads,
                        fill_val);
                    std::fill(S_padded_host.begin() +
                                  (j * max_num_index_sets + i) * num_heads,
                              S_padded_host.begin() +
                                  (j * max_num_index_sets + i + 1) * num_heads,
                              fill_val);
                }
                else {
                    std::fill(S_padded_host.begin() +
                                  (j * max_num_index_sets + i) * num_heads,
                              S_padded_host.begin() +
                                  (j * max_num_index_sets + i + 1) * num_heads,
                              -5e4);
                }
            }
        }
    }
    else {
        utils::vec_uniform_(S_ragged_host, -10, 10);
        for (uint32_t j = 0; j < seq_len; ++j) {
            std::copy(S_ragged_host.begin() + indptr[j] * num_heads,
                      S_ragged_host.begin() + indptr[j + 1] * num_heads,
                      S_padded_host.begin() +
                          (j * max_num_index_sets) * num_heads);
            std::fill(S_padded_host.begin() +
                          (j * max_num_index_sets + indptr[j + 1] - indptr[j]) *
                              num_heads,
                      S_padded_host.begin() +
                          (j + 1) * max_num_index_sets * num_heads,
                      -5e4);
        }
    }

    // Allocate device memory using HIP
    T *V_padded_device;
    T *V_ragged_device;
    float *S_padded_device;
    float *S_ragged_device;
    int32_t *indptr_device;
    T *V_merged_0_device;
    T *V_merged_1_device;
    float *S_merged_0_device;
    float *S_merged_1_device;

    hipMalloc(&V_padded_device, V_padded_host.size() * sizeof(T));
    hipMalloc(&V_ragged_device, V_ragged_host.size() * sizeof(T));
    hipMalloc(&S_padded_device, S_padded_host.size() * sizeof(float));
    hipMalloc(&S_ragged_device, S_ragged_host.size() * sizeof(float));
    hipMalloc(&indptr_device, indptr.size() * sizeof(int32_t));
    hipMalloc(&V_merged_0_device, seq_len * num_heads * head_dim * sizeof(T));
    hipMalloc(&V_merged_1_device, seq_len * num_heads * head_dim * sizeof(T));
    hipMalloc(&S_merged_0_device, seq_len * num_heads * sizeof(float));
    hipMalloc(&S_merged_1_device, seq_len * num_heads * sizeof(float));

    // Copy data from host to device
    hipMemcpy(V_padded_device, V_padded_host.data(),
              V_padded_host.size() * sizeof(T), hipMemcpyHostToDevice);
    hipMemcpy(V_ragged_device, V_ragged_host.data(),
              V_ragged_host.size() * sizeof(T), hipMemcpyHostToDevice);
    hipMemcpy(S_padded_device, S_padded_host.data(),
              S_padded_host.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(S_ragged_device, S_ragged_host.data(),
              S_ragged_host.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(indptr_device, indptr.data(), indptr.size() * sizeof(int32_t),
              hipMemcpyHostToDevice);

    // Initialize merged arrays to zero
    hipMemset(V_merged_0_device, 0, seq_len * num_heads * head_dim * sizeof(T));
    hipMemset(V_merged_1_device, 0, seq_len * num_heads * head_dim * sizeof(T));
    hipMemset(S_merged_0_device, 0, seq_len * num_heads * sizeof(float));
    hipMemset(S_merged_1_device, 0, seq_len * num_heads * sizeof(float));

    // Method 0: use MergeStates on padded data
    MergeStates(V_padded_device, S_padded_device, V_merged_0_device,
                S_merged_0_device, max_num_index_sets, seq_len, num_heads,
                head_dim);

    // Method 1: use VariableLengthMergeStates on ragged data
    VariableLengthMergeStates(V_ragged_device, S_ragged_device, indptr_device,
                              V_merged_1_device, S_merged_1_device, seq_len,
                              nullptr, num_heads, head_dim);

    // Allocate host memory for results
    std::vector<T> V_merged_0_host(seq_len * num_heads * head_dim);
    std::vector<T> V_merged_1_host(seq_len * num_heads * head_dim);
    std::vector<float> S_merged_0_host(seq_len * num_heads);
    std::vector<float> S_merged_1_host(seq_len * num_heads);

    // Copy results from device to host
    hipMemcpy(V_merged_0_host.data(), V_merged_0_device,
              seq_len * num_heads * head_dim * sizeof(T),
              hipMemcpyDeviceToHost);
    hipMemcpy(V_merged_1_host.data(), V_merged_1_device,
              seq_len * num_heads * head_dim * sizeof(T),
              hipMemcpyDeviceToHost);
    hipMemcpy(S_merged_0_host.data(), S_merged_0_device,
              seq_len * num_heads * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(S_merged_1_host.data(), S_merged_1_device,
              seq_len * num_heads * sizeof(float), hipMemcpyDeviceToHost);

    // Compare results
    size_t num_V_result_errors_atol_1e_3_rtol_1e_3 = 0,
           num_S_result_errors_atol_1e_3_rtol_1e_3 = 0;
    for (size_t i = 0; i < seq_len * num_heads * head_dim; ++i) {
        float V_merged_0_host_value =
            fi::con::explicit_casting<T, float>(V_merged_0_host[i]);
        float V_merged_1_host_value =
            fi::con::explicit_casting<T, float>(V_merged_1_host[i]);
        EXPECT_FALSE(std::isnan(V_merged_0_host_value))
            << "V_merged_0_host[" << i << "] is nan";
        EXPECT_FALSE(std::isnan(V_merged_1_host_value))
            << "V_merged_1_host[" << i << "] is nan";
        num_V_result_errors_atol_1e_3_rtol_1e_3 += (!utils::isclose(
            V_merged_0_host_value, V_merged_1_host_value, 1e-3, 1e-3));
    }
    for (size_t i = 0; i < seq_len * num_heads; ++i) {
        EXPECT_FALSE(std::isnan(S_merged_0_host[i]))
            << "S_merged_0_host[" << i << "] is nan";
        EXPECT_FALSE(std::isnan(S_merged_1_host[i]))
            << "S_merged_1_host[" << i << "] is nan";
        num_S_result_errors_atol_1e_3_rtol_1e_3 += (!utils::isclose(
            S_merged_0_host[i], S_merged_1_host[i], 1e-3, 1e-3));
    }

    float V_result_accuracy =
        1.0 - float(num_V_result_errors_atol_1e_3_rtol_1e_3) /
                  (seq_len * num_heads * head_dim);
    float S_result_accuracy =
        1.0 -
        float(num_S_result_errors_atol_1e_3_rtol_1e_3) / (seq_len * num_heads);
    std::cout << "seq_len=" << seq_len << ", num_heads=" << num_heads
              << ", head_dim=" << head_dim << ", sparse_s=" << sparse_s
              << ", V accuracy (atol=1e-3, rtol=1e-3)=" << V_result_accuracy
              << ", S accuracy (atol=1e-3, rtol=1e-3)=" << S_result_accuracy
              << std::endl;

    EXPECT_GT(V_result_accuracy, 0.99) << "V result correctness test failed.";
    EXPECT_GT(S_result_accuracy, 0.99) << "S result correctness test failed.";

    // Free device memory
    hipFree(V_padded_device);
    hipFree(V_ragged_device);
    hipFree(S_padded_device);
    hipFree(S_ragged_device);
    hipFree(indptr_device);
    hipFree(V_merged_0_device);
    hipFree(V_merged_1_device);
    hipFree(S_merged_0_device);
    hipFree(S_merged_1_device);
}

template <typename T>
void _TestVariableLengthMergeKernelPaddedCorrectness(size_t max_seq_len,
                                                     size_t seq_len)
{
    ASSERT_LE(seq_len, max_seq_len);

    const size_t num_heads = 4;
    const size_t head_dim = 64;
    const uint32_t max_num_index_sets = 512;

    std::vector<int32_t> lengths(max_seq_len);
    utils::vec_randint_(lengths, 1, max_num_index_sets);
    std::vector<int32_t> indptr(max_seq_len + 1, 0);
    for (size_t i = 0; i < seq_len; ++i) {
        indptr[i + 1] = indptr[i] + lengths[i];
    }

    uint32_t last_indptr = indptr[seq_len];
    std::vector<T> V_ragged_host(last_indptr * num_heads * head_dim);
    std::vector<float> S_ragged_host(last_indptr * num_heads);

    utils::vec_normal_(V_ragged_host);
    utils::vec_uniform_(S_ragged_host, -10, 10);

    // Allocate device memory using HIP
    T *V_ragged_device;
    float *S_ragged_device;
    int32_t *indptr_device;
    T *V_merged_0_device;
    T *V_merged_1_device;
    float *S_merged_0_device;
    float *S_merged_1_device;
    uint32_t *seq_len_device;

    hipMalloc(&V_ragged_device, V_ragged_host.size() * sizeof(T));
    hipMalloc(&S_ragged_device, S_ragged_host.size() * sizeof(float));
    hipMalloc(&indptr_device, indptr.size() * sizeof(int32_t));
    hipMalloc(&V_merged_0_device,
              max_seq_len * num_heads * head_dim * sizeof(T));
    hipMalloc(&V_merged_1_device,
              max_seq_len * num_heads * head_dim * sizeof(T));
    hipMalloc(&S_merged_0_device, max_seq_len * num_heads * sizeof(float));
    hipMalloc(&S_merged_1_device, max_seq_len * num_heads * sizeof(float));
    hipMalloc(&seq_len_device, sizeof(uint32_t));

    // Copy data from host to device
    hipMemcpy(V_ragged_device, V_ragged_host.data(),
              V_ragged_host.size() * sizeof(T), hipMemcpyHostToDevice);
    hipMemcpy(S_ragged_device, S_ragged_host.data(),
              S_ragged_host.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(indptr_device, indptr.data(), indptr.size() * sizeof(int32_t),
              hipMemcpyHostToDevice);
    uint32_t seq_len_value = static_cast<uint32_t>(seq_len);
    hipMemcpy(seq_len_device, &seq_len_value, sizeof(uint32_t),
              hipMemcpyHostToDevice);

    // Initialize merged arrays to zero
    hipMemset(V_merged_0_device, 0,
              max_seq_len * num_heads * head_dim * sizeof(T));
    hipMemset(V_merged_1_device, 0,
              max_seq_len * num_heads * head_dim * sizeof(T));
    hipMemset(S_merged_0_device, 0, max_seq_len * num_heads * sizeof(float));
    hipMemset(S_merged_1_device, 0, max_seq_len * num_heads * sizeof(float));

    // Reference: use VariableLengthMergeStates on the precisely-sized input.
    VariableLengthMergeStates(V_ragged_device, S_ragged_device, indptr_device,
                              V_merged_0_device, S_merged_0_device, seq_len,
                              nullptr, num_heads, head_dim);
    // Expected: use VariableLengthMergeStates on a padded input
    VariableLengthMergeStates(V_ragged_device, S_ragged_device, indptr_device,
                              V_merged_1_device, S_merged_1_device, max_seq_len,
                              seq_len_device, num_heads, head_dim);

    // Allocate host memory for results
    std::vector<T> V_merged_0_host(max_seq_len * num_heads * head_dim);
    std::vector<T> V_merged_1_host(max_seq_len * num_heads * head_dim);
    std::vector<float> S_merged_0_host(max_seq_len * num_heads);
    std::vector<float> S_merged_1_host(max_seq_len * num_heads);

    // Copy results from device to host
    hipMemcpy(V_merged_0_host.data(), V_merged_0_device,
              max_seq_len * num_heads * head_dim * sizeof(T),
              hipMemcpyDeviceToHost);
    hipMemcpy(V_merged_1_host.data(), V_merged_1_device,
              max_seq_len * num_heads * head_dim * sizeof(T),
              hipMemcpyDeviceToHost);
    hipMemcpy(S_merged_0_host.data(), S_merged_0_device,
              max_seq_len * num_heads * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(S_merged_1_host.data(), S_merged_1_device,
              max_seq_len * num_heads * sizeof(float), hipMemcpyDeviceToHost);

    // Compare results
    size_t num_V_result_errors_atol_1e_3_rtol_1e_3 = 0,
           num_S_result_errors_atol_1e_3_rtol_1e_3 = 0;
    for (size_t i = 0; i < seq_len * num_heads * head_dim; ++i) {
        float V_merged_1_host_value =
            fi::con::explicit_casting<T, float>(V_merged_1_host[i]);
        float V_merged_0_host_value =
            fi::con::explicit_casting<T, float>(V_merged_0_host[i]);
        EXPECT_FALSE(std::isnan(V_merged_1_host_value))
            << "V_merged_1_host[" << i << "] is nan";
        num_V_result_errors_atol_1e_3_rtol_1e_3 += (!utils::isclose(
            V_merged_0_host_value, V_merged_1_host_value, 1e-3, 1e-3));
    }
    for (size_t i = 0; i < seq_len * num_heads; ++i) {
        EXPECT_FALSE(std::isnan(float(S_merged_0_host[i])))
            << "S_merged_0_host[" << i << "] is nan";
        EXPECT_FALSE(std::isnan(float(S_merged_1_host[i])))
            << "S_merged_1_host[" << i << "] is nan";
        num_S_result_errors_atol_1e_3_rtol_1e_3 += (!utils::isclose(
            float(S_merged_0_host[i]), float(S_merged_1_host[i]), 1e-3, 1e-3));
    }
    float V_result_accuracy =
        1.0 - float(num_V_result_errors_atol_1e_3_rtol_1e_3) /
                  (seq_len * num_heads * head_dim);
    float S_result_accuracy =
        1.0 -
        float(num_S_result_errors_atol_1e_3_rtol_1e_3) / (seq_len * num_heads);
    std::cout << "seq_len=" << seq_len << ", num_heads=" << num_heads
              << ", head_dim=" << head_dim
              << ", V accuracy (atol=1e-3, rtol=1e-3)=" << V_result_accuracy
              << ", S accuracy (atol=1e-3, rtol=1e-3)=" << S_result_accuracy
              << std::endl;

    EXPECT_GT(V_result_accuracy, 0.99) << "V result correctness test failed.";
    EXPECT_GT(S_result_accuracy, 0.99) << "S result correctness test failed.";

    // Free device memory
    hipFree(V_ragged_device);
    hipFree(S_ragged_device);
    hipFree(indptr_device);
    hipFree(V_merged_0_device);
    hipFree(V_merged_1_device);
    hipFree(S_merged_0_device);
    hipFree(S_merged_1_device);
    hipFree(seq_len_device);
}

template <typename T>
void _TestMergeKernelCorrectness(size_t num_index_sets,
                                 size_t seq_len,
                                 size_t num_heads,
                                 size_t head_dim,
                                 bool sparse_s)
{
    std::vector<T> V_host(seq_len * num_index_sets * num_heads * head_dim);
    std::vector<float> V_host_trans_f32(num_index_sets * seq_len * num_heads *
                                        head_dim);
    std::vector<float> S_host(seq_len * num_index_sets * num_heads);
    std::vector<float> S_host_trans(num_index_sets * seq_len * num_heads);

    utils::vec_normal_(V_host);
    if (sparse_s) {
        for (uint32_t i = 0; i < num_index_sets; ++i) {
            float fill_val = is_prime(i) ? 10 : -10;
            for (uint32_t j = 0; j < seq_len; ++j) {
                for (uint32_t k = 0; k < num_heads; ++k) {
                    S_host[(j * num_index_sets + i) * num_heads + k] = fill_val;
                }
            }
        }
    }
    else {
        utils::vec_uniform_(S_host, -10, 10);
    }

    for (uint32_t i = 0; i < num_index_sets; ++i) {
        for (uint32_t j = 0; j < seq_len; ++j) {
            std::transform(
                V_host.begin() +
                    (j * num_index_sets + i) * num_heads * head_dim,
                V_host.begin() +
                    (j * num_index_sets + i + 1) * num_heads * head_dim,
                V_host_trans_f32.begin() +
                    (i * seq_len + j) * num_heads * head_dim,
                [](T x) { return fi::con::explicit_casting<T, float>(x); });
            std::copy(S_host.begin() + (j * num_index_sets + i) * num_heads,
                      S_host.begin() + (j * num_index_sets + i + 1) * num_heads,
                      S_host_trans.begin() + (i * seq_len + j) * num_heads);
        }
    }

    // Allocate device memory using HIP
    T *V_device;
    float *V_device_trans_f32;
    float *S_device;
    float *S_device_trans;
    float *V_merged_0_device;
    float *S_merged_0_device;
    T *V_merged_1_device;
    float *S_merged_1_device;

    hipMalloc(&V_device, V_host.size() * sizeof(T));
    hipMalloc(&V_device_trans_f32, V_host_trans_f32.size() * sizeof(float));
    hipMalloc(&S_device, S_host.size() * sizeof(float));
    hipMalloc(&S_device_trans, S_host_trans.size() * sizeof(float));
    hipMalloc(&V_merged_0_device,
              seq_len * num_heads * head_dim * sizeof(float));
    hipMalloc(&S_merged_0_device, seq_len * num_heads * sizeof(float));
    hipMalloc(&V_merged_1_device, seq_len * num_heads * head_dim * sizeof(T));
    hipMalloc(&S_merged_1_device, seq_len * num_heads * sizeof(float));

    // Copy data from host to device
    hipMemcpy(V_device, V_host.data(), V_host.size() * sizeof(T),
              hipMemcpyHostToDevice);
    hipMemcpy(V_device_trans_f32, V_host_trans_f32.data(),
              V_host_trans_f32.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(S_device, S_host.data(), S_host.size() * sizeof(float),
              hipMemcpyHostToDevice);
    hipMemcpy(S_device_trans, S_host_trans.data(),
              S_host_trans.size() * sizeof(float), hipMemcpyHostToDevice);

    // Initialize merged arrays to zero
    hipMemset(V_merged_0_device, 0,
              seq_len * num_heads * head_dim * sizeof(float));
    hipMemset(S_merged_0_device, 0, seq_len * num_heads * sizeof(float));
    hipMemset(V_merged_1_device, 0, seq_len * num_heads * head_dim * sizeof(T));
    hipMemset(S_merged_1_device, 0, seq_len * num_heads * sizeof(float));

    if (num_index_sets > 1) {
        // Method 0: use MergeState
        MergeState(V_device_trans_f32, S_device_trans,
                   V_device_trans_f32 + seq_len * num_heads * head_dim,
                   S_device_trans + seq_len * num_heads, V_merged_0_device,
                   S_merged_0_device, seq_len, num_heads, head_dim);
        for (uint i = 2; i < num_index_sets; ++i) {
            MergeStateInPlace(V_merged_0_device, S_merged_0_device,
                              V_device_trans_f32 +
                                  i * seq_len * num_heads * head_dim,
                              S_device_trans + i * seq_len * num_heads, seq_len,
                              num_heads, head_dim);
        }
    }
    else {
        hipMemcpy(V_merged_0_device, V_device,
                  seq_len * num_heads * head_dim * sizeof(T),
                  hipMemcpyDeviceToDevice);
        hipMemcpy(S_merged_0_device, S_device,
                  seq_len * num_heads * sizeof(float), hipMemcpyDeviceToDevice);
    }

    // Method 1: use MergeStates
    MergeStates(V_device, S_device, V_merged_1_device, S_merged_1_device,
                num_index_sets, seq_len, num_heads, head_dim);

    // Allocate host memory for results
    std::vector<float> V_merged_0_host(seq_len * num_heads * head_dim);
    std::vector<T> V_merged_1_host(seq_len * num_heads * head_dim);
    std::vector<float> S_merged_0_host(seq_len * num_heads);
    std::vector<float> S_merged_1_host(seq_len * num_heads);

    // Copy results from device to host
    hipMemcpy(V_merged_0_host.data(), V_merged_0_device,
              seq_len * num_heads * head_dim * sizeof(float),
              hipMemcpyDeviceToHost);
    hipMemcpy(V_merged_1_host.data(), V_merged_1_device,
              seq_len * num_heads * head_dim * sizeof(T),
              hipMemcpyDeviceToHost);
    hipMemcpy(S_merged_0_host.data(), S_merged_0_device,
              seq_len * num_heads * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(S_merged_1_host.data(), S_merged_1_device,
              seq_len * num_heads * sizeof(float), hipMemcpyDeviceToHost);

    // Compare results
    size_t num_V_result_errors_atol_1e_3_rtol_1e_3 = 0,
           num_S_result_errors_atol_1e_3_rtol_1e_3 = 0;
    for (size_t i = 0; i < seq_len * num_heads * head_dim; ++i) {

        float V_merged_0_host_value =
            V_merged_0_host[i]; // V_merged_0_host is already float
        float V_merged_1_host_value =
            fi::con::explicit_casting<T, float>(V_merged_1_host[i]);

        EXPECT_FALSE(std::isnan(V_merged_0_host_value))
            << "V_merged_0_host[" << i << "] is nan";
        EXPECT_FALSE(std::isnan(V_merged_1_host_value))
            << "V_merged_1_host[" << i << "] is nan";
        num_V_result_errors_atol_1e_3_rtol_1e_3 += (!utils::isclose(
            V_merged_0_host_value, V_merged_1_host_value, 1e-3, 1e-3));
    }
    for (size_t i = 0; i < seq_len * num_heads; ++i) {
        EXPECT_FALSE(std::isnan(float(S_merged_0_host[i])))
            << "S_merged_0_host[" << i << "] is nan";
        EXPECT_FALSE(std::isnan(float(S_merged_1_host[i])))
            << "S_merged_1_host[" << i << "] is nan";
        num_S_result_errors_atol_1e_3_rtol_1e_3 += (!utils::isclose(
            float(S_merged_0_host[i]), float(S_merged_1_host[i]), 1e-3, 1e-3));
    }
    float V_result_accuracy =
        1.0 - float(num_V_result_errors_atol_1e_3_rtol_1e_3) /
                  (seq_len * num_heads * head_dim);
    float S_result_accuracy =
        1.0 -
        float(num_S_result_errors_atol_1e_3_rtol_1e_3) / (seq_len * num_heads);
    std::cout << "num_index_sets=" << num_index_sets << ", seq_len=" << seq_len
              << ", num_heads=" << num_heads << ", head_dim=" << head_dim
              << ", sparse_s=" << sparse_s
              << ", V accuracy (atol=1e-3, rtol=1e-3)=" << V_result_accuracy
              << ", S accuracy (atol=1e-3, rtol=1e-3)=" << S_result_accuracy
              << std::endl;
    EXPECT_GT(V_result_accuracy, 0.99) << "V result correctness test failed.";
    EXPECT_GT(S_result_accuracy, 0.99) << "S result correctness test failed.";

    // Free device memory
    hipFree(V_device);
    hipFree(V_device_trans_f32);
    hipFree(S_device);
    hipFree(S_device_trans);
    hipFree(V_merged_0_device);
    hipFree(S_merged_0_device);
    hipFree(V_merged_1_device);
    hipFree(S_merged_1_device);
}

template <typename T> void TestMergeKernelCorrectness()
{
    for (size_t num_index_sets : {2, 9, 81, 513}) {
        for (size_t seq_len : {4, 16, 77}) {
            for (size_t num_heads : {1, 21, 32}) {
                for (size_t head_dim : {64, 128, 256}) {
                    for (bool sparse_s : {false, true}) {
                        _TestMergeKernelCorrectness<T>(num_index_sets, seq_len,
                                                       num_heads, head_dim,
                                                       sparse_s);
                    }
                }
            }
        }
    }
}

template <typename T> void TestVariableLengthMergeKernelCorrectness()
{
    for (size_t seq_len : {1, 3, 77, 191}) {
        for (size_t num_heads : {1, 4, 32}) {
            for (size_t head_dim : {64, 128, 256}) {
                for (bool sparse_s : {false, true}) {
                    _TestVariableLengthMergeKernelCorrectness<T>(
                        seq_len, num_heads, head_dim, sparse_s);
                }
            }
        }
    }
}

template <typename T> void TestVariableLengthMergeKernelPaddedCorrectness()
{
    _TestVariableLengthMergeKernelPaddedCorrectness<T>(8, 1);
    _TestVariableLengthMergeKernelPaddedCorrectness<T>(128, 77);
}

TEST(FlashInferCorrectnessTest, MergeKernelCorrectnessTestFP16)
{
    TestMergeKernelCorrectness<__half>();
}

TEST(FlashInferCorrectnessTest,
     VariableLengthMergeKernelPaddedCorrectnessTestFP16)
{
    TestVariableLengthMergeKernelPaddedCorrectness<__half>();
}

TEST(FlashInferCorrectnessTest, VariableLengthMergeKernelCorrectnessTestFP16)
{
    TestVariableLengthMergeKernelCorrectness<__half>();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
