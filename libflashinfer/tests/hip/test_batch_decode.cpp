// SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#include "flashinfer/attention/generic/decode.cuh"
#include "flashinfer/attention/generic/default_decode_params.cuh"
#include "flashinfer/attention/generic/variants.cuh"

#include "../../utils/cpu_reference_hip.h"
#include "../../utils/flashinfer_batch_decode_test_ops.hip.h"
#include "../../utils/utils_hip.h"

#include <type_traits>

#include <gtest/gtest.h>

using namespace flashinfer;

constexpr QKVLayout kv_layout = QKVLayout::kNHD;

template <typename DType>
std::pair<float, bool>
nan_detection_and_accuracy(const std::vector<DType> &o_host,
                           const std::vector<DType> &o_ref,
                           uint64_t batch_size,
                           uint64_t num_qo_heads,
                           uint64_t head_dim)
{

    uint64_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
    bool nan_detected = false;
    uint64_t num_values = batch_size * num_qo_heads * head_dim;
    for (size_t i = 0; i < o_host.size(); ++i) {
        float o_host_value = fi::con::explicit_casting<DType, float>(o_host[i]);
        float o_ref_value = fi::con::explicit_casting<DType, float>(o_ref[i]);
        if (std::isnan(o_host_value) || std::isnan(o_ref_value)) {
            nan_detected = true;
        }
        num_result_errors_atol_1e_3_rtol_1e_3 +=
            (!utils::isclose(o_host_value, o_ref_value, 1e-3, 1e-3));
    }

    float result_accuracy =
        1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / float(num_values);

    return {result_accuracy, nan_detected};
}

template <typename DTypeQO, typename DTypeKV>
void _TestBatchDecodingKernelCorrectness(size_t page_size,
                                         size_t batch_size,
                                         size_t num_qo_heads,
                                         size_t num_kv_heads,
                                         size_t head_dim,
                                         PosEncodingMode pos_encoding_mode)
{

    std::vector<int32_t> seq_lens(batch_size);
    utils::vec_randint_(seq_lens, 1, 1024);
    std::vector<int32_t> append_indptr{0};
    for (size_t i = 0; i < batch_size; ++i) {
        append_indptr.push_back(append_indptr.back() + seq_lens[i]);
    }

    std::vector<DTypeQO> q;
    std::vector<DTypeQO> o_ref;
    std::vector<DTypeKV> k_data;
    std::vector<DTypeKV> v_data;
    std::vector<int32_t> kv_indptr{0};
    std::vector<int32_t> kv_indices;
    std::vector<int32_t> kv_last_page_len;
    size_t page_counter = 0;

    std::vector<std::vector<DTypeKV>> keys, values;
    for (size_t i = 0; i < batch_size; ++i) {
        size_t seq_len = seq_lens[i];
        size_t num_pages = (seq_len + page_size - 1) / page_size;
        size_t last_page_len = (seq_len - 1) % page_size + 1;
        std::vector<DTypeQO> qi(num_qo_heads * head_dim);
        std::vector<DTypeKV> ki(seq_len * num_kv_heads * head_dim),
            vi(seq_len * num_kv_heads * head_dim);
        utils::vec_normal_(qi);
        utils::vec_normal_(ki);
        utils::vec_normal_(vi);

        // compute reference output
        std::vector<DTypeQO> o_ref_i =
            cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
                qi, ki, vi, 1, seq_len, num_qo_heads, num_kv_heads, head_dim,
                false, QKVLayout::kNHD, pos_encoding_mode);
        keys.push_back(ki);
        values.push_back(vi);
        // append new q and o_ref
        q.insert(q.end(), qi.begin(), qi.end());
        o_ref.insert(o_ref.end(), o_ref_i.begin(), o_ref_i.end());
        // append new kv_indptr, kv_indices and kv_last_page_len
        kv_last_page_len.push_back(last_page_len);
        kv_indptr.push_back(kv_indptr.back() + num_pages);
        for (size_t j = 0; j < num_pages; ++j) {
            kv_indices.push_back(page_counter++);
        }
    }

    k_data.resize(page_counter * num_kv_heads * page_size * head_dim);
    v_data.resize(page_counter * num_kv_heads * page_size * head_dim);
    utils::vec_zero_(k_data);
    utils::vec_zero_(v_data);
    assert(q.size() == batch_size * num_qo_heads * head_dim);
    assert(o_ref.size() == batch_size * num_qo_heads * head_dim);

    paged_kv_t<DTypeKV, int32_t> paged_kv_cpu(
        num_kv_heads, page_size, head_dim, batch_size, kv_layout, k_data.data(),
        v_data.data(), kv_indices.data(), kv_indptr.data(),
        kv_last_page_len.data());
    cpu_reference::append_paged_kv_cache<DTypeKV, int32_t>(
        paged_kv_cpu, keys, values, append_indptr);

    DTypeKV *k_data_device;
    DTypeKV *v_data_device;
    int32_t *kv_indptr_device;
    int32_t *kv_indices_device;
    int32_t *kv_last_page_len_device;
    DTypeQO *q_device;
    DTypeQO *o_device;

    hipMalloc(&k_data_device, k_data.size() * sizeof(DTypeKV));
    hipMalloc(&v_data_device, v_data.size() * sizeof(DTypeKV));
    hipMalloc(&kv_indptr_device, kv_indptr.size() * sizeof(int32_t));
    hipMalloc(&kv_indices_device, kv_indices.size() * sizeof(int32_t));
    hipMalloc(&kv_last_page_len_device,
              kv_last_page_len.size() * sizeof(int32_t));
    hipMalloc(&q_device, q.size() * sizeof(DTypeQO));
    hipMalloc(&o_device, o_ref.size() * sizeof(DTypeQO));

    hipMemcpy(k_data_device, k_data.data(), k_data.size() * sizeof(DTypeKV),
              hipMemcpyHostToDevice);
    hipMemcpy(v_data_device, v_data.data(), v_data.size() * sizeof(DTypeKV),
              hipMemcpyHostToDevice);
    hipMemcpy(kv_indptr_device, kv_indptr.data(),
              kv_indptr.size() * sizeof(int32_t), hipMemcpyHostToDevice);
    hipMemcpy(kv_indices_device, kv_indices.data(),
              kv_indices.size() * sizeof(int32_t), hipMemcpyHostToDevice);
    hipMemcpy(kv_last_page_len_device, kv_last_page_len.data(),
              kv_last_page_len.size() * sizeof(int32_t), hipMemcpyHostToDevice);
    hipMemcpy(q_device, q.data(), q.size() * sizeof(DTypeQO),
              hipMemcpyHostToDevice);

    // create paged_kv object
    paged_kv_t<DTypeKV, int32_t> paged_kv(
        num_kv_heads, page_size, head_dim, batch_size, kv_layout, k_data_device,
        v_data_device, kv_indices_device, kv_indptr_device,
        kv_last_page_len_device);

    BatchDecodeHandler handler;

    size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
    char *float_buffer;
    hipMalloc(&float_buffer, float_workspace_size_in_bytes * sizeof(char));

    size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
    char *int_buffer;
    hipMalloc(&int_buffer, int_workspace_size_in_bytes * sizeof(char));

    BatchDecodeHandlerPlan<DTypeQO, DTypeKV, DTypeQO, int32_t>(
        &handler, (void *)float_buffer, float_workspace_size_in_bytes,
        (void *)int_buffer, int_workspace_size_in_bytes, kv_indptr.data(),
        kv_last_page_len.data(), batch_size, num_qo_heads, num_kv_heads,
        head_dim, page_size, pos_encoding_mode);

    hipError_t status =
        BatchDecodeWithPagedKVCacheWrapper<DTypeQO, DTypeKV, DTypeQO, int32_t>(
            &handler, q_device, /*q_rope_offset=*/nullptr, paged_kv, o_device,
            /*lse=*/nullptr, num_qo_heads, pos_encoding_mode);
    EXPECT_EQ(status, hipSuccess)
        << "HIP error: " + std::string(hipGetErrorString(status));

    // compare result
    std::vector<DTypeQO> o_host(o_ref.size());
    hipMemcpy(o_host.data(), o_device, o_ref.size() * sizeof(DTypeQO),
              hipMemcpyDeviceToHost);

    bool is_empty = o_host.empty();
    EXPECT_EQ(is_empty, false) << "Output is empty.";

    auto [result_accuracy, nan_detected] = nan_detection_and_accuracy<DTypeQO>(
        o_host, o_ref, batch_size, num_qo_heads, head_dim);

    std::cout << "page_size=" << page_size << ", num_qo_heads=" << num_qo_heads
              << ", num_kv_heads=" << num_kv_heads
              << ", batch_size=" << batch_size << ", head_dim=" << head_dim
              << ", pos_encoding_mode="
              << PosEncodingModeToString(pos_encoding_mode)
              << ", result accuracy (atol=1e-3, rtol=1e-3): " << result_accuracy
              << std::endl;
    EXPECT_GT(result_accuracy, 0.90) << "Result correctness test failed.";
    EXPECT_EQ(nan_detected, false) << "NaN detected.";

    hipFree(k_data_device);
    hipFree(v_data_device);
    hipFree(kv_indptr_device);
    hipFree(kv_indices_device);
    hipFree(kv_last_page_len_device);
    hipFree(q_device);
    hipFree(o_device);
    hipFree(float_buffer);
    hipFree(int_buffer);
}

template <typename DTypeQO, typename DTypeKV>
void TestBatchDecodeKernelCorrectness()
{
    for (size_t page_size : {1, 3, 7, 16}) {
        for (size_t batch_size : {1, 2, 4, 8}) {
            for (size_t num_qo_heads : {32}) {
                for (size_t num_kv_heads : {32, 8, 4}) {
                    for (size_t head_dim : {64, 128, 256}) {
                        for (size_t pos_encoding_mode : {0U, 1U}) {
                            _TestBatchDecodingKernelCorrectness<DTypeQO,
                                                                DTypeKV>(
                                page_size, batch_size, num_qo_heads,
                                num_kv_heads, head_dim,
                                PosEncodingMode(pos_encoding_mode));
                        }
                    }
                }
            }
        }
    }
}

TEST(FlashInferCorrectnessTest, BatchDecodeKernelCorrectnessTestFP16)
{
    TestBatchDecodeKernelCorrectness<__half, __half>();
}

// Disabled for now - Look at https://github.com/AMD-AIOSS/flashinfer/issues/36
// TEST(FlashInferCorrectnessTest, TestBatchDecodeKernelCorrectnessBF16) {
//   TestBatchDecodeKernelCorrectness<__hip_bfloat16, __hip_bfloat16>();
// }

//***********************************************************************
// The following tests are disabled because we dont support fp8 <-> float
// conversions

// TEST(FlashInferCorrectnessTest, TestBatchDecodeKernelCorrectnessE4M3) {
//   TestBatchDecodeKernelCorrectness<half, __hip_fp8_e4m3_fnuz>();
// }

// TEST(FlashInferCorrectnessTest, TestBatchDecodeKernelCorrectnessE5M2) {
//   TestBatchDecodeKernelCorrectness<half, __hip_fp8_e5m2_fnuz>();
// }
///************************************************************************/

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
