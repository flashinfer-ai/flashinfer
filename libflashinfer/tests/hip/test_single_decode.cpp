// SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#include "flashinfer/attention/generic/decode.cuh"
#include "flashinfer/attention/generic/default_decode_params.cuh"
#include "flashinfer/attention/generic/variants.cuh"

#include "../../utils/cpu_reference_hip.h"
#include "../../utils/utils_hip.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <optional>
#include <type_traits>

#include <gtest/gtest.h>

using namespace flashinfer;

namespace test::ops
{
template <typename DTypeQ, typename DTypeKV, typename DTypeO>
hipError_t SingleDecodeWithKVCache(
    DTypeQ *q,
    DTypeKV *k,
    DTypeKV *v,
    DTypeO *o,
    DTypeO *tmp,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    QKVLayout kv_layout = QKVLayout::kNHD,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    std::optional<float> maybe_sm_scale = std::nullopt,
    float rope_scale = 1.f,
    float rope_theta = 1e4,
    hipStream_t stream = nullptr)
{
    float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
    if (num_qo_heads % num_kv_heads != 0) {
        std::ostringstream err_msg;
        err_msg << "num_qo_heads " << num_qo_heads
                << " is not a multiple of num_kv_heads " << num_kv_heads;
        FLASHINFER_ERROR(err_msg.str());
    }

    DISPATCH_head_dim(
        head_dim, HEAD_DIM,
        {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
            using Params = SingleDecodeParams<DTypeQ, DTypeKV, DTypeO>;
            using AttentionVariant = DefaultAttention<
                /*use_custom_mask=*/false, /*use_sliding_window=*/false,
                /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;
            Params params(q, k, v, o, /*alibi_slopes=*/nullptr, seq_len,
                          num_qo_heads, num_kv_heads, kv_layout, head_dim,
                          /*window_left=*/-1, /*logits_soft_cap=*/0.f, sm_scale,
                          rope_scale, rope_theta);

            SingleDecodeWithKVCacheDispatched<HEAD_DIM, POS_ENCODING_MODE,
                                              AttentionVariant>(params, tmp,
                                                                stream);
        })});
    return hipSuccess;
}
} // namespace test::ops

template <typename DTypeQO, typename DTypeKV>
std::vector<DTypeQO> getCPUReference(const std::vector<DTypeQO> &Q_host,
                                     const std::vector<DTypeKV> &K_host,
                                     const std::vector<DTypeKV> &V_host,
                                     size_t num_qo_heads,
                                     size_t num_kv_heads,
                                     size_t seq_len,
                                     size_t head_dim,
                                     QKVLayout kv_layout,
                                     PosEncodingMode pos_encoding_mode)
{

    return cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
        Q_host, K_host, V_host, 1, seq_len, num_qo_heads, num_kv_heads,
        head_dim, false, kv_layout, pos_encoding_mode);
}

template <typename DType>
std::pair<float, bool>
nan_detection_and_accuracy(const std::vector<DType> &cpu_results,
                           const std::vector<DType> &gpu_results,
                           uint64_t num_qo_heads,
                           uint64_t head_dim)
{

    size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
    bool nan_detected = false;

    for (size_t i = 0; i < num_qo_heads * head_dim; ++i) {
        float cpu_result =
            fi::con::explicit_casting<DType, float>(cpu_results[i]);
        float gpu_result =
            fi::con::explicit_casting<DType, float>(gpu_results[i]);

        if (std::is_same_v<DType, __half>) {
            if (isnan(gpu_result)) {
                nan_detected = true;
            }
            num_result_errors_atol_1e_3_rtol_1e_3 +=
                (!utils::isclose(gpu_result, cpu_result, 1e-2, 1e-2));
        }
    }

    float result_accuracy = 1. - float(num_result_errors_atol_1e_3_rtol_1e_3) /
                                     float(num_qo_heads * head_dim);

    return {result_accuracy, nan_detected};
}

template <typename DTypeQO, typename DTypeKV>
void _TestDecodingKernelCorrectness(size_t num_qo_heads,
                                    size_t num_kv_heads,
                                    size_t seq_len,
                                    size_t head_dim,
                                    QKVLayout kv_layout,
                                    PosEncodingMode pos_encoding_mode)
{

    std::vector<DTypeQO> Q_host(num_qo_heads * head_dim);
    std::vector<DTypeKV> K_host(seq_len * num_kv_heads * head_dim);
    std::vector<DTypeKV> V_host(seq_len * num_kv_heads * head_dim);
    std::vector<DTypeQO> O_host(num_qo_heads * head_dim);

    utils::vec_normal_(Q_host);
    utils::vec_normal_(K_host);
    utils::vec_normal_(V_host);
    utils::vec_zero_(O_host);

    DTypeQO *Q;
    DTypeKV *K;
    DTypeKV *V;
    DTypeQO *O;
    DTypeQO *tmp;

    hipMalloc(&Q, num_qo_heads * head_dim * sizeof(DTypeQO));
    hipMalloc(&K, seq_len * num_kv_heads * head_dim * sizeof(DTypeKV));
    hipMalloc(&V, seq_len * num_kv_heads * head_dim * sizeof(DTypeKV));
    hipMalloc(&O, num_qo_heads * head_dim * sizeof(DTypeQO));
    hipMalloc(&tmp, num_qo_heads * head_dim * sizeof(DTypeQO));

    hipMemcpy(Q, Q_host.data(), num_qo_heads * head_dim * sizeof(DTypeQO),
              hipMemcpyHostToDevice);
    hipMemcpy(K, K_host.data(),
              seq_len * num_kv_heads * head_dim * sizeof(DTypeKV),
              hipMemcpyHostToDevice);
    hipMemcpy(V, V_host.data(),
              seq_len * num_kv_heads * head_dim * sizeof(DTypeKV),
              hipMemcpyHostToDevice);
    hipMemcpy(O, O_host.data(), num_qo_heads * head_dim * sizeof(DTypeQO),
              hipMemcpyHostToDevice);
    hipMemcpy(tmp, O_host.data(), num_qo_heads * head_dim * sizeof(DTypeQO),
              hipMemcpyHostToDevice);

    std::vector<DTypeQO> o_ref_host = getCPUReference<DTypeQO, DTypeKV>(
        Q_host, K_host, V_host, num_qo_heads, num_kv_heads, seq_len, head_dim,
        QKVLayout(kv_layout), PosEncodingMode(pos_encoding_mode));

    hipError_t status =
        test::ops::SingleDecodeWithKVCache<DTypeQO, DTypeKV, DTypeQO>(
            Q, K, V, O, tmp, num_qo_heads, num_kv_heads, seq_len, head_dim,
            kv_layout, pos_encoding_mode);

    if (status != hipSuccess) {
        std::cout
            << "SingleDecodeWithKVCache kernel launch failed, error message: "
            << hipGetErrorString(status) << std::endl;
    }

    std::vector<DTypeQO> o_host(num_qo_heads * head_dim);
    hipMemcpy(o_host.data(), O, num_qo_heads * head_dim * sizeof(DTypeQO),
              hipMemcpyDeviceToHost);

    auto [result_accuracy, nan_detected] =
        nan_detection_and_accuracy(o_ref_host, o_host, num_qo_heads, head_dim);

    std::cout << "num_qo_heads=" << num_qo_heads
              << ", num_kv_heads=" << num_kv_heads << ", seq_len=" << seq_len
              << ", head_dim=" << head_dim
              << ", kv_layout=" << QKVLayoutToString(kv_layout)
              << ", pos_encoding_mode="
              << PosEncodingModeToString(pos_encoding_mode)
              << ", result accuracy (atol=1e-3, rtol=1e-3): " << result_accuracy
              << std::endl;
    EXPECT_GT(result_accuracy, 0.90) << "Result correctness test failed.";
    EXPECT_FALSE(nan_detected) << "NaN detected.";

    hipFree(Q);
    hipFree(K);
    hipFree(V);
    hipFree(O);
    hipFree(tmp);
}

// Potential issue:
// HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access
// memory beyond the largest legal address. code: 0x29 Please look at
// https://github.com/AMD-AIOSS/flashinfer/issues/33
template <typename DTypeQO, typename DTypeKV>
void TestSingleDecodeKernelCorrectness()
{
    for (size_t num_qo_heads : {32}) {
        for (size_t num_kv_heads : {4, 8, 32}) {
            for (size_t seq_len : {64, 128, 256}) {
                for (size_t head_dim : {64, 128, 256}) {
                    for (unsigned int kv_layout : {0U, 1U}) {
                        for (unsigned int pos_encoding_mode : {0U, 1U}) {
                            _TestDecodingKernelCorrectness<DTypeQO, DTypeKV>(
                                num_qo_heads, num_kv_heads, seq_len, head_dim,
                                QKVLayout(kv_layout),
                                PosEncodingMode(pos_encoding_mode));
                        }
                    }
                }
            }
        }
    }
}

TEST(FlashInferCorrectnessTest, SingleDecodeKernelCorrectnessTestFP16)
{
    TestSingleDecodeKernelCorrectness<__hip_bfloat16, __half>();
}

TEST(FlashInferCorrectnessTest, SingleDecodeKernelCorrectnessTestBF16)
{
    TestSingleDecodeKernelCorrectness<__hip_bfloat16, __hip_bfloat16>();
}

//*****************************************************************************
// Disabled because we don't have a way to convert from float<-> fp8
//
// TEST(FlashInferCorrectnessTest, SingleDecodeKernelCorrectnessTestE4M3) {
//   TestSingleDecodeKernelCorrectness<half, __hip_fp8_e4m3_fnuz>();
// }

// TEST(FlashInferCorrectnessTest, SingleDecodeKernelCorrectnessTestE5M2) {
//   TestSingleDecodeKernelCorrectness<half, __hip_fp8_e5m2_fnuz>();
// }
//*****************************************************************************

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
