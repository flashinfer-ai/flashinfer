// SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#include "flashinfer/attention/generic/page.cuh"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <random>
#include <type_traits>

namespace utils
{

template <typename T>
void vec_normal_(std::vector<T> &vec, float mean = 0.f, float std = 1.f)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution d{mean, std};
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = T(d(gen));
    }
}

template <typename T>
void vec_uniform_(std::vector<T> &vec, float a = 0.f, float b = 1.f)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution d{a, b};
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = T(d(gen));
    }
}

template <typename T> void vec_zero_(std::vector<T> &vec)
{
    std::fill(vec.begin(), vec.end(), T(0));
}

template <typename T> void vec_fill_(std::vector<T> &vec, T val)
{
    std::fill(vec.begin(), vec.end(), val);
}

template <typename T> void vec_randint_(std::vector<T> &vec, int low, int high)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_int_distribution d{low, high};
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = T(d(gen));
    }
}

template <typename T> size_t vec_bytes(const T &vec)
{
    return vec.size() * sizeof(typename T::value_type);
}

template <typename T>
bool isclose(T a, T b, float rtol = 1e-5, float atol = 1e-8)
{
    return fabs(a - b) <= (atol + rtol * fabs(b));
}
} // namespace utils

using namespace flashinfer;

template <typename T, typename IdxType>
void append_paged_kv_cache(paged_kv_t<T, IdxType> page_cpu,
                           const std::vector<std::vector<T>> &keys,
                           const std::vector<std::vector<T>> &values,
                           const std::vector<IdxType> &append_indptr)
{
    size_t batch_size = page_cpu.batch_size;
    size_t num_heads = page_cpu.num_heads;
    size_t head_dim = page_cpu.head_dim;
    size_t page_size = page_cpu.page_size;
    for (size_t i = 0; i < batch_size; ++i) {
        const std::vector<T> &ki = keys[i];
        const std::vector<T> &vi = values[i];
        size_t append_seq_len = append_indptr[i + 1] - append_indptr[i];
        size_t num_pages_i = page_cpu.indptr[i + 1] - page_cpu.indptr[i];
        size_t seq_len =
            (num_pages_i - 1) * page_size + page_cpu.last_page_len[i];
        assert(append_seq_len <= seq_len);
        size_t append_start = seq_len - append_seq_len;

        for (size_t j = 0; j < append_seq_len; ++j) {
            size_t page_seq_idx = j + append_start;
            size_t page_idx =
                page_cpu.indices[page_cpu.indptr[i] + page_seq_idx / page_size];
            size_t entry_idx = page_seq_idx % page_size;
            for (size_t h = 0; h < num_heads; ++h) {
                std::copy(ki.begin() + (j * num_heads + h) * head_dim,
                          ki.begin() + (j * num_heads + h + 1) * head_dim,
                          page_cpu.k_data + page_cpu.get_elem_offset(
                                                page_idx, h, entry_idx, 0));
                std::copy(vi.begin() + (j * num_heads + h) * head_dim,
                          vi.begin() + (j * num_heads + h + 1) * head_dim,
                          page_cpu.v_data + page_cpu.get_elem_offset(
                                                page_idx, h, entry_idx, 0));
            }
        }
    }
}

class PagedKVTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure CUDA is available
        ASSERT_TRUE(torch::cuda::is_available());
    }
};

// Helper function to check for NaN values in a tensor
bool hasNaN(const torch::Tensor &tensor)
{
    return torch::isnan(tensor).any().item<bool>();
}

// Helper function to convert vector to tensor
template <typename T> torch::Tensor vectorToTensor(const std::vector<T> &vec)
{
    torch::Tensor tensor =
        torch::from_blob(
            const_cast<T *>(vec.data()), {static_cast<int64_t>(vec.size())},
            torch::TensorOptions().dtype(
                std::is_same<T, float>::value     ? torch::kFloat32
                : std::is_same<T, half>::value    ? torch::kFloat16
                : std::is_same<T, int32_t>::value ? torch::kInt32
                                                  : torch::kFloat32))
            .clone();
    return tensor;
}

// Helper function to check tensor closeness
bool tensorIsClose(const torch::Tensor &a,
                   const torch::Tensor &b,
                   float atol = 1e-3,
                   float rtol = 1e-3)
{
    return torch::isclose(a, b, atol, rtol).all().item<bool>();
}

template <typename T>
void _TestAppendPagedKVKernelCorrectness(size_t page_size,
                                         size_t batch_size,
                                         size_t num_heads,
                                         size_t head_dim,
                                         QKVLayout kv_layout)
{
    // number of conversation rounds
    size_t num_conv_rounds = 3;
    size_t max_decode_len = 1;
    size_t max_prefill_len = 128;
    size_t max_num_pages = num_conv_rounds * batch_size *
                           ((max_decode_len + max_prefill_len) / page_size + 1);

    // Define tensor options based on the type T
    torch::TensorOptions tensor_options =
        torch::TensorOptions()
            .dtype(std::is_same<T, float>::value ? torch::kFloat32
                   : std::is_same<T, half>::value
                       ? torch::kFloat16
                       : torch::kFloat32) // Default to float32 if type is not
                                          // recognized
            .device(torch::kCUDA);

    torch::TensorOptions int_options =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    // Create CPU tensors for reference
    std::vector<T> k_data_cpu_vec(
        max_num_pages * page_size * num_heads * head_dim, 0);
    std::vector<T> v_data_cpu_vec(
        max_num_pages * page_size * num_heads * head_dim, 0);

    // Create GPU tensors
    torch::Tensor k_data_gpu =
        torch::zeros({static_cast<int64_t>(max_num_pages * page_size *
                                           num_heads * head_dim)},
                     tensor_options);

    torch::Tensor v_data_gpu =
        torch::zeros({static_cast<int64_t>(max_num_pages * page_size *
                                           num_heads * head_dim)},
                     tensor_options);

    std::vector<int32_t> seq_len(batch_size, 0);
    std::vector<std::vector<int32_t>> page_indices(batch_size);
    std::vector<int32_t> last_page_len(batch_size, 0);
    size_t page_counter = 0;

    for (size_t round = 0; round < 2 * num_conv_rounds; ++round) {
        std::vector<int32_t> append_len(batch_size);
        std::vector<int32_t> append_indptr{0};
        std::vector<int32_t> batch_indices;
        std::vector<int32_t> positions;
        std::vector<std::vector<T>> keys;
        std::vector<std::vector<T>> values;

        // Generate random lengths for prefill rounds, fixed for decode rounds
        if (round % 2 == 0) {
            utils::vec_randint_(append_len, 1, max_prefill_len + 1);
        }
        else {
            std::fill(append_len.begin(), append_len.end(), max_decode_len);
        }

        for (size_t i = 0; i < batch_size; ++i) {
            append_indptr.push_back(append_indptr.back() + append_len[i]);
            seq_len[i] += append_len[i];
            for (size_t j = 0; j < append_len[i]; ++j) {
                if (last_page_len[i] % page_size == 0) {
                    page_indices[i].push_back(page_counter++);
                    last_page_len[i] = 1;
                }
                else {
                    last_page_len[i] += 1;
                }
                batch_indices.push_back(i);
                positions.push_back(seq_len[i] - append_len[i] + j);
            }

            // Generate random keys and values
            std::vector<T> ki(append_len[i] * num_heads * head_dim);
            std::vector<T> vi(append_len[i] * num_heads * head_dim);
            utils::vec_normal_(ki);
            utils::vec_normal_(vi);
            keys.push_back(ki);
            values.push_back(vi);
        }

        // Create CPU paged KV cache
        std::vector<int32_t> indptr_cpu{0};
        std::vector<int32_t> indices_cpu;
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < page_indices[i].size(); ++j) {
                indices_cpu.push_back(page_indices[i][j]);
            }
            indptr_cpu.push_back(indptr_cpu.back() + page_indices[i].size());
        }

        paged_kv_t<T, int32_t> paged_kv_cpu(
            num_heads, page_size, head_dim, batch_size, kv_layout,
            /*k_data=*/k_data_cpu_vec.data(),
            /*v_data=*/v_data_cpu_vec.data(), indices_cpu.data(),
            indptr_cpu.data(), last_page_len.data());

        // Apply CPU reference implementation
        append_paged_kv_cache(paged_kv_cpu, keys, values, append_indptr);

        // Create GPU tensors for indices, indptr, and last_page_len
        torch::Tensor indptr_gpu =
            torch::from_blob(indptr_cpu.data(),
                             {static_cast<int64_t>(indptr_cpu.size())},
                             torch::kInt32)
                .clone()
                .to(torch::kCUDA);

        torch::Tensor indices_gpu =
            torch::from_blob(indices_cpu.data(),
                             {static_cast<int64_t>(indices_cpu.size())},
                             torch::kInt32)
                .clone()
                .to(torch::kCUDA);

        torch::Tensor last_page_len_gpu =
            torch::from_blob(last_page_len.data(),
                             {static_cast<int64_t>(batch_size)}, torch::kInt32)
                .clone()
                .to(torch::kCUDA);

        // Create GPU paged KV cache
        paged_kv_t<T, int32_t> paged_kv_gpu(
            num_heads, page_size, head_dim, batch_size, kv_layout,
            /*k_data=*/static_cast<T *>(k_data_gpu.data_ptr()),
            /*v_data=*/static_cast<T *>(v_data_gpu.data_ptr()),
            static_cast<int32_t *>(indices_gpu.data_ptr()),
            static_cast<int32_t *>(indptr_gpu.data_ptr()),
            static_cast<int32_t *>(last_page_len_gpu.data_ptr()));

        // Create batch indices and positions tensors
        torch::Tensor batch_indices_gpu =
            torch::from_blob(batch_indices.data(),
                             {static_cast<int64_t>(batch_indices.size())},
                             torch::kInt32)
                .clone()
                .to(torch::kCUDA);

        torch::Tensor positions_gpu =
            torch::from_blob(positions.data(),
                             {static_cast<int64_t>(positions.size())},
                             torch::kInt32)
                .clone()
                .to(torch::kCUDA);

        // Create keys and values tensors
        torch::Tensor keys_gpu = torch::zeros(
            {static_cast<int64_t>(append_indptr.back() * num_heads * head_dim)},
            tensor_options);

        torch::Tensor values_gpu = torch::zeros(
            {static_cast<int64_t>(append_indptr.back() * num_heads * head_dim)},
            tensor_options);

        // Copy keys and values to GPU
        for (size_t i = 0; i < batch_size; ++i) {
            torch::Tensor ki =
                torch::from_blob(keys[i].data(),
                                 {static_cast<int64_t>(keys[i].size())},
                                 tensor_options.device(torch::kCPU))
                    .clone()
                    .to(torch::kCUDA);

            torch::Tensor vi =
                torch::from_blob(values[i].data(),
                                 {static_cast<int64_t>(values[i].size())},
                                 tensor_options.device(torch::kCPU))
                    .clone()
                    .to(torch::kCUDA);

            keys_gpu
                .slice(0, append_indptr[i] * num_heads * head_dim,
                       append_indptr[i + 1] * num_heads * head_dim)
                .copy_(ki);

            values_gpu
                .slice(0, append_indptr[i] * num_heads * head_dim,
                       append_indptr[i + 1] * num_heads * head_dim)
                .copy_(vi);
        }

        if (round % 2 == 0) {
            // Call prefill kernel
            hipError_t status = AppendPagedKVCache(
                paged_kv_gpu, static_cast<T *>(keys_gpu.data_ptr()),
                static_cast<T *>(values_gpu.data_ptr()),
                static_cast<int32_t *>(batch_indices_gpu.data_ptr()),
                static_cast<int32_t *>(positions_gpu.data_ptr()),
                /*nnz=*/append_indptr.back(),
                /*append_k_stride_n=*/num_heads * head_dim,
                /*append_k_stride_h=*/head_dim,
                /*append_v_stride_n=*/num_heads * head_dim,
                /*append_v_stride_h=*/head_dim);

            EXPECT_EQ(status, hipSuccess)
                << "AppendPagedKVCache kernel launch failed, error message: "
                << hipGetErrorString(status);
        }
        else {
            // Call decode kernel
            hipError_t status = AppendPagedKVCacheDecode(
                paged_kv_gpu, static_cast<T *>(keys_gpu.data_ptr()),
                static_cast<T *>(values_gpu.data_ptr()));

            EXPECT_EQ(status, hipSuccess) << "AppendPagedKVCacheDecode kernel "
                                             "launch failed, error message: "
                                          << hipGetErrorString(status);
        }
    }

    // Copy data back to CPU for verification
    torch::Tensor k_data_cpu =
        torch::from_blob(k_data_cpu_vec.data(),
                         {static_cast<int64_t>(k_data_cpu_vec.size())},
                         tensor_options.device(torch::kCPU))
            .clone();

    torch::Tensor v_data_cpu =
        torch::from_blob(v_data_cpu_vec.data(),
                         {static_cast<int64_t>(v_data_cpu_vec.size())},
                         tensor_options.device(torch::kCPU))
            .clone();

    torch::Tensor k_data_gpu_cpu = k_data_gpu.to(torch::kCPU);
    torch::Tensor v_data_gpu_cpu = v_data_gpu.to(torch::kCPU);

    // Check for NaNs
    bool nan_detected = hasNaN(k_data_gpu_cpu) || hasNaN(v_data_gpu_cpu);

    // Convert to float for comparison
    torch::Tensor k_data_cpu_f32 = k_data_cpu.to(torch::kFloat32);
    torch::Tensor v_data_cpu_f32 = v_data_cpu.to(torch::kFloat32);
    torch::Tensor k_data_gpu_cpu_f32 = k_data_gpu_cpu.to(torch::kFloat32);
    torch::Tensor v_data_gpu_cpu_f32 = v_data_gpu_cpu.to(torch::kFloat32);

    // Check accuracy
    torch::Tensor k_close =
        torch::isclose(k_data_cpu_f32, k_data_gpu_cpu_f32, 1e-3, 1e-3);
    torch::Tensor v_close =
        torch::isclose(v_data_cpu_f32, v_data_gpu_cpu_f32, 1e-3, 1e-3);

    float k_accuracy = k_close.sum().item<float>() / k_close.numel();
    float v_accuracy = v_close.sum().item<float>() / v_close.numel();
    float result_accuracy = (k_accuracy + v_accuracy) / 2.0f;

    std::cout << "kv_layout=" << QKVLayoutToString(kv_layout)
              << ", page_size=" << page_size << ", batch_size=" << batch_size
              << ", num_heads=" << num_heads << ", head_dim=" << head_dim
              << ", result_accuracy=" << result_accuracy << std::endl;

    EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
    EXPECT_FALSE(nan_detected) << "Nan detected in the result.";
}

// Test fixture for parameterized tests
class PagedKVParameterizedTest
    : public PagedKVTest,
      public ::testing::WithParamInterface<
          std::tuple<size_t, size_t, size_t, size_t, QKVLayout>>
{
};

// This is disabled because std::vector cant handle __half dtypes. We will need
// a torch tensor wrapper to handle this.

//  TEST_P(PagedKVParameterizedTest, AppendPagedKVKernelCorrectnessTestFP16) {
//    auto params = GetParam();
//    size_t page_size = std::get<0>(params);
//    size_t batch_size = std::get<1>(params);
//    size_t num_heads = std::get<2>(params);
//    size_t head_dim = std::get<3>(params);
//    QKVLayout kv_layout = std::get<4>(params);

//    _TestAppendPagedKVKernelCorrectness<half>(page_size, batch_size,
//    num_heads, head_dim, kv_layout);
//  }

TEST_P(PagedKVParameterizedTest, AppendPagedKVKernelCorrectnessTestFP32)
{
    auto params = GetParam();
    size_t page_size = std::get<0>(params);
    size_t batch_size = std::get<1>(params);
    size_t num_heads = std::get<2>(params);
    size_t head_dim = std::get<3>(params);
    QKVLayout kv_layout = std::get<4>(params);

    _TestAppendPagedKVKernelCorrectness<float>(page_size, batch_size, num_heads,
                                               head_dim, kv_layout);
}

// Define parameter combinations
INSTANTIATE_TEST_SUITE_P(PagedKVTests,
                         PagedKVParameterizedTest,
                         ::testing::Combine(
                             // page_size
                             ::testing::Values(1, 3, 7, 17),
                             // batch_size
                             ::testing::Values(1, 3, 7, 23),
                             // num_heads
                             ::testing::Values(32),
                             // head_dim
                             ::testing::Values(64, 128),
                             // kv_layout
                             ::testing::Values(QKVLayout::kNHD,
                                               QKVLayout::kHND)));

// Individual test cases for specific configurations
//  TEST_F(PagedKVTest, AppendPagedKVKernelSmallConfigFP16) {
//    _TestAppendPagedKVKernelCorrectness<half>(2, 3, 32, 64, QKVLayout::kHND);
//  }

TEST_F(PagedKVTest, AppendPagedKVKernelLargeConfigFP32)
{
    _TestAppendPagedKVKernelCorrectness<float>(16, 5, 32, 128, QKVLayout::kHND);
}

#ifdef FLASHINFER_ENABLE_BF16
TEST_F(PagedKVTest, AppendPagedKVKernelCorrectnessTestBF16)
{
    _TestAppendPagedKVKernelCorrectness<__hip_bfloat16>(4, 2, 32, 64,
                                                        QKVLayout::kHND);
}
#endif

#ifdef FLASHINFER_ENABLE_FP8_E4M3
TEST_F(PagedKVTest, AppendPagedKVKernelCorrectnessTestE4M3)
{
    _TestAppendPagedKVKernelCorrectness<__hip_fp8_e4m3_fnuz>(4, 2, 32, 64,
                                                             QKVLayout::kHND);
}
#endif

#ifdef FLASHINFER_ENABLE_FP8_E5M2
TEST_F(PagedKVTest, AppendPagedKVKernelCorrectnessTestE5M2)
{
    _TestAppendPagedKVKernelCorrectness<__hip_fp8_e5m2_fnuz>(4, 2, 32, 64,
                                                             QKVLayout::kHND);
}
#endif

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
