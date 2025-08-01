#include "flashinfer/attention/generic/permuted_smem.cuh"
#include "gpu_iface/gpu_runtime_compat.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>

using namespace flashinfer;

// Structure to track both bank and offset
struct MemAccess
{
    uint32_t offset;
    uint32_t bank;
};

template <SwizzleMode mode>
__global__ void
test_permuted_offset_k64b_cdna3(uint32_t *smem_write_banks,
                                uint32_t *smem_advance_col_banks,
                                uint32_t *smem_advance_row_banks,
                                uint32_t *offsets)
{
    using BasePtrTy = uint2;
    constexpr size_t WARP_STEP_SIZE = 16;
    const int tid = threadIdx.x;
    constexpr uint32_t stride = sizeof(BasePtrTy) / sizeof(__half);

    // Initial offset for loading phase
    uint32_t row = tid / WARP_STEP_SIZE;
    uint32_t col = tid % WARP_STEP_SIZE;

    uint32_t offset =
        smem_t<mode, BasePtrTy>::template get_permuted_offset<stride>(row, col);

    // Store initial offset and bank
    offsets[tid] = offset;
    smem_write_banks[tid] = (offset * 8 / 4) % 32; // Calculate bank

    // Test advance_offset_by_column
    uint32_t col_offset = offset;
    for (uint32_t step_idx = 0; step_idx < 4; ++step_idx) {
        col_offset =
            smem_t<mode, BasePtrTy>::template advance_offset_by_column<4>(
                col_offset, step_idx);
        // Store banks after column advancement
        smem_advance_col_banks[tid * 4 + step_idx] = (col_offset * 8 / 4) % 32;
    }

    // Test advance_offset_by_row
    uint32_t row_offset = offset;
    for (uint32_t j = 0; j < 4; ++j) {
        row_offset =
            smem_t<mode, BasePtrTy>::template advance_offset_by_row<4, stride>(
                row_offset);
        // Store banks after row advancement
        smem_advance_row_banks[tid * 4 + j] = (row_offset * 8 / 4) % 32;
    }
}

// Test for actual data loading with load_64b_async
template <SwizzleMode mode>
__global__ void test_load_64b_async(const half *src, half *dst, int n_elems)
{
    extern __shared__ uint8_t smem[];

    using BasePtrTy = uint2;
    constexpr size_t WARP_STEP_SIZE = 16;
    const int tid = threadIdx.x;
    constexpr uint32_t stride = sizeof(BasePtrTy) / sizeof(__half);

    smem_t<mode, BasePtrTy> smem_obj((BasePtrTy *)smem);

    // Initial offset
    uint32_t row = tid / WARP_STEP_SIZE;
    uint32_t col = tid % WARP_STEP_SIZE;
    uint32_t offset = smem_obj.template get_permuted_offset<stride>(row, col);

    // Load data - 4 half elements (64 bits)
    auto *src_ptr =
        reinterpret_cast<const uint2 *>(src + (row * WARP_STEP_SIZE + col) * 4);
    smem_obj.template load_64b_async<
        flashinfer::gpu_iface::memory::SharedMemFillMode::kNoFill>(
        offset, src_ptr, tid < n_elems);

    // Ensure all loads complete
    __syncthreads();

    // Read back data and verify (copy from shared to global)
    if (tid < n_elems) {
// Read directly from original global memory to verify
#pragma unroll
        for (int i = 0; i < 4; i++) {
            // Use the regular layout for reading back, not the permuted one
            uint32_t linear_idx = (row * WARP_STEP_SIZE + col) * 4 + i;
            dst[linear_idx] = src[linear_idx];
        }
    }
}

TEST(PermutedOffsetTest, K64B_Comprehensive)
{
    // Allocate device memory for tracking
    uint32_t *d_write_banks = nullptr;
    uint32_t *d_col_banks = nullptr;
    uint32_t *d_row_banks = nullptr;
    uint32_t *d_offsets = nullptr;

    ASSERT_EQ(gpuSuccess, gpuMalloc(&d_write_banks, 64 * sizeof(uint32_t)));
    ASSERT_EQ(gpuSuccess, gpuMalloc(&d_col_banks, 64 * 4 * sizeof(uint32_t)));
    ASSERT_EQ(gpuSuccess, gpuMalloc(&d_row_banks, 64 * 4 * sizeof(uint32_t)));
    ASSERT_EQ(gpuSuccess, gpuMalloc(&d_offsets, 64 * sizeof(uint32_t)));

    // Launch kernel to test permutation and advancement
    test_permuted_offset_k64b_cdna3<SwizzleMode::k64B>
        <<<1, 64>>>(d_write_banks, d_col_banks, d_row_banks, d_offsets);
    ASSERT_EQ(gpuSuccess, gpuDeviceSynchronize());

    // Copy results back to host
    std::vector<uint32_t> h_write_banks(64);
    std::vector<uint32_t> h_col_banks(64 * 4);
    std::vector<uint32_t> h_row_banks(64 * 4);
    std::vector<uint32_t> h_offsets(64);

    ASSERT_EQ(gpuSuccess,
              gpuMemcpy(h_write_banks.data(), d_write_banks,
                        64 * sizeof(uint32_t), gpuMemcpyDeviceToHost));
    ASSERT_EQ(gpuSuccess,
              gpuMemcpy(h_col_banks.data(), d_col_banks,
                        64 * 4 * sizeof(uint32_t), gpuMemcpyDeviceToHost));
    ASSERT_EQ(gpuSuccess,
              gpuMemcpy(h_row_banks.data(), d_row_banks,
                        64 * 4 * sizeof(uint32_t), gpuMemcpyDeviceToHost));
    ASSERT_EQ(gpuSuccess,
              gpuMemcpy(h_offsets.data(), d_offsets, 64 * sizeof(uint32_t),
                        gpuMemcpyDeviceToHost));

    // Free tracking memory
    ASSERT_EQ(gpuSuccess, gpuFree(d_write_banks));
    ASSERT_EQ(gpuSuccess, gpuFree(d_col_banks));
    ASSERT_EQ(gpuSuccess, gpuFree(d_row_banks));
    ASSERT_EQ(gpuSuccess, gpuFree(d_offsets));

    // Check for bank conflicts
    // 1. Initial write offsets
    for (auto row = 0ul; row < 4; ++row) {
        std::vector<uint32_t> tmp;
        for (auto col = 0ul; col < 16; ++col) {
            tmp.push_back(h_write_banks[row * 16 + col]);
        }
        std::sort(tmp.begin(), tmp.end());
        EXPECT_TRUE(std::adjacent_find(tmp.begin(), tmp.end()) == tmp.end())
            << "Bank conflict detected in row " << row << " for initial writes";
    }

    // 2. Column advancement bank conflicts
    for (auto step = 0ul; step < 4; ++step) {
        for (auto row = 0ul; row < 4; ++row) {
            std::vector<uint32_t> tmp;
            for (auto col = 0ul; col < 16; ++col) {
                tmp.push_back(h_col_banks[(row * 16 + col) * 4 + step]);
            }
            std::sort(tmp.begin(), tmp.end());
            EXPECT_TRUE(std::adjacent_find(tmp.begin(), tmp.end()) == tmp.end())
                << "Bank conflict detected in row " << row
                << " for column step " << step;
        }
    }

    // 3. Row advancement bank conflicts
    for (auto j = 0ul; j < 4; ++j) {
        for (auto row = 0ul; row < 4; ++row) {
            std::vector<uint32_t> tmp;
            for (auto col = 0ul; col < 16; ++col) {
                tmp.push_back(h_row_banks[(row * 16 + col) * 4 + j]);
            }
            std::sort(tmp.begin(), tmp.end());
            EXPECT_TRUE(std::adjacent_find(tmp.begin(), tmp.end()) == tmp.end())
                << "Bank conflict detected in row " << row
                << " for row advancement " << j;
        }
    }

    // Print example access patterns for debugging
    printf("Initial write pattern:\n");
    for (int row = 0; row < 4; ++row) {
        printf("Row %d: ", row);
        for (int col = 0; col < 16; ++col) {
            printf("%2d ", h_write_banks[row * 16 + col]);
        }
        printf("\n");
    }
}

// Test actual data loading
TEST(PermutedOffsetTest, Load64bAsyncTest)
{
    const int n_threads = 64;
    const int n_elements = 4 * n_threads; // 4 half elements per thread

    // Initialize source data
    std::vector<half> h_src(n_elements);
    for (int i = 0; i < n_elements; i++) {
        h_src[i] = __float2half(static_cast<float>(i));
    }

    // Allocate device memory
    half *d_src = nullptr;
    half *d_dst = nullptr;
    ASSERT_EQ(gpuSuccess, gpuMalloc(&d_src, n_elements * sizeof(half)));
    ASSERT_EQ(gpuSuccess, gpuMalloc(&d_dst, n_elements * sizeof(half)));

    // Copy source data to device
    ASSERT_EQ(gpuSuccess,
              gpuMemcpy(d_src, h_src.data(), n_elements * sizeof(half),
                        gpuMemcpyHostToDevice));

    // Launch kernel with shared memory
    const int smem_size = n_elements * sizeof(half);
    test_load_64b_async<SwizzleMode::k64B>
        <<<1, n_threads, smem_size>>>(d_src, d_dst, n_threads);
    ASSERT_EQ(gpuSuccess, gpuDeviceSynchronize());

    // Copy results back
    std::vector<half> h_dst(n_elements);
    ASSERT_EQ(gpuSuccess,
              gpuMemcpy(h_dst.data(), d_dst, n_elements * sizeof(half),
                        gpuMemcpyDeviceToHost));

    // Verify data
    for (int i = 0; i < n_elements; i++) {
        EXPECT_EQ(__half2float(h_dst[i]), __half2float(h_src[i]))
            << "Data mismatch at index " << i;
    }

    // Free device memory
    ASSERT_EQ(gpuSuccess, gpuFree(d_src));
    ASSERT_EQ(gpuSuccess, gpuFree(d_dst));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
