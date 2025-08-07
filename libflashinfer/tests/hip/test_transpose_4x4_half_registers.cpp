// test_transpose_4x4_half_registers.cpp
#include "gpu_iface/backend/hip/mma_hip.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define FI_GPU_CALL(call)                                                      \
    do {                                                                       \
        gpuError_t err = (call);                                               \
        if (err != gpuSuccess) {                                               \
            std::ostringstream err_msg;                                        \
            err_msg << "GPU error: " << gpuGetErrorString(err) << " at "       \
                    << __FILE__ << ":" << __LINE__;                            \
            throw std::runtime_error(err_msg.str());                           \
        }                                                                      \
    } while (0)

__device__ __forceinline__ void debug_print_registers(const char *stage,
                                                      uint32_t lane_id,
                                                      uint32_t lane_in_group,
                                                      uint32_t *regs,
                                                      int num_regs,
                                                      uint32_t debug_group = 0)
{

    // Only debug a specific group to avoid excessive output
    if (lane_id / 4 != debug_group)
        return;

    // Print identification info
    printf("STAGE: %s | Thread %d (lane_in_group=%d): ", stage, lane_id,
           lane_in_group);

    // Print raw 32-bit values
    printf("RAW=[");
    for (int i = 0; i < num_regs; i++) {
        printf("0x%08x", regs[i]);
        if (i < num_regs - 1)
            printf(", ");
    }
    printf("] | ");

    // Print unpacked 16-bit values
    printf("UNPACKED=[");
    for (int i = 0; i < num_regs; i++) {
        uint16_t hi = (regs[i] >> 16) & 0xFFFF;
        uint16_t lo = regs[i] & 0xFFFF;
        printf("%d,%d", hi, lo);
        if (i < num_regs - 1)
            printf(", ");
    }
    printf("]\n");
}

__device__ __forceinline__ void transpose_4x4_half_registers_opt(uint32_t *R)
{
    // Calculate lane within 4-thread group
    uint32_t lane_id = threadIdx.x % 64;
    uint32_t lane_in_group = lane_id % 4;

    // === ROUND 1: Exchange with neighbor (XOR with 1) ===
    // T0↔T1, T2↔T3 partial exchange
    uint32_t reg_idx = (lane_in_group >> 1) & 0x1;
    uint32_t exchanged_val = __shfl_xor(R[reg_idx], 0x1);
    uint32_t shift = (lane_in_group & 1) * 16;
    uint32_t keep_mask = 0xFFFF0000 >> shift;
    int right_shift_amount = 16 * (1 - (lane_in_group & 1));
    int left_shift_amount = 16 * (lane_in_group & 1);
    R[reg_idx] = (R[reg_idx] & keep_mask) |
                 ((exchanged_val >> right_shift_amount) << left_shift_amount);

    // === ROUND 2: Exchange with one hop (XOR with 2) ===
    // T0↔T2, T1↔T3 exchange R[0] and R[1]
    // Swap entire registers based on thread position
    uint32_t is_top = 1 - reg_idx;
    uint32_t temp0 = __shfl_xor(R[0], 0x2);
    uint32_t temp1 = __shfl_xor(R[1], 0x2);

    // Compute both possibilities and select
    R[0] = R[0] * is_top + temp1 * reg_idx;
    R[1] = temp0 * is_top + R[1] * reg_idx;

    // === ROUND 3: Exchange with neighbor again (XOR with 1) ===
    // T0↔T1, T2↔T3 exchange remaining parts

    reg_idx = 1 - reg_idx;
    exchanged_val = __shfl_xor(R[reg_idx], 0x1);
    R[reg_idx] = (R[reg_idx] & keep_mask) |
                 ((exchanged_val >> right_shift_amount) << left_shift_amount);
}

__device__ __forceinline__ void transpose_4x4_half_registers_naive(uint32_t *R)
{
    // Calculate lane within 4-thread group
    uint32_t lane_id = threadIdx.x % 64;
    uint32_t lane_in_group = lane_id % 4;

    if (lane_id == 0) {
        debug_print_registers("Initial", lane_id, lane_in_group, R, 2, 0);
    }

    // === ROUND 1: Exchange with neighbor (XOR with 1) ===
    // T0↔T1, T2↔T3 partial exchange

    // Update based on thread position
    if (lane_in_group < 2) {
        uint32_t r0_exchanged = __shfl_xor(R[0], 0x1);
        // Top half (T0, T1) update R[0]
        if (lane_in_group & 1) { // T1
            R[0] = (R[0] & 0x0000FFFF) | (r0_exchanged << 16);
        }
        else { // T0
            R[0] = (R[0] & 0xFFFF0000) | (r0_exchanged >> 16);
        }
    }
    else {
        uint32_t r1_exchanged = __shfl_xor(R[1], 0x1);
        // Bottom half (T2, T3) update R[1]
        if (lane_in_group & 1) { // T1
            R[1] = (R[1] & 0x0000FFFF) | (r1_exchanged << 16);
        }
        else { // T0
            R[1] = (R[1] & 0xFFFF0000) | (r1_exchanged >> 16);
        }
    }

    // Debug after first recombination
    if (lane_id == 3) {
        debug_print_registers("After Round 1 shuffles", lane_id, lane_in_group,
                              R, 2, 0);
    }

    // === ROUND 2: Exchange with one hop (XOR with 2) ===
    // T0↔T2, T1↔T3 exchange R[0] and R[1]
    uint32_t temp0_exchanged = __shfl_xor(R[0], 0x2);
    uint32_t temp1_exchanged = __shfl_xor(R[1], 0x2);

    // Swap entire registers based on thread position
    if (lane_in_group < 2) {
        R[1] = temp0_exchanged;
    }
    else {
        // Bottom threads (T2, T3) get R[1] from partner, keep own R[0]
        R[0] = temp1_exchanged;
    }

    if (lane_id == 0) {
        debug_print_registers("After Round 2 shuffles", lane_id, lane_in_group,
                              R, 2, 0);
    }

    // === ROUND 3: Exchange with neighbor again (XOR with 1) ===
    // T0↔T1, T2↔T3 exchange remaining parts

    if (lane_in_group < 2) {
        uint32_t r1_exchanged = __shfl_xor(R[1], 0x1);
        // Top half (T0, T1) update R[0]
        if (lane_in_group & 1) { // T1
            R[1] = (R[1] & 0x0000FFFF) | (r1_exchanged << 16);
        }
        else { // T0
            R[1] = (R[1] & 0xFFFF0000) | (r1_exchanged >> 16);
        }
    }
    else {
        uint32_t r1_exchanged = __shfl_xor(R[0], 0x1);
        // Bottom half (T2, T3) update R[1]
        if (lane_in_group & 1) { // T1
            R[0] = (R[0] & 0x0000FFFF) | (r1_exchanged << 16);
        }
        else { // T0
            R[0] = (R[0] & 0xFFFF0000) | (r1_exchanged >> 16);
        }
    }

    if (lane_id == 3) {
        debug_print_registers("After Round 2 shuffles", lane_id, lane_in_group,
                              R, 2, 0);
    }
}

// Helper function to convert two uint16_t values to a single uint32_t
__host__ __device__ uint32_t pack_half2(uint16_t a, uint16_t b)
{
    return ((uint32_t)a << 16) | (uint32_t)b;
}

// Helper function to extract two uint16_t values from a single uint32_t
__host__ __device__ void unpack_half2(uint32_t packed, uint16_t &a, uint16_t &b)
{
    a = (packed >> 16) & 0xFFFF;
    b = packed & 0xFFFF;
}

// Kernel to test the transpose function
__global__ void test_transpose_kernel(uint16_t *output)
{
    uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = thread_id % 64;

    // Calculate the thread's position in the logical 4x4 grid
    uint32_t lane_in_group = lane_id % 4; // Position within group

    // Initialize test data - each thread creates a row of the matrix B
    // Values are designed for easy verification: lane_in_group * 100 + column
    uint16_t row_elements[4];
    for (int i = 0; i < 4; i++) {
        row_elements[i] = lane_in_group * 100 + i; // B[lane_in_group][i]
    }

    // Pack the 4 half-precision values into 2 registers
    uint32_t R[2];
    R[0] = pack_half2(row_elements[0], row_elements[1]);
    R[1] = pack_half2(row_elements[2], row_elements[3]);

    // Call the transpose function
    flashinfer::gpu_iface::mma_impl::hip::transpose_4x4_half_registers(R);

    // Unpack the transposed results
    uint16_t transposed[4];
    unpack_half2(R[0], transposed[0], transposed[1]);
    unpack_half2(R[1], transposed[2], transposed[3]);

    // Write output - store both original and transposed values for verification
    for (int i = 0; i < 4; i++) {
        // Original values (row-major layout)
        output[thread_id * 8 + i] = row_elements[i];
        // Transposed values (column-major layout)
        output[thread_id * 8 + 4 + i] = transposed[i];
    }
}

int main()
{
    // Allocate memory for output (both original and transposed data)
    const int num_threads = 64; // One wavefront
    const int values_per_thread =
        8; // Each thread stores 4 original + 4 transposed values
    const int total_values = num_threads * values_per_thread;

    std::vector<uint16_t> h_output(total_values);
    uint16_t *d_output;

    FI_GPU_CALL(hipMalloc(&d_output, total_values * sizeof(uint16_t)));

    // Launch the kernel
    test_transpose_kernel<<<1, num_threads>>>(d_output);

    // Copy results back to host
    FI_GPU_CALL(hipMemcpy(h_output.data(), d_output,
                          total_values * sizeof(uint16_t),
                          hipMemcpyDeviceToHost));

    // Verify the results
    bool success = true;
    std::cout << "Testing matrix transposition with shuffle operations..."
              << std::endl;

    for (int group = 0; group < num_threads / 4; group++) {
        std::cout << "\nGroup " << group << " results:" << std::endl;

        for (int lane = 0; lane < 4; lane++) {
            int thread_idx = group * 4 + lane;

            // Print original values
            std::cout << "Thread " << thread_idx << " original: ";
            for (int i = 0; i < 4; i++) {
                std::cout << h_output[thread_idx * 8 + i] << " ";
            }
            std::cout << std::endl;

            // Print and verify transposed values
            std::cout << "Thread " << thread_idx << " transposed: ";
            for (int i = 0; i < 4; i++) {
                uint16_t actual = h_output[thread_idx * 8 + 4 + i];
                std::cout << actual << " ";

                // Expected after transpose: Thread N gets column N
                // Thread 0 should have [0*100+0, 1*100+0, 2*100+0, 3*100+0]
                // Thread 1 should have [0*100+1, 1*100+1, 2*100+1, 3*100+1]
                uint16_t expected = i * 100 + lane;

                if (actual != expected) {
                    success = false;
                    std::cout << "(Expected: " << expected << ") ";
                }
            }
            std::cout << std::endl;
        }
    }

    if (success) {
        std::cout << "\nTranspose test PASSED! All values correctly transposed."
                  << std::endl;
    }
    else {
        std::cout << "\nTranspose test FAILED! Some values were not correctly "
                     "transposed."
                  << std::endl;
    }

    // Clean up
    FI_GPU_CALL(hipFree(d_output));

    return success ? 0 : 1;
}
