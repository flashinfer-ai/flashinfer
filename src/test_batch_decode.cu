#include <gtest/gtest.h>

#include <flashinfer.cuh>
#include <type_traits>

#include "utils.cuh"

template <typename T>
void TestBatchDecodeKernelCorrectness() {}

TEST(FlashInferCorrectnessTest, BatchDecodeKernelCorrectnessTestFP16) {
  TestBatchDecodeKernelCorrectness<half>();
}