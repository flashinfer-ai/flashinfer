#include <gtest/gtest.h>

#include <flashinfer.cuh>
#include <type_traits>

#include "utils.h"

template <typename T>
void TestBatchDecodeKernelCorrectness() {
  // flashinfer::paged_kv_t<T> paged_kv(); 
   
}

TEST(FlashInferCorrectnessTest, BatchDecodeKernelCorrectnessTestFP16) {
  TestBatchDecodeKernelCorrectness<half>();
}