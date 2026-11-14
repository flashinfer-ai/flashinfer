#pragma once
#include <cuda_runtime.h>

void fp8bs_launch(void* out, const void* x, const void* w, const void* ws,
                  int M, int N, int K, float eps, cudaStream_t stream);

bool fp8bs_mwg_launch(void* out, const void* x, const void* w, const void* ws,
                      int M, int N, int K, float eps, cudaStream_t stream);













