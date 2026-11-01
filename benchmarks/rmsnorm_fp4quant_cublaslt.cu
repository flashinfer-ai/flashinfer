// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project
// Standalone real NVFP4 GEMM bridge; no course kernels or simulated GEMM.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>

struct Gemm {
  cublasLtHandle_t lt{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t a{}, b{}, d{};
  cublasLtMatmulPreference_t pref{};
  cublasLtMatmulHeuristicResult_t candidates[8]{};
  int count = 0, selected = 0, status = 0;
  void* workspace = nullptr;
  size_t bytes = 64ull << 20;
  Gemm(int M, int N, int K, const void* as, const void* bs) {
#define LT(call)        \
  do {                  \
    status = int(call); \
    if (status) return; \
  } while (0)
    LT(cublasLtCreate(&lt));
    LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &bs, sizeof(bs)));
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &as, sizeof(as)));
    LT(cublasLtMatrixLayoutCreate(&a, CUDA_R_4F_E2M1, K, N, K));
    LT(cublasLtMatrixLayoutCreate(&b, CUDA_R_4F_E2M1, K, M, K));
    LT(cublasLtMatrixLayoutCreate(&d, CUDA_R_16BF, N, M, N));
    LT(cublasLtMatmulPreferenceCreate(&pref));
    if (cudaMalloc(&workspace, bytes) != cudaSuccess) {
      status = -1;
      return;
    }
    LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes,
                                            sizeof(bytes)));
    LT(cublasLtMatmulAlgoGetHeuristic(lt, op, a, b, d, d, pref, 8, candidates, &count));
#undef LT
  }
  ~Gemm() {
    if (workspace) cudaFree(workspace);
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (d) cublasLtMatrixLayoutDestroy(d);
    if (op) cublasLtMatmulDescDestroy(op);
    if (lt) cublasLtDestroy(lt);
  }
};
extern "C" void* gemm_create(int M, int N, int K, const void* as, const void* bs) {
  return new Gemm(M, N, K, as, bs);
}
extern "C" int gemm_status(void* p) { return static_cast<Gemm*>(p)->status; }
extern "C" int gemm_count(void* p) { return static_cast<Gemm*>(p)->count; }
extern "C" void gemm_select(void* p, int i) { static_cast<Gemm*>(p)->selected = i; }
extern "C" int gemm_algo_id(void* p) {
  auto g = static_cast<Gemm*>(p);
  int id = -1;
  size_t size = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&g->candidates[g->selected].algo, CUBLASLT_ALGO_CONFIG_ID,
                                       &id, sizeof(id), &size);
  return id;
}
extern "C" int gemm_run(void* p, const void* aq, const void* bq, const void* as, const void* bs,
                        void* y, float alpha, uintptr_t stream) {
  auto g = static_cast<Gemm*>(p);
  float beta = 0;
  if (g->status || !g->count) return g->status ? g->status : -2;
  // Y^T = W X^T: weight data AND weight scales are operand A.
  auto s =
      cublasLtMatmulDescSetAttribute(g->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &bs, sizeof(bs));
  if (s != CUBLAS_STATUS_SUCCESS) return int(s);
  s = cublasLtMatmulDescSetAttribute(g->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &as, sizeof(as));
  if (s != CUBLAS_STATUS_SUCCESS) return int(s);
  return int(cublasLtMatmul(g->lt, g->op, &alpha, bq, g->a, aq, g->b, &beta, y, g->d, y, g->d,
                            &g->candidates[g->selected].algo, g->workspace, g->bytes,
                            reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" void gemm_destroy(void* p) { delete static_cast<Gemm*>(p); }
