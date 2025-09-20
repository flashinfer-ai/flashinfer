#include "tvm_binding_utils.h"

// Function declarations (implementations in grouped_gemm.cu)
IntTuple GroupedGemmGetWorkspaceSize(int64_t batch_size, int64_t max_m, int64_t max_n,
                                     int64_t max_k);

void GroupedGemmFp8Run(DLTensor* int_workspace_buffer, DLTensor* float_workspace_buffer,
                       DLTensor* A, DLTensor* B, DLTensor* SFA, DLTensor* SFB, DLTensor* D,
                       DLTensor* m_indptr, int64_t n, int64_t k, int64_t scale_granularity_m,
                       int64_t scale_granularity_n, int64_t scale_granularity_k,
                       int64_t scale_major_mode, int64_t mma_sm, TVMStreamHandle cuda_stream);
