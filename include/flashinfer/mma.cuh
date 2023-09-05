#ifndef FLASHINFER_MMA_CUH_
#define FLASHINFER_MMA_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace flashinfer {

namespace mma {

__device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t *R, uint32_t smem_ptr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_ptr));
}

__device__ __forceinline__ void mma_sync_m16n8k16_row_col_f16f16f32(float *C, uint32_t *A,
                                                                    uint32_t *B) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
        "f"(C[2]), "f"(C[3]));
}

}  // namespace mma

}  // namespace flashinfer

#endif  // FLASHINFER_MMA_CUH_
