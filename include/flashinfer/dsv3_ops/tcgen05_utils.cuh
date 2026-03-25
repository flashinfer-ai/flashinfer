#pragma once
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>
#include <string>

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/arch/cluster_sm90.hpp#L180
__device__ inline uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}

template <typename T>
__device__ inline void tma_1d_gmem2smem(const T* src, T* dst, int num_bytes, uint64_t* mbar) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"l"(dst),
      "l"(src), "r"(num_bytes), "l"(mbar)
      : "memory");
}

__device__ inline void tma_2d_gmem2smem(int dst, const void* tmap_ptr, int x, int y,
                                        int mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_"
      "tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr)
      : "memory");
}

template <typename T>
__device__ inline void tma_3d_gmem2smem(T* dst, const void* tmap_ptr, int x, int y, int z,
                                        uint64_t* mbar_addr) {
  // when CTA_GROUP=1, we can use .shared::cta instead.
  // but .shared::cluster doesn't seem to be slower, so always use it
  // unconditionally here. .cta_group::2 allows mbar_addr and dst to be in
  // different CTA's smem.
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_"
      "tx::bytes.cta_group::1 "
      "[%0], [%1, {%2, %3, %4}], [%5];" ::"l"(dst),
      "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "l"(mbar_addr)
      : "memory");
}

__device__ inline void tcgen05_mma_f8(int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
                                      int enable_input_d) {
  // no block shifts version
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"  // predicate register enable-input-d
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
      "}" ::"r"(taddr),
      "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d));
}
__device__ inline constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; }

__device__ inline void tcgen05_ld_16x256b(int addr, float (&tmp)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1, %2, %3, %4, "
      "%5, %6, %7}, [%8];"
      : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]), "=f"(tmp[4]), "=f"(tmp[5]),
        "=f"(tmp[6]), "=f"(tmp[7])
      : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ inline void tcgen05_ld_32x32b_x2(int addr, float (&tmp)[2]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];"
               : "=f"(tmp[0]), "=f"(tmp[1])
               : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ inline void tcgen05_ld_32x32b_x4(int addr, float (&tmp)[4]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
               : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3])
               : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ inline void tcgen05_ld_32x32b_x8(int addr, float (&tmp)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, "
      "[%8];"
      : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]), "=f"(tmp[4]), "=f"(tmp[5]),
        "=f"(tmp[6]), "=f"(tmp[7])
      : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ inline void tcgen05_ld_32x32b_x16(int addr, float (&tmp)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, "
      "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
      : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]), "=f"(tmp[4]), "=f"(tmp[5]),
        "=f"(tmp[6]), "=f"(tmp[7]), "=f"(tmp[8]), "=f"(tmp[9]), "=f"(tmp[10]), "=f"(tmp[11]),
        "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15])
      : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ inline void tcgen05_ld_32x32b_x32(int addr, float (&tmp)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, "
      "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, "
      "%19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, "
      "%31}, [%32];"
      : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]), "=f"(tmp[4]), "=f"(tmp[5]),
        "=f"(tmp[6]), "=f"(tmp[7]), "=f"(tmp[8]), "=f"(tmp[9]), "=f"(tmp[10]), "=f"(tmp[11]),
        "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]), "=f"(tmp[16]), "=f"(tmp[17]),
        "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
        "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]),
        "=f"(tmp[30]), "=f"(tmp[31])
      : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ inline void tcgen05_ld_32x32b_x64(int addr, float (&tmp)[64]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0, %1, %2, %3, %4, %5, "
      "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, "
      "%19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, "
      "%31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, "
      "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, "
      "%55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
      : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]), "=f"(tmp[4]), "=f"(tmp[5]),
        "=f"(tmp[6]), "=f"(tmp[7]), "=f"(tmp[8]), "=f"(tmp[9]), "=f"(tmp[10]), "=f"(tmp[11]),
        "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]), "=f"(tmp[16]), "=f"(tmp[17]),
        "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
        "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]),
        "=f"(tmp[30]), "=f"(tmp[31]), "=f"(tmp[32]), "=f"(tmp[33]), "=f"(tmp[34]), "=f"(tmp[35]),
        "=f"(tmp[36]), "=f"(tmp[37]), "=f"(tmp[38]), "=f"(tmp[39]), "=f"(tmp[40]), "=f"(tmp[41]),
        "=f"(tmp[42]), "=f"(tmp[43]), "=f"(tmp[44]), "=f"(tmp[45]), "=f"(tmp[46]), "=f"(tmp[47]),
        "=f"(tmp[48]), "=f"(tmp[49]), "=f"(tmp[50]), "=f"(tmp[51]), "=f"(tmp[52]), "=f"(tmp[53]),
        "=f"(tmp[54]), "=f"(tmp[55]), "=f"(tmp[56]), "=f"(tmp[57]), "=f"(tmp[58]), "=f"(tmp[59]),
        "=f"(tmp[60]), "=f"(tmp[61]), "=f"(tmp[62]), "=f"(tmp[63])
      : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

template <int WIDTH>
__device__ inline void tcgen05_ld_32x32b(int addr, float (&tmp)[WIDTH]) {
  if constexpr (WIDTH == 2) {
    tcgen05_ld_32x32b_x2(addr, tmp);
  } else if constexpr (WIDTH == 4) {
    tcgen05_ld_32x32b_x4(addr, tmp);
  } else if constexpr (WIDTH == 8) {
    tcgen05_ld_32x32b_x8(addr, tmp);
  } else if constexpr (WIDTH == 16) {
    tcgen05_ld_32x32b_x16(addr, tmp);
  } else if constexpr (WIDTH == 32) {
    tcgen05_ld_32x32b_x32(addr, tmp);
  } else if constexpr (WIDTH == 64) {
    tcgen05_ld_32x32b_x64(addr, tmp);
  } else {
    static_assert(false, "WIDTH must be 2, 4, 8, 16, 32, or 64");
  }
}

// ---------------------------------------------------------------------------
// Shared-memory address conversion
// ---------------------------------------------------------------------------

// Convert a generic pointer to a 32-bit shared-memory address suitable for
// use in inline PTX instructions that expect a .shared::cta address operand.
template <typename T>
__device__ inline int cvt_addr(T* addr) {
  return static_cast<int>(__cvta_generic_to_shared(addr));
}

// ---------------------------------------------------------------------------
// TMA tensor-map helpers (host-side)
// ---------------------------------------------------------------------------

// Encode an N-dimensional tiled tensor map for TMA.
// Template parameter rank must match the lengths of globalDim, globalStrides,
// and boxDim.  All dimensions and strides are in units of uint8 elements.
template <int rank>
inline void init_tensormap_nd(CUtensorMap* tmap, uint8_t* ptr, uint64_t* globalDim,
                              uint64_t* globalStrides, uint32_t* boxDim) {
  uint32_t elem_strides[rank];
  for (int i = 0; i < rank; ++i) {
    elem_strides[i] = 1;
  }
  auto err = cuTensorMapEncodeTiled(tmap, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, rank,
                                    (void*)ptr, globalDim, globalStrides, boxDim, elem_strides,
                                    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                                    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                                    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (err != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    cuGetErrorString(err, &err_str);
    throw std::runtime_error(std::string("cuTensorMapEncodeTiled failed: ") +
                             (err_str ? err_str : "unknown error"));
  }
}

// Build Q and K tensor maps for the MQA kernel.
//   Q layout: [batch_size, Q_NEXT * 64, 128] uint8, stride [Q_NEXT * 64 * 128, 128]
//   K layout: [num_pages,          64, 128] uint8, stride [64 * 132, 128]
// The K global stride is 132 * 64 (not 128 * 64) because each page row is
// padded to 132 bytes to hold 128 fp8 values + 64 fp32 per-token scales.
template <int Q_NEXT>
inline void prep_tmaps(CUtensorMap* q_tmap, CUtensorMap* k_tmap, uint8_t* q_ptr, uint8_t* k_ptr,
                       int batch_size, int num_pages) {
  uint32_t q_boxDim[3] = {128, 64 * Q_NEXT, 1};
  uint32_t k_boxDim[3] = {128, 64, 1};

  uint64_t q_globalDim[3] = {128, 64 * Q_NEXT, (uint64_t)batch_size};
  uint64_t k_globalDim[3] = {128, 64, (uint64_t)num_pages};
  uint64_t q_globalStrides[2] = {128, 128 * 64 * Q_NEXT};
  uint64_t k_globalStrides[2] = {128, 132 * 64};
  init_tensormap_nd<3>(q_tmap, q_ptr, q_globalDim, q_globalStrides, q_boxDim);
  init_tensormap_nd<3>(k_tmap, k_ptr, k_globalDim, k_globalStrides, k_boxDim);
}
