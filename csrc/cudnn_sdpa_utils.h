/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

namespace flashinfer {
namespace cudnn_sdpa_kernel_launcher {

namespace tma {

typedef enum { TILED = 0, IM2COL } cudaTmaDescType;

typedef enum { TENSOR_ZFILL = 0, TENSOR_CFILL } cudaTmaDescOobFillMode;

typedef enum { FP32_TO_TF32_DISABLED = 0, FP32_TO_TF32_ENABLED } cudaTmaDescFp32toTf32Mode;

typedef enum {
  SWIZZLE_DISABLED = 0,
  SWIZZLE_32B,
  SWIZZLE_64B,
  SWIZZLE_128B,
  SWIZZLE_MAX
} cudaTmaDescSwizzle;

typedef enum {
  INTERLEAVE_DISABLED,
  INTERLEAVE_16B,
  INTERLEAVE_32B,
  INTERLEAVE_MAX
} cudaTmaDescInterleave;

typedef enum {
  PROMOTION_DISABLED = 0,
  PROMOTION_64B,
  PROMOTION_128B,
  PROMOTION_256B
} cudaTmaDescPromotion;

typedef enum {
  U8 = 0,
  U16,
  U32,
  S32,
  U64,
  S64,
  F16_RN,
  F32_RN,
  F32_FTZ_RN,
  F64_RN,
  BF16_RN,
  FORMAT_MAX
} cudaTmaDescFormat;

typedef struct {
  uint64_t tensor_common0;
  uint32_t tensor_common1;

  uint32_t tensor_stride_lower[4];  //< 36b of 64b with 4B aligned
  uint32_t tensor_stride_upper;
  uint32_t tensor_size[5];          //< value -1
  uint32_t traversal_stride_box_0;  //< packed 3b (-1)

  uint32_t box_size_end;
} cudaTmaDescTiled;

typedef struct {
  uint64_t tensor_common0;
  uint32_t tensor_common1;

  uint32_t tensor_stride_lower[4];
  uint32_t tensor_stride_upper;
  uint32_t tensor_size[5];
  uint32_t traversal_stride_range_c;

  uint32_t box_corner_dhw;
  uint32_t range_ndhw;
} cudaTmaDescIm2Col;

typedef struct alignas(64) {
  uint64_t data[8];
} cudaTmaDesc;

static inline __host__ __device__ void set_tensor_common_0(cudaTmaDesc* p_desc, uint64_t addr) {
  cudaTmaDescTiled* desc = reinterpret_cast<cudaTmaDescTiled*>(p_desc);
  desc->tensor_common0 = 0;
  desc->tensor_common0 |= addr;
}

static inline __host__ __device__ void set_tensor_common_1(
    cudaTmaDesc* p_desc, cudaTmaDescType desc_type, uint32_t dims, cudaTmaDescFormat format,
    cudaTmaDescInterleave interleave, cudaTmaDescSwizzle swizzle, cudaTmaDescOobFillMode fill,
    cudaTmaDescFp32toTf32Mode f32_to_tf32, cudaTmaDescPromotion promotion) {
  cudaTmaDescTiled* desc = reinterpret_cast<cudaTmaDescTiled*>(p_desc);

  desc->tensor_common1 = 0;
  desc->tensor_common1 |= desc_type == TILED ? 0x0 : 0x1;

  constexpr uint32_t VERSION_SHIFT = 1;
  constexpr uint32_t VERSION_BITS = 3;
  desc->tensor_common1 |= (1u << VERSION_SHIFT);

  constexpr uint32_t DIM_BITS = 3;
  constexpr uint32_t DIM_SHIFT = VERSION_SHIFT + VERSION_BITS;
  constexpr uint32_t DIM_MASK = (1u << DIM_BITS) - 1;
  desc->tensor_common1 |= ((dims - 1) & DIM_MASK) << DIM_SHIFT;

  constexpr uint32_t FORMAT_BITS = 4;
  constexpr uint32_t FORMAT_SHIFT = DIM_SHIFT + DIM_BITS;
  constexpr uint32_t FORMAT_MASK = (1u << FORMAT_BITS) - 1;
  desc->tensor_common1 |= (static_cast<uint32_t>(format) & FORMAT_MASK) << FORMAT_SHIFT;

  constexpr uint32_t INTERLEAVE_BITS = 2;
  constexpr uint32_t INTERLEAVE_SHIFT = FORMAT_SHIFT + FORMAT_BITS;
  constexpr uint32_t INTERLEAVE_MASK = (1u << INTERLEAVE_BITS) - 1;
  desc->tensor_common1 |= (static_cast<uint32_t>(interleave) & INTERLEAVE_MASK) << INTERLEAVE_SHIFT;

  constexpr uint32_t SWIZZLE_BITS = 2;
  constexpr uint32_t SWIZZLE_SHIFT = INTERLEAVE_SHIFT + INTERLEAVE_BITS;
  constexpr uint32_t SWIZZLE_MASK = (1u << SWIZZLE_BITS) - 1;
  desc->tensor_common1 |= (static_cast<uint32_t>(swizzle) & SWIZZLE_MASK) << SWIZZLE_SHIFT;

  constexpr uint32_t FILL_BITS = 1;
  constexpr uint32_t FILL_SHIFT = SWIZZLE_SHIFT + SWIZZLE_BITS;
  constexpr uint32_t FILL_MASK = (1u << FILL_BITS) - 1;
  desc->tensor_common1 |= (static_cast<uint32_t>(fill) & FILL_MASK) << FILL_SHIFT;

  constexpr uint32_t F32_TO_TF32_BITS = 1;
  constexpr uint32_t F32_TO_TF32_SHIFT = FILL_SHIFT + FILL_BITS;
  constexpr uint32_t F32_TO_TF32_MASK = (1u << F32_TO_TF32_BITS) - 1;
  desc->tensor_common1 |= (static_cast<uint32_t>(f32_to_tf32) & F32_TO_TF32_MASK)
                          << F32_TO_TF32_SHIFT;

  constexpr uint32_t PROMOTION_BITS = 2;
  constexpr uint32_t PROMOTION_SHIFT = F32_TO_TF32_SHIFT + F32_TO_TF32_BITS;
  constexpr uint32_t PROMOTION_MASK = (1u << PROMOTION_BITS) - 1;
  desc->tensor_common1 |= (static_cast<uint32_t>(promotion) & PROMOTION_MASK) << PROMOTION_SHIFT;
}

static inline __host__ __device__ void set_tensor_stride(cudaTmaDesc* p_desc,
                                                         uint64_t* p_tensor_stride, uint32_t dims) {
  cudaTmaDescTiled* desc = reinterpret_cast<cudaTmaDescTiled*>(p_desc);

  constexpr uint32_t TENSOR_STRIDE_UPPER_BITS = 4;
  constexpr uint32_t TENSOR_STRIDE_UPPER_MASK = (1u << TENSOR_STRIDE_UPPER_BITS) - 1;

  for (uint32_t i = 0; i < dims - 1; i++) {
    desc->tensor_stride_lower[i] = 0u;
    uint64_t tensor_stride_lower_64b = (p_tensor_stride[i] >> 4) & 0xFFFFFFFFlu;
    desc->tensor_stride_lower[i] = static_cast<uint32_t>(tensor_stride_lower_64b);
  }
  desc->tensor_stride_upper = 0u;

  for (uint32_t i = 0; i < dims - 1; i++) {
    uint64_t tensor_stride = p_tensor_stride[i];
    tensor_stride = tensor_stride >> 4;
    uint64_t tensor_stride_upper = tensor_stride >> 32;
    uint32_t tensor_stride_upper_32b = static_cast<uint32_t>(tensor_stride_upper);
    desc->tensor_stride_upper |=
        ((tensor_stride_upper_32b & TENSOR_STRIDE_UPPER_MASK) << (i * TENSOR_STRIDE_UPPER_BITS));
  }
}

static inline __host__ __device__ void set_tensor_size(cudaTmaDesc* p_desc, uint32_t* p_tensor_size,
                                                       uint32_t dims) {
  cudaTmaDescTiled* desc = reinterpret_cast<cudaTmaDescTiled*>(p_desc);
  for (uint32_t dim = 0; dim < dims; dim++) {
    desc->tensor_size[dim] = p_tensor_size[dim] - 1;
  }
}

static inline __host__ __device__ void set_traversal_stride_tiled(cudaTmaDesc* p_desc,
                                                                  uint32_t* p_traversal_stride,
                                                                  uint32_t dims) {
  cudaTmaDescTiled* desc = reinterpret_cast<cudaTmaDescTiled*>(p_desc);

  desc->traversal_stride_box_0 = 0;

  constexpr uint32_t TRAVERSAL_STRIDE_BITS = 3;
  constexpr uint32_t TRAVERSAL_STRIDE_MASK = (1u << TRAVERSAL_STRIDE_BITS) - 1;

  for (uint32_t dim = 0; dim < dims; dim++) {
    uint32_t traversal_stride = p_traversal_stride[dim] - 1;
    traversal_stride = (traversal_stride & TRAVERSAL_STRIDE_MASK) << (dim * TRAVERSAL_STRIDE_BITS);
    desc->traversal_stride_box_0 |= traversal_stride;
  }
}

static inline __host__ __device__ void set_box_size(cudaTmaDesc* p_desc, uint32_t* p_box_size,
                                                    uint32_t dims) {
  cudaTmaDescTiled* desc = reinterpret_cast<cudaTmaDescTiled*>(p_desc);

  desc->box_size_end = 0;

  constexpr uint32_t BOX_SIZE_BITS = 8;
  constexpr uint32_t BOX_SIZE_MASK = (1 << BOX_SIZE_BITS) - 1;

  if (dims > 1) {
    uint32_t box_size_0 = p_box_size[0] - 1;
    box_size_0 = box_size_0 & BOX_SIZE_MASK;
    box_size_0 = box_size_0 << 24;
    desc->traversal_stride_box_0 |= box_size_0;
  }

  for (uint32_t dim = 1; dim < dims; dim++) {
    uint32_t box_size = p_box_size[dim] - 1;
    box_size = box_size & BOX_SIZE_MASK;
    box_size = box_size << ((dim - 1) * BOX_SIZE_BITS);
    desc->box_size_end |= box_size;
  }
}

static inline __host__ __device__ void set_traversal_stride_im2col(cudaTmaDesc* p_desc,
                                                                   uint32_t* p_traversal_stride,
                                                                   uint32_t dims) {
  cudaTmaDescIm2Col* desc = reinterpret_cast<cudaTmaDescIm2Col*>(p_desc);

  constexpr uint32_t TRAVERSAL_STRIDE_BITS = 3;
  constexpr uint32_t TRAVERSAL_STRIDE_MASK = (1u << (TRAVERSAL_STRIDE_BITS + 1)) - 1;

  desc->traversal_stride_range_c = 0;
  for (uint32_t dim = 0; dim < dims; dim++) {
    uint32_t traversal_stride = p_traversal_stride[dim] - 1;
    traversal_stride = (traversal_stride & TRAVERSAL_STRIDE_MASK) << (dim * TRAVERSAL_STRIDE_BITS);
    desc->traversal_stride_range_c |= traversal_stride;
  }
}

static inline __host__ __device__ void set_range_c(cudaTmaDesc* p_desc, uint32_t range_c) {
  cudaTmaDescIm2Col* desc = reinterpret_cast<cudaTmaDescIm2Col*>(p_desc);

  constexpr uint32_t RANGE_C_BITS = 8;
  constexpr uint32_t RANGE_C_MASK = (1u << RANGE_C_BITS) - 1;

  range_c = range_c & RANGE_C_MASK;
  desc->traversal_stride_range_c |= ((range_c - 1) << 24);
}

static inline __host__ __device__ void set_box_corner_dhw(cudaTmaDesc* p_desc,
                                                          uint32_t* p_base_corner,
                                                          uint32_t* p_far_corner, uint32_t dims) {
  cudaTmaDescIm2Col* desc = reinterpret_cast<cudaTmaDescIm2Col*>(p_desc);

  desc->box_corner_dhw = 0;

  uint32_t box_base_corner = 0, box_far_corner = 0;
  uint32_t box_corner_dhw = 0;

  if (dims == 3) {
    constexpr uint32_t BOX_CORNER_BITS = 16;
    constexpr uint32_t BOX_CORNER_MASK = (1u << BOX_CORNER_BITS) - 1;

    box_base_corner = p_base_corner[0] & BOX_CORNER_MASK;
    box_far_corner = p_far_corner[0] & BOX_CORNER_MASK;
  }

  if (dims == 4) {
    constexpr uint32_t BOX_CORNER_BITS = 8;
    constexpr uint32_t BOX_CORNER_MASK = (1u << BOX_CORNER_BITS) - 1;

    box_base_corner = p_base_corner[0] & BOX_CORNER_MASK;
    box_base_corner |= ((p_base_corner[1] & BOX_CORNER_MASK) << BOX_CORNER_BITS);

    box_far_corner = p_far_corner[0] & BOX_CORNER_MASK;
    box_far_corner |= ((p_far_corner[1] & BOX_CORNER_MASK) << BOX_CORNER_BITS);
  }

  if (dims == 5) {
    constexpr uint32_t BOX_CORNER_BITS = 5;
    constexpr uint32_t BOX_CORNER_MASK = (1u << BOX_CORNER_BITS) - 1;

    box_base_corner = p_base_corner[0] & BOX_CORNER_MASK;
    box_base_corner |= ((p_base_corner[1] & BOX_CORNER_MASK) << BOX_CORNER_BITS);
    box_base_corner |= ((p_base_corner[2] & BOX_CORNER_MASK) << (2 * BOX_CORNER_BITS));

    box_far_corner = p_far_corner[0] & BOX_CORNER_MASK;
    box_far_corner |= ((p_far_corner[1] & BOX_CORNER_MASK) << BOX_CORNER_BITS);
    box_far_corner |= ((p_far_corner[2] & BOX_CORNER_MASK) << (2 * BOX_CORNER_BITS));
  }

  box_corner_dhw = box_base_corner;
  box_corner_dhw |= (box_far_corner << 16);

  desc->box_corner_dhw = box_corner_dhw;
}

static inline __host__ __device__ void set_range_ndhw(cudaTmaDesc* p_desc, uint32_t ndhw) {
  cudaTmaDescIm2Col* desc = reinterpret_cast<cudaTmaDescIm2Col*>(p_desc);

  constexpr uint32_t RANGE_NDHW_BITS = 10;
  constexpr uint32_t RANGE_NDHW_MASK = (1u << RANGE_NDHW_BITS) - 1;

  desc->range_ndhw = 0;
  desc->range_ndhw = ((ndhw - 1) & RANGE_NDHW_MASK);
}

static inline __host__ __device__ void cudaSetTmaTileDescriptor(
    cudaTmaDesc* p_desc, const void* p_addr, uint32_t dims, uint32_t* p_tensor_size,
    uint64_t* p_tensor_stride, uint32_t* p_traversal_stride, uint32_t* p_box_size,
    cudaTmaDescFormat format, cudaTmaDescSwizzle swizzle,
    cudaTmaDescPromotion promotion = cudaTmaDescPromotion::PROMOTION_DISABLED,
    cudaTmaDescInterleave interleave = cudaTmaDescInterleave::INTERLEAVE_DISABLED,
    cudaTmaDescOobFillMode fill_oob = cudaTmaDescOobFillMode::TENSOR_ZFILL,
    cudaTmaDescFp32toTf32Mode round_to_tf32 = cudaTmaDescFp32toTf32Mode::FP32_TO_TF32_DISABLED) {
  set_tensor_common_0(p_desc, reinterpret_cast<uint64_t>(p_addr));
  set_tensor_common_1(p_desc, TILED, dims, format, interleave, swizzle, fill_oob, round_to_tf32,
                      promotion);
  set_tensor_stride(p_desc, p_tensor_stride, dims);
  set_tensor_size(p_desc, p_tensor_size, dims);
  set_traversal_stride_tiled(p_desc, p_traversal_stride, dims);
  set_box_size(p_desc, p_box_size, dims);
}

static inline __host__ __device__ void cudaSetTmaIm2ColDescriptor(
    cudaTmaDesc* p_desc, const void* p_addr, uint32_t dims, uint32_t* p_tensor_size,
    uint64_t* p_tensor_stride, uint32_t* p_traversal_stride, uint32_t range_c, uint32_t range_ndhw,
    uint32_t* p_box_base_corner_dhw, uint32_t* p_box_far_corner_dhw, cudaTmaDescFormat format,
    cudaTmaDescSwizzle swizzle,
    cudaTmaDescPromotion promotion = cudaTmaDescPromotion::PROMOTION_DISABLED,
    cudaTmaDescInterleave interleave = cudaTmaDescInterleave::INTERLEAVE_DISABLED,
    cudaTmaDescOobFillMode fill_oob = cudaTmaDescOobFillMode::TENSOR_ZFILL,
    cudaTmaDescFp32toTf32Mode round_to_tf32 = cudaTmaDescFp32toTf32Mode::FP32_TO_TF32_DISABLED) {
  set_tensor_common_0(p_desc, reinterpret_cast<uint64_t>(p_addr));
  set_tensor_common_1(p_desc, IM2COL, dims, format, interleave, swizzle, fill_oob, round_to_tf32,
                      promotion);

  set_tensor_stride(p_desc, p_tensor_stride, dims);
  set_tensor_size(p_desc, p_tensor_size, dims);
  set_traversal_stride_im2col(p_desc, p_traversal_stride, dims);
  set_range_c(p_desc, range_c);
  set_box_corner_dhw(p_desc, p_box_base_corner_dhw, p_box_far_corner_dhw, dims);
  set_range_ndhw(p_desc, range_ndhw);
}

}  // namespace tma

namespace cudnn_sdpa {

typedef struct AttentionDescriptor {
  // Input parameters
  // b - batch
  // q_h - num heads of q/dq/o/do
  // k_h - num heads of k/dk
  // v_h - num heads of v/dv
  // s_q - max sequence length of q
  // s_kv - max sequence length of kv
  // d - hidden dim (head dim)
  // is_kv_ragged - whether kv is ragged
  uint32_t b, q_h, k_h, v_h, s_q, s_kv, d, q_heads_per_kv, is_kv_ragged;
} AttentionDescriptor_t;

typedef struct FastDivisor {
  uint32_t val, shr, mul;
} FastDivisor_t;

typedef struct strides {
  int64_t strides[4];

  int __attribute__((host)) __attribute__((device)) operator[](int i) const { return strides[i]; }
} strides_t;

}  // namespace cudnn_sdpa

}  // namespace cudnn_sdpa_kernel_launcher
}  // namespace flashinfer
