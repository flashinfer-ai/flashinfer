/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>
#include <flashinfer/exception.h>
#include <flashinfer/trtllm/common/sageQuant.h>

#include <algorithm>
#include <cstdint>
#include <cute/tensor.hpp>
#include <type_traits>

namespace flashinfer::trtllm {

// SageAttention quantization kernel. Tensors are interpreted as column-major
// [D, H, S], which is the same physical layout as contiguous PyTorch [S, H, D].
// Each launch quantizes Q or K. It can simultaneously collect V scales
// (VStage=1) or quantize V using scales collected by an earlier launch
// (VStage=2).
template <typename Element, typename ElementQuantized, int TokenPerScale, int HeadDim, bool KSmooth,
          int VStage>
__global__ void sageQuantQkvKernel(int sumSeqLensQk, void const* ptrQk, void* ptrQkQuant,
                                   float* ptrQkScale, float* ptrKMean, int sumSeqLensV,
                                   int numHeadsV, void const* ptrV, void* ptrVQuant,
                                   float* ptrVScale) {
  using namespace cute;
  using namespace cutlass;
  static_assert(!KSmooth, "K-smoothing not implemented yet");
  static_assert(std::is_same_v<ElementQuantized, float_e4m3_t> ||
                    std::is_same_v<ElementQuantized, std::int8_t>,
                "Unrecognized target dtype for quantization");
  constexpr float TypeMax =
      cute::is_same_v<ElementQuantized, float_e4m3_t> ? 448.0f : static_cast<float>(126.9f);
  constexpr int BestVL = 128 / sizeof_bits_v<Element>;
  using VL = Int<BestVL>;

  (void)ptrKMean;

  int const numHeads = gridDim.y;
  int const headIdx = blockIdx.y;
  int const numWarpsPerCta = blockDim.x / 32;
  int const numWarps = gridDim.x * numWarpsPerCta;
  int const warpId = blockIdx.x * numWarpsPerCta + threadIdx.x / 32;
  int const thrId = threadIdx.x % 32;

  if (blockIdx.z == 0) {
    Tensor gQk = make_tensor(reinterpret_cast<Element const*>(ptrQk),
                             make_shape(Int<HeadDim>{}, numHeads, sumSeqLensQk));
    Tensor gQkQuant = make_tensor(reinterpret_cast<ElementQuantized*>(ptrQkQuant),
                                  make_shape(Int<HeadDim>{}, numHeads, sumSeqLensQk));
    Tensor gQkScale =
        make_tensor(ptrQkScale, make_shape(ceil_div(sumSeqLensQk, TokenPerScale), numHeads));

    Tensor gQkSeq = gQk(_, headIdx, _);
    Tensor gQkSeqQuant = gQkQuant(_, headIdx, _);
    Tensor gQkSeqScale = gQkScale(_, headIdx);
    Tensor gQkVecs = tiled_divide(gQkSeq, Shape<VL, Int<TokenPerScale>>{});
    Tensor gQkVecsQuant = tiled_divide(gQkSeqQuant, Shape<VL, Int<TokenPerScale>>{});

    Tensor rQk = make_tensor<Element>(Shape<VL, Int<TokenPerScale>>{});
    Tensor rQkQuant = make_tensor<ElementQuantized>(Shape<VL, Int<TokenPerScale>>{});
    Tensor rQkCompute = make_tensor<float>(Shape<VL, Int<TokenPerScale>>{});
    Tensor rQk_x2 = recast<Array<Element, 2>>(rQk);
    Tensor rQkCompute_x2 = recast<Array<float, 2>>(rQkCompute);
    Tensor rQk_x4 = recast<Array<Element, 4>>(rQk);
    Tensor rQkQuant_x4 = recast<Array<ElementQuantized, 4>>(rQkQuant);

    constexpr int threadsPerScale = size<1>(gQkVecs);
    static_assert(threadsPerScale <= 32, "One token block should never exceed warp scope");
    int const numScalesPerWarp = 32 / threadsPerScale;
    int const numScalesPerWave = numWarps * numScalesPerWarp;
    int const numWholeScales = sumSeqLensQk / TokenPerScale;
    int tokBlkIdx = warpId * numScalesPerWarp + thrId / threadsPerScale;
    int const threadInScaleIdx = thrId % threadsPerScale;

    for (; tokBlkIdx < numWholeScales; tokBlkIdx += numScalesPerWave) {
      cute::copy(AutoVectorizingCopy{}, gQkVecs(_, threadInScaleIdx, tokBlkIdx), rQk);
      cute::transform(rQk_x2, rQkCompute_x2, NumericArrayConverter<float, Element, 2>::convert);

      float maxScale = 1e-3f;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(rQk); ++i) {
        maxScale = ::fmaxf(maxScale, ::fabsf(rQkCompute(i)));
      }
      CUTLASS_PRAGMA_UNROLL
      for (int delta = 1; delta < threadsPerScale; delta <<= 1) {
        maxScale = ::fmaxf(maxScale, __shfl_xor_sync(0xffffffffu, maxScale, delta));
      }

      maxScale /= TypeMax;
      gQkSeqScale(tokBlkIdx) = maxScale;
      Array<Element, 2> scaleQuant =
          NumericArrayConverter<Element, float, 2>::convert(Array<float, 2>{maxScale, maxScale});
      scaleQuant = cutlass::reciprocal_approximate<Array<Element, 2>>{}(scaleQuant);
      cutlass::multiplies<Array<Element, 2>> scaleQuantOp;
      cute::transform(rQk_x2, rQk_x2, [&](auto& x) { return scaleQuantOp(x, scaleQuant); });
      cute::transform(rQk_x4, rQkQuant_x4,
                      NumericArrayConverter<ElementQuantized, Element, 4>::convert);
      cute::copy(AutoVectorizingCopy{}, rQkQuant, gQkVecsQuant(_, threadInScaleIdx, tokBlkIdx));
    }

    int const lastIterTokenIdx = tokBlkIdx * TokenPerScale;
    if (lastIterTokenIdx < sumSeqLensQk) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<1>(rQk); ++i) {
        if (lastIterTokenIdx + i < sumSeqLensQk) {
          cute::copy(AutoVectorizingCopy{}, gQkVecs(make_tuple(_, i), threadInScaleIdx, tokBlkIdx),
                     rQk(_, i));
        } else {
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < BestVL; ++j) {
            rQk(j, i) = static_cast<Element>(0);
          }
        }
      }
      cute::transform(rQk_x2, rQkCompute_x2, NumericArrayConverter<float, Element, 2>::convert);

      float maxScale = 1e-3f;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(rQk); ++i) {
        maxScale = ::fmaxf(maxScale, ::fabsf(rQkCompute(i)));
      }
      CUTLASS_PRAGMA_UNROLL
      for (int delta = 1; delta < threadsPerScale; delta <<= 1) {
        maxScale = ::fmaxf(maxScale, __shfl_xor_sync(0xffffffffu, maxScale, delta));
      }

      maxScale /= TypeMax;
      gQkSeqScale(tokBlkIdx) = maxScale;
      Array<Element, 2> scaleQuant =
          NumericArrayConverter<Element, float, 2>::convert(Array<float, 2>{maxScale, maxScale});
      scaleQuant = cutlass::reciprocal_approximate<Array<Element, 2>>{}(scaleQuant);
      cutlass::multiplies<Array<Element, 2>> scaleQuantOp;
      cute::transform(rQk_x2, rQk_x2, [&](auto& x) { return scaleQuantOp(x, scaleQuant); });
      cute::transform(rQk_x4, rQkQuant_x4,
                      NumericArrayConverter<ElementQuantized, Element, 4>::convert);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<1>(rQk); ++i) {
        if (lastIterTokenIdx + i < sumSeqLensQk) {
          cute::copy(AutoVectorizingCopy{}, rQkQuant(_, i),
                     gQkVecsQuant(make_tuple(_, i), threadInScaleIdx, tokBlkIdx));
        }
      }
    }
  } else if (blockIdx.z == 1) {
    using ElementQuantizedV = cutlass::float_e4m3_t;
    constexpr int threadsPerHead = HeadDim / BestVL;
    static_assert(HeadDim % BestVL == 0, "VL must divide HeadDim");
    static_assert(threadsPerHead <= 32, "One token block should never exceed warp scope");
    Tensor gV = make_tensor(reinterpret_cast<Element const*>(ptrV),
                            make_shape(VL{}, Int<threadsPerHead>{}, numHeadsV, sumSeqLensV));
    Tensor gVQuant = make_tensor(reinterpret_cast<ElementQuantizedV*>(ptrVQuant),
                                 make_shape(VL{}, Int<threadsPerHead>{}, numHeadsV, sumSeqLensV));
    Tensor gVScale = make_tensor(ptrVScale, make_shape(VL{}, Int<threadsPerHead>{}, numHeadsV));

    Tensor rV = make_tensor<Element>(Shape<VL>{});
    Tensor rVMax = make_tensor<Element>(Shape<VL>{});
    Tensor rVQuant = make_tensor<ElementQuantizedV>(Shape<VL>{});
    Tensor rVScale = make_tensor<float>(Shape<VL>{});
    Tensor rVCompute = make_tensor<float>(Shape<VL>{});
    Tensor rV_x2 = recast<Array<Element, 2>>(rV);
    Tensor rVMax_x2 = recast<Array<Element, 2>>(rVMax);
    Tensor rVScale_x2 = recast<Array<float, 2>>(rVScale);
    Tensor rVCompute_x2 = recast<Array<float, 2>>(rVCompute);
    Tensor rVCompute_x4 = recast<Array<float, 4>>(rVCompute);
    Tensor rVQuant_x4 = recast<Array<ElementQuantizedV, 4>>(rVQuant);

    if (headIdx < numHeadsV) {
      int const numToksPerWarp = 32 / threadsPerHead;
      int tokIdx = warpId * numToksPerWarp + thrId / threadsPerHead;
      int const threadInTokIdx = thrId % threadsPerHead;
      Tensor gVSeq = gV(_, threadInTokIdx, headIdx, _);
      Tensor gVSeqQuant = gVQuant(_, threadInTokIdx, headIdx, _);
      Tensor gVSeqScale = gVScale(_, threadInTokIdx, headIdx);

      if constexpr (VStage == 1) {
        int const numWarpsToUse = cutlass::fast_min(numWarps, 256);
        int const numToksPerWave = numWarpsToUse * numToksPerWarp;
        if (warpId >= numWarpsToUse) {
          return;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(rVScale); ++i) {
          rVScale(i) = 1e-3f;
        }
        cute::transform(rVScale_x2, rVMax_x2,
                        cutlass::NumericArrayConverter<Element, float, 2>::convert);
        for (; tokIdx < sumSeqLensV; tokIdx += numToksPerWave) {
          cute::copy(AutoVectorizingCopy{}, gVSeq(_, tokIdx), rV);
          cute::transform(rV_x2, rV_x2, cutlass::absolute_value_op<Array<Element, 2>>{});
          cute::transform(rV_x2, rVMax_x2, rVMax_x2, cutlass::maximum<Array<Element, 2>>{});
        }
        cute::transform(rVMax_x2, rVScale_x2,
                        cutlass::NumericArrayConverter<float, Element, 2>::convert);
        cute::transform(rVScale_x2, rVScale_x2, cutlass::scale<Array<float, 2>>{1 / 448.0f});
        for (int delta = threadsPerHead; delta < 32; delta <<= 1) {
          cute::transform(rVScale, rVScale, [&](auto const& x) {
            return ::fmaxf(x, __shfl_xor_sync(0xffffffffu, x, delta));
          });
        }
        if (threadInTokIdx == thrId) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < BestVL; ++i) {
            atomicMax(reinterpret_cast<int32_t*>(&gVSeqScale(i)),
                      *reinterpret_cast<int32_t const*>(&rVScale(i)));
          }
        }
      } else if constexpr (VStage == 2) {
        int const numToksPerWave = numWarps * numToksPerWarp;
        cute::copy(AutoVectorizingCopy{}, gVSeqScale, rVScale);
        cute::transform(rVScale_x2, rVScale_x2, cutlass::reciprocal_approximate<Array<float, 2>>{});
        for (; tokIdx < sumSeqLensV; tokIdx += numToksPerWave) {
          cute::copy(AutoVectorizingCopy{}, gVSeq(_, tokIdx), rV);
          cute::transform(rV_x2, rVCompute_x2,
                          cutlass::NumericArrayConverter<float, Element, 2>::convert);
          cute::transform(rVCompute_x2, rVScale_x2, rVCompute_x2,
                          cutlass::multiplies<Array<float, 2>>{});
          cute::transform(rVCompute_x4, rVQuant_x4,
                          cutlass::NumericArrayConverter<ElementQuantizedV, float, 4>::convert);
          cute::copy(AutoVectorizingCopy{}, rVQuant, gVSeqQuant(_, tokIdx));
        }
      }
    }
  }
}

template <typename Element>
void invokeSageQuantQkvImpl(SageQuantParams const& params) {
  using namespace cute;
  FLASHINFER_CHECK(params.sumSeqLensQk > 0 && params.numHeads > 0 && params.headDim > 0 &&
                       params.tokenBlockSize > 0 && params.ptrQk != nullptr &&
                       params.ptrQkQuant != nullptr && params.ptrQkScale != nullptr &&
                       params.smCount > 0,
                   "Invalid SageQuantQk parameters");
  FLASHINFER_CHECK(params.vStage == 0 ||
                       (params.sumSeqLensV > 0 && params.numHeadsV > 0 && params.ptrV != nullptr &&
                        params.ptrVQuant != nullptr && params.ptrVScale != nullptr),
                   "Invalid SageQuantV parameters");
  FLASHINFER_CHECK(!params.kSmooth, "SageQuantQk K-smoothing is not supported yet");

  auto invokeKernel = [&](auto headDimStatic, auto tokenBlockSizeStatic) {
    constexpr int HeadDim_ = headDimStatic;
    constexpr int TokenBlockSize_ = tokenBlockSizeStatic;
    SageQuantParams kernelParams = params;
    void* kernelArgs[] = {&kernelParams.sumSeqLensQk, &kernelParams.ptrQk,
                          &kernelParams.ptrQkQuant,   &kernelParams.ptrQkScale,
                          &kernelParams.ptrKMean,     &kernelParams.sumSeqLensV,
                          &kernelParams.numHeadsV,    &kernelParams.ptrV,
                          &kernelParams.ptrVQuant,    &kernelParams.ptrVScale};

    auto launchWithVStage = [&](auto vStageStatic) {
      constexpr int VStage_ = vStageStatic;
      void const* kernelFunc = nullptr;
      if (params.quantType == DATA_TYPE_E4M3) {
        kernelFunc = reinterpret_cast<void const*>(
            sageQuantQkvKernel<Element, cutlass::float_e4m3_t, TokenBlockSize_, HeadDim_, false,
                               VStage_>);
      } else if (params.quantType == DATA_TYPE_INT8) {
        kernelFunc = reinterpret_cast<void const*>(
            sageQuantQkvKernel<Element, std::int8_t, TokenBlockSize_, HeadDim_, false, VStage_>);
      } else {
        FLASHINFER_ERROR("SageQuant Q/K output must be INT8 or FP8 E4M3");
      }
      uint32_t const gridX =
          static_cast<uint32_t>(std::max(1, (params.smCount * 32) / params.numHeads));
      dim3 const launchGrid{gridX, static_cast<uint32_t>(params.numHeads), VStage_ > 0 ? 2U : 1U};
      auto status =
          cudaLaunchKernel(kernelFunc, launchGrid, dim3{64U, 1U, 1U}, kernelArgs, 0, params.stream);
      FLASHINFER_CHECK(status == cudaSuccess, cudaGetErrorString(status));
      status = cudaPeekAtLastError();
      FLASHINFER_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    };

    switch (params.vStage) {
      case 0:
        launchWithVStage(Int<0>{});
        return;
      case 1:
        launchWithVStage(Int<1>{});
        return;
      case 2:
        launchWithVStage(Int<2>{});
        return;
      default:
        FLASHINFER_ERROR("Unsupported SageQuantV stage");
    }
  };

#define FLASHINFER_SAGE_DISPATCH_HEAD_DIM(HEAD_DIM) \
  if (params.headDim == HEAD_DIM) {                 \
    switch (params.tokenBlockSize) {                \
      case 1:                                       \
        invokeKernel(Int<HEAD_DIM>{}, Int<1>{});    \
        return;                                     \
      case 4:                                       \
        invokeKernel(Int<HEAD_DIM>{}, Int<4>{});    \
        return;                                     \
      case 16:                                      \
        invokeKernel(Int<HEAD_DIM>{}, Int<16>{});   \
        return;                                     \
      default:                                      \
        break;                                      \
    }                                               \
  }
  FLASHINFER_SAGE_DISPATCH_HEAD_DIM(64)
  FLASHINFER_SAGE_DISPATCH_HEAD_DIM(128)
  FLASHINFER_SAGE_DISPATCH_HEAD_DIM(256)
#undef FLASHINFER_SAGE_DISPATCH_HEAD_DIM
  FLASHINFER_ERROR(
      "Unsupported SageQuant dispatch config (head_dim must be 64, 128, or 256; "
      "token_block_size must be 1, 4, or 16)");
}

void invokeSageQuant(SageQuantParams const& params) {
  if (params.inputType == DATA_TYPE_FP16) {
    invokeSageQuantQkvImpl<cutlass::half_t>(params);
    return;
  }
  if (params.inputType == DATA_TYPE_BF16) {
    invokeSageQuantQkvImpl<cutlass::bfloat16_t>(params);
    return;
  }
  FLASHINFER_ERROR("SageQuant input must be FP16 or BF16");
}

}  // namespace flashinfer::trtllm
