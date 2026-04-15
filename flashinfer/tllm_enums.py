from enum import IntEnum
import torch
from typing import Optional


# The type of method in top-K routing, for use in torch custom op
# Please keep this in sync with the counterpart defined in include/flashinfer/trtllm/fused_moe/runner.h
class RoutingMethodType(IntEnum):
    # Default: Softmax -> TopK
    Default = (0,)
    # Renormalize: TopK -> Softmax
    Renormalize = (1,)
    # DeepSeekV3: Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts from the Top4 groups
    DeepSeekV3 = (2,)
    # Llama4: Top1 -> Sigmoid
    Llama4 = (3,)
    # Qwen3: Softmax -> TopK -> Renormalize
    RenormalizeNaive = (4,)
    # TopK only (no softmax)
    TopK = (5,)
    # SigmoidRenorm: Sigmoid -> TopK -> Renormalize (divide by sum of top-K weights)
    SigmoidRenorm = (6,)
    # MiniMax2: Sigmoid + Bias -> TopK -> ScaledSumNormalize (routeScale=1.0, epsilon=1e-20)
    MiniMax2 = (7,)
    # Unspecified
    Unspecified = (8,)


# Copied from csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/common.h
class ActivationType(IntEnum):
    Gelu = 0
    Relu = 1
    Silu = 2
    Swiglu = 3
    Geglu = 4
    SwigluBias = 5
    Relu2 = 6
    Identity = 7
    InvalidType = 8


class DtypeTrtllmGen(IntEnum):
    def __new__(cls, block_format_bit, signed_bit, integer_bit, num_bits, uid):
        value = (
            (block_format_bit << 24)
            | (signed_bit << 20)
            | (integer_bit << 16)
            | (num_bits << 8)
            | uid
        )
        obj = int.__new__(cls, value)
        obj._value_ = value
        return obj

    # keep the values in sync with include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h
    Bfloat16 = (0, 1, 0, 16, 0)
    Bool = (0, 0, 1, 1, 1)
    E2m1 = (1, 1, 0, 4, 2)
    E2m3 = (1, 1, 0, 6, 3)
    E3m2 = (1, 1, 0, 6, 4)
    E4m3 = (0, 1, 0, 8, 5)
    E5m2 = (0, 1, 0, 8, 6)
    Fp16 = (0, 1, 0, 16, 7)
    Fp32 = (0, 1, 0, 32, 8)
    Int8 = (0, 1, 1, 8, 9)
    Int32 = (0, 1, 1, 32, 10)
    Int64 = (0, 1, 1, 64, 11)
    MxE2m1 = (1, 1, 0, 4, 12)
    MxE4m3 = (1, 1, 0, 8, 13)
    MxInt4 = (1, 1, 1, 4, 14)
    UE8m0 = (0, 0, 0, 8, 15)
    UInt8 = (0, 0, 1, 8, 16)
    UInt16 = (0, 0, 1, 16, 17)
    UInt32 = (0, 0, 1, 32, 18)
    UInt64 = (0, 0, 1, 64, 19)
    UInt128 = (0, 0, 1, 128, 20)
    Void = (0, 1, 0, 0, 21)


def trtllm_gen_dtype_has_scale(dtype: DtypeTrtllmGen) -> bool:
    if dtype in [
        DtypeTrtllmGen.E2m1,
        DtypeTrtllmGen.MxE2m1,
        DtypeTrtllmGen.MxE4m3,
        DtypeTrtllmGen.MxInt4,
    ]:
        return True
    else:
        return False


def deduce_trtllm_gen_tensor_dtype(
    x: torch.Tensor, scale: Optional[torch.Tensor]
) -> DtypeTrtllmGen:
    hidden_size = x.shape[-1]
    if x.dtype == torch.uint8:  # FIXME(siyuan): use torch.float4_e2m1x2 after torch 2.8
        hidden_size *= 2
    if x.dtype == torch.bfloat16:
        dtype = DtypeTrtllmGen.Bfloat16
    elif x.dtype == torch.float8_e4m3fn:
        dtype = DtypeTrtllmGen.E4m3 if scale is None else DtypeTrtllmGen.MxE4m3
    elif (
        x.dtype == torch.uint8
    ):  # FIXME(siyuan): use torch.float4_e2m1x2 after torch 2.8
        assert scale is not None, "Scale tensor must be provided for float4x2 input"
        if scale.shape[-1] == hidden_size // 16:
            dtype = DtypeTrtllmGen.E2m1
        else:
            dtype = DtypeTrtllmGen.MxE2m1
    else:
        raise ValueError("Unsupported trtllm-gen input tensor.")
    return dtype


# Please keep the values in sync with include/flashinfer/fp4_layout.cuh
class SfLayout(IntEnum):
    """
    Layout of scale factors for quantization.
    """

    layout_128x4 = 0
    layout_8x4 = 1
    layout_linear = 2


# See MatrixLayout from include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/Enums.h
class WeightLayout(IntEnum):
    # K-major layout (default). [Mn, K]
    MajorK = 0
    # M-major for A and N-major for B. [K, Mn]
    MajorMn = 1
    # Layout is blocked along the K dimension. [K / blockK, Mn, blockK]
    # where blockK is fixed at 128B
    BlockMajorK = 2


# The type of gated activation function
# Please keep this in sync with the counterpart defined in include/flashinfer/trtllm/fused_moe/runner.h
class GatedActType(IntEnum):
    # SwiGlu
    SwiGlu = 0
    # GeGlu
    GeGlu = 1


# The type of FP8 quantization
# Please keep this in sync with the counterpart defined in trtllm_fused_moe_kernel_launcher.cu
class Fp8QuantizationType(IntEnum):
    # No FP8 quantization
    NoneFp8 = 0
    # DeepSeek FP8
    DeepSeekFp8 = 1
    # MxFp8 x MxFp8
    MxFp8 = 2
