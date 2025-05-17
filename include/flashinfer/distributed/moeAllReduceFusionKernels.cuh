// adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/communicationKernels/moeAllReduceFusionKernels.h

#pragma once
// #include <NvInferRuntime.h> // we can remove this since it's about datatype
#include "flashinfer/distributed/trtllm/types.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// #include "flashinfer/distributed/trtllm/common/assert.h"
// #include "flashinfer/distributed/trtllm/common/cudaUtils.h"
// #include "flashinfer/distributed/trtllm/kernels/quantization.h"
// #include "flashinfer/distributed/trtllm/runtime/ipcUtils.h"

namespace tensorrt_llm::kernels::ar_fusion::moe
{
static constexpr int kElemsPerAccess = 8;
static constexpr int kOneShotMaxToken = 128;
static constexpr int kBarrierFlagCount = 256;

// DS R1
// pattern1: AR+Add_RMS+Quant
// [m, 7168] bf16 allreduce_in, [m, 7168] bf16 residual_in
// [m, 7168] bf16 residual_out, [m, 7168] fp4 quant_out
// pattern2: AR+AddRMS
// [m, 7168] bf16 allreduce_in, [m, 7168] bf16 residual_in
// [m, 7168] bf16 norm_out
struct AllReduceFusionParams
{
    int nranks;
    int rank;
    DataType dtype;
    // size = token_num * hidden_dim
    int size;
    int hidden_dim;
    void** workspace;
    void* allreduce_in;
    void* residual_in;
    void* residual_out;
    void* norm_out;
    void* quant_out;
    void* scale_out;
    void* rms_gamma;
    float rms_eps;
    float* scale_factor;
    FP4QuantizationSFLayout layout = FP4QuantizationSFLayout::SWIZZLED;
    cudaStream_t stream;
};

/////////////////////////////////////////////////////////////////
//                  * MoE Reduction Fusion *                   //
/////////////////////////////////////////////////////////////////

// Fuse MoE Reduction before AR + RMS
// pattern1: MoE Reduction + Add Residual + AR + ADD_RMS + Quant
// pattern2: MoE Reduction + Add Residual + AR + ADD_RMS
// [device_num_experts, m, 7168] bf16 moe_reduction_active_experts_token_input
// [m, 7168] bf16 moe_reduction_token_input
// [device_num_experts, m] moe_reduction_scale_input
struct MoeReductionAllReduceFusionParams : public AllReduceFusionParams
{
    // * moe reduction specific params
    // Refer to kernel implementation on layout of those params
    // number of active experts on current device
    int* moe_reduction_device_num_experts = nullptr;
    // per token per expert fp32 scale
    float* moe_reduction_scale_input = nullptr;
    // per token per expert input
    void* moe_reduction_active_experts_token_input = nullptr;
    // per token input
    void* moe_reduction_token_input = nullptr;
};

void moereduction_allreduce_fusion_op(MoeReductionAllReduceFusionParams const& params);

} // namespace tensorrt_llm::kernels::ar_fusion::moe
