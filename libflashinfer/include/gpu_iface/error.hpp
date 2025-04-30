// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache - 2.0

#pragma once
#include "platform.hpp"
#include <sstream>
#include <stdexcept>
#include <string>

namespace flashinfer
{
namespace gpu_iface
{

// Platform-agnostic error type
class GpuError
{
private:
    int code_;
    std::string message_;

public:
    GpuError() : code_(0) {}
    GpuError(int code, std::string message)
        : code_(code), message_(std::move(message))
    {
    }

    bool isSuccess() const { return code_ == 0; }
    int code() const { return code_; }
    const std::string &message() const { return message_; }

#if defined(PLATFORM_CUDA_DEVICE)
    cudaError_t getNative() const { return static_cast<cudaError_t>(code_); }
#elif defined(PLATFORM_HIP_DEVICE)
    hipError_t getNative() const { return static_cast<hipError_t>(code_); }
#endif
};

// Create error from message
inline GpuError CreateError(std::string message)
{
#if defined(PLATFORM_CUDA_DEVICE)
    return GpuError(static_cast<int>(cudaErrorUnknown), std::move(message));
#elif defined(PLATFORM_HIP_DEVICE)
    return GpuError(static_cast<int>(hipErrorUnknown), std::move(message));
#endif
}

} // namespace gpu_iface
} // namespace flashinfer
