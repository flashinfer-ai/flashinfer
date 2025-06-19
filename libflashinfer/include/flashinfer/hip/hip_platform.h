#pragma once

#define HIP_ENABLE_WARP_SYNC_BUILTINS 1

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
// #include <hip/hip_cooperative_groups.h>

#if defined(FLASHINFER_ENABLE_BF16)
#include <hip/hip_bf16.h>
#endif

#if defined(FLASHINFER_ENABLE_F16)
#include <hip/hip_fp16.h>
#endif

#if defined(FLASHINFER_ENABLE_FP8)
#include <hip/hip_fp8.h>
#endif

#define FI_GPU_CALL(call)                                                      \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            std::ostringstream err_msg;                                        \
            err_msg << "GPU error: " << hipGetErrorString(err) << " at "       \
                    << __FILE__ << ":" << __LINE__;                            \
            throw std::runtime_error(err_msg.str());                           \
        }                                                                      \
    } while (0)
