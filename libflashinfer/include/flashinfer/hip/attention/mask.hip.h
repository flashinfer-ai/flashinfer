// SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#pragma once
#ifndef FLASHINFER_ATTENTION_MASK_CUH_
#define FLASHINFER_ATTENTION_MASK_CUH_

namespace flashinfer
{

enum class MaskMode
{
    kNone = 0U,   // No mask
    kCausal = 1U, // Causal mask
    kCustom = 2U, // Custom mask
};

} // namespace flashinfer

#endif // FLASHINFER_ATTENTION_MASK_CUH_
