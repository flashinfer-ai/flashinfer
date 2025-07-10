// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace flashinfer
{

/*!
 * \brief An enumeration class that defines different modes for applying RoPE
 *   (Rotary Positional Embeddings).
 */
enum class PosEncodingMode
{
    // No rotary positional embeddings
    kNone = 0U,
    // Apply Llama-style rope.
    kRoPELlama = 1U,
    // Apply ALiBi bias
    kALiBi = 2U
};

enum class MaskMode
{
    kNone = 0U,   // No mask
    kCausal = 1U, // Causal mask
    kCustom = 2U, // Custom mask
};

} // namespace flashinfer
