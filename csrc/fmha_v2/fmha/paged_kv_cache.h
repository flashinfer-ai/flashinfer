/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cuda_runtime.h>
#include <math.h>

namespace fmha {

// This needs to be aligned with the definition in TRT-LLM
struct Kv_block_array {
  using PtrType = int32_t;

  // Maximum number of sequences supported by the kv-cache.
  int32_t mMaxSeqs;
  // Max number of blocks per sequence
  int32_t mMaxBlocksPerSeq;
  // Number of tokens. It must be power of 2.
  int32_t mTokensPerBlock;
  // Exponent of number of tokens with base 2.
  // E.g. for mTokensPerBlock 64, mTokensPerBlockLog2 equals to 6
  int32_t mTokensPerBlockLog2;
  // Table maps logical block idx to the data pointer of k/v cache block pool
  //
  // When mUsesSharedPagedKvIdx is false (default, TRT-LLM native format):
  //   Shape [B, W, 2, M], where 2 is table for K and V,
  //   B is current number of sequences, W is beam width,
  //   M is max number of blocks per sequence.
  //
  // When mUsesSharedPagedKvIdx is true (FlashInfer interleaved KV pool format):
  //   Shape [B, M] containing logical page indices.
  //   K and V share the same index; the kernel computes pool offsets on-the-fly as:
  //     K pool offset = page_idx * 2
  //     V pool offset = page_idx * 2 + 1

  // Size of KV cache blocks in bytes (H*D*T*sizeof(DataType))
  int32_t mBytesPerBlock;
  // Pointer to beginning of pool.
  void* mPoolPtr;
  // Pointer to block offsets.
  PtrType* mBlockOffsets;
  // When true, mBlockOffsets is [B, M] with shared K/V page indices that need
  // the *2/+1 transform, instead of pre-expanded [B, 2, M] separate offsets.
  bool mUsesSharedPagedKvIdx;

  Kv_block_array() = default;

  Kv_block_array(int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock,
                 int32_t bytesPerBlock, void* poolPtr)
      : mMaxSeqs(batchSize),
        mMaxBlocksPerSeq(maxBlocksPerSeq),
        mTokensPerBlock(tokensPerBlock),
        mBytesPerBlock{bytesPerBlock},
        mPoolPtr{poolPtr},
        mBlockOffsets{nullptr},
        mUsesSharedPagedKvIdx{false} {
    float const tokensPerBlockSeqLog2 = log2(mTokensPerBlock);
    mTokensPerBlockLog2 = static_cast<int>(tokensPerBlockSeqLog2);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fmha
