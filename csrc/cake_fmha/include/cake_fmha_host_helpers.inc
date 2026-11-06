/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host helpers shared verbatim by the Cake FMHA JIT adapters.
//
// This is a textual fragment, not a self-contained header: every adapter
// includes it from inside its anonymous namespace
// (``namespace flashinfer { namespace cake_fmha { namespace {``), after its
// own includes and ``using tvm::ffi::TensorView;``, at the exact position
// where it used to define these helpers inline.  The fragment therefore has
// no includes and opens no namespace of its own, and its body is the
// adapters' former text unchanged, so each adapter's preprocessed translation
// unit is the same as it was with the inline copy.  Requires
// ``tvm_ffi_utils.h`` (``TensorView``, ``TVM_FFI_ICHECK_*``) in scope.

#pragma once

void CheckSameDevice(TensorView query, TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(query.device().device_type, tensor.device().device_type)
      << name << " must be on the query device";
  TVM_FFI_ICHECK_EQ(query.device().device_id, tensor.device().device_id)
      << name << " must be on the query device";
}
