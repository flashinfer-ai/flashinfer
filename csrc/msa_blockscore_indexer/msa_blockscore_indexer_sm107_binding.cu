/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// MSA (MiniMax-style sparse attention) stage-1 block-score top-k indexer,
// SM107 (Rubin).  Dispatch between the two kernel families: the staged
// score/select path serves every shape; the fully fused single-pass path
// serves the large multi-request long-context shape where the staged score
// plane is wasteful.  Both paths implement the identical contract (per-block
// causal MAX score, forced own block + top-15 strictly-prior blocks,
// deterministic lower-block-id tie-break, ascending int32 ids with -1 tail).

#include <tvm/ffi/tvm_ffi.h>

namespace ffi = tvm::ffi;

namespace c64_regbuf_impl {
void Run(ffi::TensorView qi, ffi::TensorView ki, ffi::TensorView cu_seqlens,
         ffi::TensorView kv_base, ffi::TensorView seq_lens, ffi::TensorView prefix_lens,
         ffi::TensorView sel);
}

namespace msa_fused {
void Run(ffi::TensorView qi, ffi::TensorView ki, ffi::TensorView cu_seqlens,
         ffi::TensorView kv_base, ffi::TensorView seq_lens, ffi::TensorView prefix_lens,
         ffi::TensorView sel);
}

void Run(ffi::TensorView qi, ffi::TensorView ki, ffi::TensorView cu_seqlens,
         ffi::TensorView kv_base, ffi::TensorView seq_lens, ffi::TensorView prefix_lens,
         ffi::TensorView sel) {
  const int tq = (int)qi.shape()[0];
  const int tkpad = (int)ki.shape()[0];
  const int batch = (int)kv_base.shape()[0];
  if (tq == 16384 && batch == 4 && tkpad == 278528)
    msa_fused::Run(qi, ki, cu_seqlens, kv_base, seq_lens, prefix_lens, sel);
  else
    c64_regbuf_impl::Run(qi, ki, cu_seqlens, kv_base, seq_lens, prefix_lens, sel);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
