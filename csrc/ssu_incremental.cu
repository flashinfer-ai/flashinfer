/*
 * Copyright (c) 2025 by FlashInfer team.
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
// clang-format off
// config.inc MUST come before the header: it defines DIM, DSTATE, NTOKENS_MTP
// constexprs that the header's function templates rely on.
#include "ssu_incremental_config.inc"
#include <flashinfer/mamba/ssu_incremental.cuh>
#include <flashinfer/mamba/kernel_ssu_incremental.cuh>
// clang-format on
#include "tvm_ffi_utils.h"

using namespace flashinfer;
using tvm::ffi::Optional;

namespace flashinfer::mamba::incremental {

void ssu_incremental(TensorView state,   // (cache, nheads, dim, dstate)
                     TensorView x,       // (batch, T, nheads, dim)
                     TensorView dt,      // (batch, T, nheads, dim) tie_hdim
                     TensorView A,       // (nheads, dim, dstate) tie_hdim
                     TensorView B,       // (batch, T, ngroups, dstate)
                     TensorView C,       // (batch, T, ngroups, dstate)
                     TensorView output,  // (batch, T, nheads, dim)
                     // Cache tensors
                     TensorView old_x,              // (cache, T, nheads, dim)
                     TensorView old_B,              // (cache, 2, T, ngroups, dstate)
                     TensorView old_dt_proc,        // (cache, 2, nheads, T) f32
                     TensorView old_cumAdt,         // (cache, 2, nheads, T) f32
                     TensorView cache_buf_idx,      // (cache,) int32
                     TensorView prev_num_accepted,  // (cache,) int32
                     // Optional tensors
                     Optional<TensorView> D,        // (nheads, dim)
                     Optional<TensorView> z,        // (batch, T, nheads, dim)
                     Optional<TensorView> dt_bias,  // (nheads, dim) tie_hdim
                     bool dt_softplus,
                     Optional<TensorView> state_batch_indices,  // (batch,) int32
                     int64_t pad_slot_id,
                     Optional<TensorView> state_scale,  // (cache, nheads, dim) f32
                     Optional<TensorView> rand_seed) {  // single int64

  // ── Extract dimensions ──
  auto const state_cache_size = state.size(0);
  auto const nheads = state.size(1);
  auto const dim = state.size(2);
  auto const dstate = state.size(3);
  auto const batch = x.size(0);
  auto const ntokens_mtp = x.size(1);
  auto const ngroups = B.size(2);

  // ── Validate state ──
  CHECK_CUDA(state);
  CHECK_DIM(4, state);
  {
    auto s = state.strides();
    auto sz = state.sizes();
    FLASHINFER_CHECK(s[3] == 1, "state dim 3 (dstate) must have stride 1, got ", s[3]);
    FLASHINFER_CHECK(s[2] == sz[3], "state dim 2 (dim) must be contiguous with dim 3, got stride ",
                     s[2], " expected ", sz[3]);
    FLASHINFER_CHECK(s[1] == sz[2] * sz[3],
                     "state dim 1 (nheads) must be contiguous with dim 2, got stride ", s[1],
                     " expected ", sz[2] * sz[3]);
  }

  // ── Validate x: (batch, T, nheads, dim) ──
  CHECK_CUDA(x);
  CHECK_DIM(4, x);
  FLASHINFER_CHECK(x.size(2) == nheads, "x.size(2)=", x.size(2), " must equal nheads=", nheads);
  FLASHINFER_CHECK(x.size(3) == dim, "x.size(3)=", x.size(3), " must equal dim=", dim);
  CHECK_LAST_DIM_CONTIGUOUS(x);

  // ── Validate dt: (batch, T, nheads, dim) tie_hdim ──
  CHECK_CUDA(dt);
  CHECK_DIM(4, dt);
  FLASHINFER_CHECK(dt.size(0) == batch, "dt.size(0)=", dt.size(0), " must equal batch=", batch);
  FLASHINFER_CHECK(dt.size(1) == ntokens_mtp, "dt.size(1)=", dt.size(1),
                   " must equal ntokens_mtp=", ntokens_mtp);
  FLASHINFER_CHECK(dt.size(2) == nheads, "dt.size(2)=", dt.size(2), " must equal nheads=", nheads);
  FLASHINFER_CHECK(dt.stride(2) == 1, "dt.stride(2) must be 1 (tie_hdim), got ", dt.stride(2));
  FLASHINFER_CHECK(dt.stride(3) == 0, "dt.stride(3) must be 0 (tie_hdim), got ", dt.stride(3));

  // ── Validate A: (nheads, dim, dstate) tie_hdim ──
  CHECK_CUDA(A);
  CHECK_DIM(3, A);
  FLASHINFER_CHECK(A.size(0) == nheads, "A.size(0)=", A.size(0), " must equal nheads=", nheads);
  FLASHINFER_CHECK(A.stride(0) == 1, "A.stride(0) must be 1, got ", A.stride(0));
  FLASHINFER_CHECK(A.stride(1) == 0, "A.stride(1) must be 0 (tie_hdim), got ", A.stride(1));
  FLASHINFER_CHECK(A.stride(2) == 0, "A.stride(2) must be 0 (tie_hdim), got ", A.stride(2));

  // ── Validate B: (batch, T, ngroups, dstate) ──
  CHECK_CUDA(B);
  CHECK_DIM(4, B);
  FLASHINFER_CHECK(B.size(0) == batch, "B.size(0)=", B.size(0), " must equal batch=", batch);
  FLASHINFER_CHECK(B.size(1) == ntokens_mtp, "B.size(1)=", B.size(1),
                   " must equal ntokens_mtp=", ntokens_mtp);
  FLASHINFER_CHECK(B.size(3) == dstate, "B.size(3)=", B.size(3), " must equal dstate=", dstate);
  CHECK_LAST_DIM_CONTIGUOUS(B);
  FLASHINFER_CHECK(nheads % ngroups == 0, "nheads=", nheads,
                   " must be divisible by ngroups=", ngroups);

  // ── Validate C: (batch, T, ngroups, dstate) ──
  CHECK_CUDA(C);
  CHECK_DIM(4, C);
  FLASHINFER_CHECK(C.size(0) == batch, "C.size(0)=", C.size(0), " must equal batch=", batch);
  FLASHINFER_CHECK(C.size(1) == ntokens_mtp, "C.size(1)=", C.size(1),
                   " must equal ntokens_mtp=", ntokens_mtp);
  FLASHINFER_CHECK(C.size(2) == ngroups, "C.size(2)=", C.size(2), " must equal ngroups=", ngroups);
  FLASHINFER_CHECK(C.size(3) == dstate, "C.size(3)=", C.size(3), " must equal dstate=", dstate);
  CHECK_LAST_DIM_CONTIGUOUS(C);

  // ── Validate output: (batch, T, nheads, dim) ──
  CHECK_CUDA(output);
  CHECK_DIM(4, output);
  FLASHINFER_CHECK(output.size(0) == batch, "output batch mismatch");
  FLASHINFER_CHECK(output.size(1) == ntokens_mtp, "output T mismatch");
  FLASHINFER_CHECK(output.size(2) == nheads, "output nheads mismatch");
  FLASHINFER_CHECK(output.size(3) == dim, "output dim mismatch");
  CHECK_LAST_DIM_CONTIGUOUS(output);

  // ── Validate cache tensors ──
  CHECK_CUDA(old_x);
  CHECK_DIM(4, old_x);  // (cache, T, nheads, dim)
  FLASHINFER_CHECK(old_x.size(0) == state_cache_size, "old_x cache size mismatch");
  FLASHINFER_CHECK(old_x.size(1) == ntokens_mtp, "old_x T mismatch");
  FLASHINFER_CHECK(old_x.size(2) == nheads, "old_x nheads mismatch");
  FLASHINFER_CHECK(old_x.size(3) == dim, "old_x dim mismatch");

  CHECK_CUDA(old_B);
  CHECK_DIM(5, old_B);  // (cache, 2, T, ngroups, dstate)
  FLASHINFER_CHECK(old_B.size(0) == state_cache_size, "old_B cache size mismatch");
  FLASHINFER_CHECK(old_B.size(1) == 2, "old_B.size(1) must be 2 (double-buffered), got ",
                   old_B.size(1));
  FLASHINFER_CHECK(old_B.size(2) == ntokens_mtp, "old_B T mismatch");
  FLASHINFER_CHECK(old_B.size(3) == ngroups, "old_B ngroups mismatch");
  FLASHINFER_CHECK(old_B.size(4) == dstate, "old_B dstate mismatch");
  CHECK_LAST_DIM_CONTIGUOUS(old_B);

  CHECK_CUDA(old_dt_proc);
  CHECK_DIM(4, old_dt_proc);  // (cache, 2, nheads, T)
  FLASHINFER_CHECK(old_dt_proc.size(0) == state_cache_size, "old_dt_proc cache size mismatch");
  FLASHINFER_CHECK(old_dt_proc.size(1) == 2, "old_dt_proc.size(1) must be 2, got ",
                   old_dt_proc.size(1));
  FLASHINFER_CHECK(old_dt_proc.size(2) == nheads, "old_dt_proc nheads mismatch");
  FLASHINFER_CHECK(old_dt_proc.size(3) == ntokens_mtp, "old_dt_proc T mismatch");
  CHECK_LAST_DIM_CONTIGUOUS(old_dt_proc);

  CHECK_CUDA(old_cumAdt);
  CHECK_DIM(4, old_cumAdt);  // (cache, 2, nheads, T)
  FLASHINFER_CHECK(old_cumAdt.size(0) == state_cache_size, "old_cumAdt cache size mismatch");
  FLASHINFER_CHECK(old_cumAdt.size(1) == 2, "old_cumAdt.size(1) must be 2, got ",
                   old_cumAdt.size(1));
  FLASHINFER_CHECK(old_cumAdt.size(2) == nheads, "old_cumAdt nheads mismatch");
  FLASHINFER_CHECK(old_cumAdt.size(3) == ntokens_mtp, "old_cumAdt T mismatch");
  CHECK_LAST_DIM_CONTIGUOUS(old_cumAdt);

  CHECK_CUDA(cache_buf_idx);
  CHECK_DIM(1, cache_buf_idx);
  FLASHINFER_CHECK(cache_buf_idx.size(0) == state_cache_size, "cache_buf_idx size mismatch");
  CHECK_CONTIGUOUS(cache_buf_idx);

  CHECK_CUDA(prev_num_accepted);
  CHECK_DIM(1, prev_num_accepted);
  FLASHINFER_CHECK(prev_num_accepted.size(0) == state_cache_size,
                   "prev_num_accepted size mismatch");
  CHECK_CONTIGUOUS(prev_num_accepted);

  // ── Validate optional D ──
  if (D.has_value()) {
    auto& Dv = D.value();
    CHECK_CUDA(Dv);
    CHECK_DIM(2, Dv);
    FLASHINFER_CHECK(Dv.size(0) == nheads, "D.size(0)=", Dv.size(0), " must equal nheads=", nheads);
    FLASHINFER_CHECK(Dv.stride(0) == 1, "D.stride(0) must be 1 (tie_hdim), got ", Dv.stride(0));
    FLASHINFER_CHECK(Dv.stride(1) == 0, "D.stride(1) must be 0 (tie_hdim), got ", Dv.stride(1));
  }

  // ── Validate optional dt_bias ──
  if (dt_bias.has_value()) {
    auto& db = dt_bias.value();
    CHECK_CUDA(db);
    CHECK_DIM(2, db);
    FLASHINFER_CHECK(db.size(0) == nheads, "dt_bias.size(0)=", db.size(0),
                     " must equal nheads=", nheads);
    FLASHINFER_CHECK(db.stride(0) == 1, "dt_bias.stride(0) must be 1 (tie_hdim), got ",
                     db.stride(0));
    FLASHINFER_CHECK(db.stride(1) == 0, "dt_bias.stride(1) must be 0 (tie_hdim), got ",
                     db.stride(1));
  }

  // ── Validate optional z ──
  if (z.has_value()) {
    auto& zv = z.value();
    CHECK_CUDA(zv);
    CHECK_DIM(4, zv);
    FLASHINFER_CHECK(zv.size(0) == batch, "z batch mismatch");
    FLASHINFER_CHECK(zv.size(1) == ntokens_mtp, "z T mismatch");
    FLASHINFER_CHECK(zv.size(2) == nheads, "z nheads mismatch");
    FLASHINFER_CHECK(zv.size(3) == dim, "z dim mismatch");
    CHECK_LAST_DIM_CONTIGUOUS(zv);
  }

  // ── Validate optional state_batch_indices ──
  if (state_batch_indices.has_value()) {
    auto& sbi = state_batch_indices.value();
    CHECK_CUDA(sbi);
    CHECK_DIM(1, sbi);
    FLASHINFER_CHECK(sbi.size(0) == batch, "state_batch_indices.size(0)=", sbi.size(0),
                     " must equal batch=", batch);
    CHECK_CONTIGUOUS(sbi);
  }

  // ── Populate params ──
  SsuIncrementalParams p;

  p.batch = batch;
  p.nheads = nheads;
  p.dim = dim;
  p.dstate = dstate;
  p.ngroups = ngroups;
  p.state_cache_size = state_cache_size;
  p.ntokens_mtp = ntokens_mtp;
  p.pad_slot_id = pad_slot_id;
  p.dt_softplus = dt_softplus;

  // Pointers
  p.state = state.data_ptr();
  p.x = const_cast<void*>(x.data_ptr());
  p.dt = const_cast<void*>(dt.data_ptr());
  p.A = const_cast<void*>(A.data_ptr());
  p.B = const_cast<void*>(B.data_ptr());
  p.C = const_cast<void*>(C.data_ptr());
  p.output = output.data_ptr();

  p.old_x = old_x.data_ptr();
  p.old_B = const_cast<void*>(old_B.data_ptr());
  p.old_dt_proc = const_cast<void*>(old_dt_proc.data_ptr());
  p.old_cumAdt = const_cast<void*>(old_cumAdt.data_ptr());
  p.cache_buf_idx = const_cast<void*>(cache_buf_idx.data_ptr());
  p.prev_num_accepted = const_cast<void*>(prev_num_accepted.data_ptr());

  if (D.has_value()) p.D = const_cast<void*>(D.value().data_ptr());
  if (z.has_value()) {
    p.z = const_cast<void*>(z.value().data_ptr());
    p.z_stride_batch = z.value().stride(0);
    p.z_stride_mtp = z.value().stride(1);
  }
  if (dt_bias.has_value()) p.dt_bias = const_cast<void*>(dt_bias.value().data_ptr());
  if (state_batch_indices.has_value())
    p.state_batch_indices = const_cast<void*>(state_batch_indices.value().data_ptr());
  if (state_scale.has_value()) {
    p.state_scale = state_scale.value().data_ptr();
    p.state_scale_stride_batch = state_scale.value().stride(0);
  }
  if (rand_seed.has_value()) {
    auto const& rs = rand_seed.value();
    CHECK_CUDA(rs);
    FLASHINFER_CHECK(rs.numel() == 1, "rand_seed must be single-element, got numel=", rs.numel());
    FLASHINFER_CHECK(rs.dtype().code == kDLInt && rs.dtype().bits == 64, "rand_seed must be int64");
    p.rand_seed = static_cast<const int64_t*>(rs.data_ptr());
  }

  // Strides
  p.state_stride_batch = state.stride(0);

  p.x_stride_batch = x.stride(0);
  p.x_stride_mtp = x.stride(1);
  p.dt_stride_batch = dt.stride(0);
  p.dt_stride_mtp = dt.stride(1);
  p.B_stride_batch = B.stride(0);
  p.B_stride_mtp = B.stride(1);
  p.C_stride_batch = C.stride(0);
  p.C_stride_mtp = C.stride(1);
  p.out_stride_batch = output.stride(0);
  p.out_stride_mtp = output.stride(1);

  p.old_x_stride_cache = old_x.stride(0);
  p.old_x_stride_mtp = old_x.stride(1);
  p.old_B_stride_cache = old_B.stride(0);
  p.old_B_stride_dbuf = old_B.stride(1);
  p.old_B_stride_mtp = old_B.stride(2);
  p.old_dt_proc_stride_cache = old_dt_proc.stride(0);
  p.old_dt_proc_stride_dbuf = old_dt_proc.stride(1);
  p.old_dt_proc_stride_head = old_dt_proc.stride(2);
  p.old_cumAdt_stride_cache = old_cumAdt.stride(0);
  p.old_cumAdt_stride_dbuf = old_cumAdt.stride(1);
  p.old_cumAdt_stride_head = old_cumAdt.stride(2);

  // Launch
  ffi::CUDADeviceGuard device_guard(state.device().device_id);
  const cudaStream_t stream = get_stream(state.device());

  launchSsuIncremental<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t, state_scale_t>(
      p, stream);
}

}  // namespace flashinfer::mamba::incremental
