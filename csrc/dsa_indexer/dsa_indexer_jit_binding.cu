// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project

#include "dsa_indexer.cu"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

void dsa_indexer_seed_prep(TensorView slog, int64_t num_buckets, int64_t topk, int64_t cand_cap,
                           int64_t emit_limit, double headroom, int64_t probe_stride_tok,
                           int64_t hist_stride, TensorView origin, TensorView inv_delta,
                           TensorView th_bucket, TensorView bcount, TensorView cand_val,
                           TensorView cand_idx, TensorView cand_cnt);

void dsa_indexer_scan(TensorView q, TensorView kv, TensorView kv_scales, TensorView weights,
                      TensorView cu_start, TensorView cu_end, TensorView origin,
                      TensorView inv_delta, TensorView th_bucket, TensorView cand_val,
                      TensorView cand_idx, TensorView cand_cnt, TensorView bcount,
                      int64_t num_buckets, int64_t topk, int64_t refresh_every,
                      int64_t num_kv_splits_override, int64_t probe_group, int64_t probe_add_max);

void dsa_indexer_select(TensorView cand_val, TensorView cand_idx, TensorView cand_cnt,
                        TensorView origin, TensorView inv_delta, TensorView th_bucket,
                        int64_t num_buckets, int64_t topk, int64_t probe_group,
                        int64_t probe_add_max, Optional<TensorView> seed_base, TensorView out_val,
                        TensorView out_idx);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(seed_prep, dsa_indexer_seed_prep);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(scan, dsa_indexer_scan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(select, dsa_indexer_select);
