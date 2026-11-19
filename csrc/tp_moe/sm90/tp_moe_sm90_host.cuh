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
// Host-side helpers shared by the SM90 TP-MoE entry points (tp_moe_sm90_flash.cu,
// tp_moe_sm90_dsv41.cu): tensor-map encoding, the per-rank workspace layout (one zero-initialized
// byte buffer carved into the kernel's scratch tensors).
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace tp_moe {
namespace sm90 {
namespace host {

using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                   const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                   const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                   CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

inline EncodeTiledFn encode_fn() {
  static EncodeTiledFn fn = nullptr;
  if (!fn) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    TVM_FFI_ICHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000,
                                                    cudaEnableDefault, &q) == cudaSuccess &&
                   p)
        << "cuTensorMapEncodeTiled not available";
    fn = reinterpret_cast<EncodeTiledFn>(p);
  }
  return fn;
}

// rank-D map without swizzle (the Humming weight / scale maps), element strides 1
inline void encode_map(CUtensorMap* m, const void* base, CUtensorMapDataType dt, uint32_t rank,
                       const cuuint64_t* dims, const cuuint64_t* strides, const cuuint32_t* box,
                       CUtensorMapL2promotion promo) {
  const cuuint32_t estr[5] = {1, 1, 1, 1, 1};
  const CUresult r = encode_fn()(m, dt, rank, const_cast<void*>(base), dims, strides, box, estr,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, promo,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(r == CUDA_SUCCESS) << "cuTensorMapEncodeTiled failed: " << static_cast<int>(r);
}

// 2-D uint8 map [rows][cols], box {128 B, box_rows}, 128B swizzle (out-of-bounds rows read as zero)
inline void encode_rows_sw128(CUtensorMap* m, const void* base, uint64_t cols, uint64_t rows,
                              uint32_t box_rows) {
  const cuuint64_t dims[2] = {cols, rows}, strides[1] = {cols};
  const cuuint32_t box[2] = {128, box_rows}, estr[2] = {1, 1};
  const CUresult r =
      encode_fn()(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box,
                  estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(r == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled (rows) failed: " << static_cast<int>(r);
}

// Kernel constants the workspace sizes depend on (each kernel header's constexprs)
struct Dims {
  int64_t maxm, maxe, maxp, inter, nqg, nkt2, nmax, ncons, nrt, topk_max, dim, nexp;
};

// One rank's scratch, carved from one zero-initialized buffer.  Flags and counters are epoch-tagged
// or parity-reset by the kernel itself (the epoch lives in the workspace: CUDA graph replays need
// no reset).
struct Workspace {
  int64_t act, amax, part, part_flags, rt_cnt, work, outp, hq, hs, h_flags, pan_cnt, xg, xg_cnt,
      fc2_next;
  int64_t xq, xs, tok_flags, gcnt, gids, grw, xqg, xsT, hs4, shout, bytes;
};

inline Workspace workspace_layout(const Dims& d, int64_t grid) {
  Workspace w{};
  int64_t off = 0;
  auto take = [&](int64_t bytes) {
    const int64_t at = off;
    off += (bytes + 1023) / 1024 * 1024;
    return at;
  };
  w.act = take(d.maxe * d.inter * 2);
  w.amax = take(d.maxe * d.nqg * 4);
  w.part = take(2 * grid * d.ncons * 32 * 4);
  w.part_flags = take((2 * grid > 256 ? 2 * grid : 256) * 4);
  w.rt_cnt = take(2 * d.nrt * 4);
  w.work = take(4 * 4);
  w.outp = take(d.maxm * d.topk_max * d.dim * 2);
  w.hq = take(d.maxp * d.nkt2 * d.nmax * 128);
  w.hs = take(d.maxp * d.nmax * 4);
  w.h_flags = take(d.maxp * 4);
  w.pan_cnt = take(2 * d.maxp * 4);
  w.xg = take((d.maxe + 8 * d.maxp) * d.dim);
  w.xg_cnt = take(2 * d.maxp * 4);
  w.fc2_next = take(4 * 4);
  w.xq = take(d.maxm * d.dim);
  w.xs = take(d.maxm * 4);
  w.tok_flags = take(d.maxm * 4);  // xflag [MAXM]
  w.gcnt = take(8 * 4);
  w.gids = take(d.maxm * 8 * 4);
  w.grw = take(d.maxm * 8 * 4);
  w.xqg = take(d.maxm * d.dim);
  w.xsT = take((d.dim / 128) * d.maxm * 4);
  w.hs4 = take(d.maxp * d.nkt2 * d.nmax * 4);
  w.shout = take(d.maxm * d.dim * 2);
  w.bytes = off;
  return w;
}

inline int sm_count(int dev) {
  static int sms[64] = {0};
  if (!sms[dev]) cudaDeviceGetAttribute(&sms[dev], cudaDevAttrMultiProcessorCount, dev);
  return sms[dev];
}

}  // namespace host
}  // namespace sm90
}  // namespace tp_moe
}  // namespace flashinfer
