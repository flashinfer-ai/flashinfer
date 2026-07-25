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

// TVM-FFI binding + launcher for the SM120/SM121 MXFP8 / per-tensor-FP8 ragged prefill
// kernel (warp-specialized persistent kernel in
// include/flashinfer/attention/sm120/mxfp8_attention_sm120/kernel.cuh).
//
// Contract (all CUDA tensors, caller pre-pads each request to 128-row multiples and
// pre-builds the LPT work lists, see flashinfer/mxfp8_attention_sm120.py):
//   q / k / v : float8_e4m3fn ragged [Sq_pad, Hq, D] / [Sk_pad, Hkv, D] / [Sk_pad, Hkv, D]
//   work_indptr [num_sm+1], head_indices / qo_tile_indices / qo_indptr / kv_indptr /
//     qo_lens / kv_lens / batch_indices: int32 per-work-item arrays (LPT order)
//   o : float32 [Sq_pad, Hq, D]; lse / l : float32 head-major [Hq, Sq_pad]
//   sm_scale : full score scale, host-folded (softmax_scale * q_scale * k_scale)
//   o_scale  : per-tensor v_scale (folded into the PV output by the kernel)

#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <flashinfer/attention/sm120/mxfp8_attention_sm120/kernel.cuh>

using tvm::ffi::TensorView;
namespace ffi = tvm::ffi;

constexpr DLDataType dl_float32 = DLDataType{kDLFloat, 32, 1};
constexpr DLDataType dl_int32 = DLDataType{kDLInt, 32, 1};
constexpr DLDataType dl_float8_e4m3fn = DLDataType{kDLFloat8_e4m3fn, 8, 1};

#define CHECK_CUDA(x) \
  TVM_FFI_ICHECK_EQ(x.device().device_type, kDLCUDA) << #x " must be a CUDA tensor";
#define CHECK_CONTIGUOUS(x) TVM_FFI_ICHECK(x.IsContiguous()) << #x " must be contiguous";
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_DIM(d, x) TVM_FFI_ICHECK_EQ(x.ndim(), d) << #x " must be a " #d "D tensor";

inline cudaStream_t get_stream(DLDevice device) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
}

namespace flashinfer {
namespace mxfp8_attention_sm120 {

namespace {

void run_fwd(TensorView q, TensorView k, TensorView v, TensorView work_indptr,
             TensorView head_indices, TensorView qo_tile_indices, TensorView qo_indptr,
             TensorView kv_indptr, TensorView qo_lens, TensorView kv_lens, TensorView batch_indices,
             TensorView o, TensorView lse, TensorView l, int64_t Hq_, int64_t Hkv_, double sm_scale,
             double o_scale, bool causal, int num_sm, cudaStream_t stream) {
  const int Hq = int(Hq_), Hkv = int(Hkv_), group = Hq / Hkv;
  const int Sq_pad = int(q.size(0)), Sk_pad = int(k.size(0));
  const int HD = kHeadDim;

  Element* dQ = reinterpret_cast<Element*>(q.data_ptr());
  Element* dK = reinterpret_cast<Element*>(k.data_ptr());
  Element* dV = reinterpret_cast<Element*>(v.data_ptr());
  // kUniformFp8: no SF tensors exist; the descriptors below point at o purely as a valid
  // aligned address (never dereferenced -- the kernel skips every SF TMA load).
  ElementSF* dSF = reinterpret_cast<ElementSF*>(o.data_ptr());

  auto layoutSFQ = BlkSF::tile_atom_to_shape_SFA(make_shape(Sq_pad, int(kBlockN), HD, Hq));
  auto layoutSFK = BlkSF::tile_atom_to_shape_SFA(make_shape(Sk_pad, int(kBlockN), HD, Hkv));
  auto layoutSFV = BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), HD, Sk_pad, Hkv));
  Tensor mQ =
      make_tensor(make_gmem_ptr(dQ), make_shape(Sq_pad, HD, Hq), make_stride(Hq * HD, _1{}, HD));
  Tensor mK =
      make_tensor(make_gmem_ptr(dK), make_shape(Sk_pad, HD, Hkv), make_stride(Hkv * HD, _1{}, HD));
  // V arrives physically transposed as [Hkv, D, Sk_pad]: the smem V tile atom is
  // Sk-major (shared with K), so the TMA traversal needs Sk-contiguous gmem. The
  // natural [Sk, Hkv, D] layout makes the gmem Sk mode non-contiguous, which
  // cuTensorMapEncodeTiled rejects (dim0 must be contiguous) -- verified empirically;
  // an HD-major smem atom + ldmatrix.trans would be needed to drop the transpose.
  Tensor mV = make_tensor(make_gmem_ptr(dV), make_shape(HD, Sk_pad, Hkv),
                          make_stride(Sk_pad, _1{}, HD * Sk_pad));
  Tensor mSFQ = make_tensor(make_gmem_ptr(dSF), layoutSFQ);
  Tensor mSFK = make_tensor(make_gmem_ptr(dSF), layoutSFK);
  Tensor mSFV = make_tensor(make_gmem_ptr(dSF), layoutSFV);

  Params p;
  p.tma_q = make_tma_copy(SM90_TMA_LOAD{}, mQ, SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{});
  p.tma_k = make_tma_copy(SM90_TMA_LOAD{}, mK, SmemLayoutK{}(_, _, _0{}),
                          select<1, 2>(TileShape_MNK{}), _1{});
  p.tma_v = make_tma_copy(SM90_TMA_LOAD{}, mV, SmemLayoutVt{},
                          make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), _1{});
  p.tma_sfq = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFQ, SmemLayoutSFQ{},
                                      make_shape(Int<kBlockM>{}, Int<kSFPadHD>{}), _1{});
  p.tma_sfk = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFK, SmemLayoutSFK{}(_, _, _0{}),
                                      make_shape(Int<kSFBlockN>{}, Int<kSFPadHD>{}), _1{});
  p.tma_sfv = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFV, SmemLayoutSFV{},
                                      make_shape(Int<kSFPadHD>{}, Int<kSFBlockN>{}), _1{});
  p.layout_sfq = layoutSFQ;
  p.layout_sfv = layoutSFV;
  p.seqlen_q = Sq_pad;
  p.seqlen_k = Sk_pad;
  p.n_block_total = Sk_pad / kBlockN;
  p.sm_scale = float(sm_scale);
  p.o_scale = float(o_scale);
  p.num_qo_heads = Hq;
  p.num_kv_heads = Hkv;
  p.tile_kv_len = nullptr;
  p.out_O = reinterpret_cast<float*>(o.data_ptr());
  p.out_lse = reinterpret_cast<float*>(lse.data_ptr());
  p.out_l = reinterpret_cast<float*>(l.data_ptr());
  p.out_Ppre = nullptr;
  p.out_Mnb = nullptr;
  p.out_dbg = nullptr;

  using Sched = flashinfer::BatchPrefillPersistentTileScheduler<int>;
  Sched::Arguments sa;
  sa.work_indptr = reinterpret_cast<int*>(work_indptr.data_ptr());
  sa.head_indices = reinterpret_cast<int*>(head_indices.data_ptr());
  sa.qo_tile_indices = reinterpret_cast<int*>(qo_tile_indices.data_ptr());
  sa.qo_indptr = reinterpret_cast<int*>(qo_indptr.data_ptr());
  sa.kv_indptr = reinterpret_cast<int*>(kv_indptr.data_ptr());
  sa.qo_lens = reinterpret_cast<int*>(qo_lens.data_ptr());
  sa.kv_lens = reinterpret_cast<int*>(kv_lens.data_ptr());
  sa.batch_indices = reinterpret_cast<int*>(batch_indices.data_ptr());
  sa.group_size_fastdiv = cutlass::FastDivmod(group);
  sa.num_qo_heads = Hq;
  dim3 grid = Sched::get_grid_dim(sa, num_sm);
  typename Sched::Params sp = Sched::to_underlying_arguments(sa);
  int smem = int(sizeof(SharedStorage));

  if (causal) {
    cudaFuncSetAttribute(s3_kernel<Sched, true, SFSource::kUniformFp8>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    s3_kernel<Sched, true, SFSource::kUniformFp8><<<grid, kNThreads, smem, stream>>>(p, sp);
  } else {
    cudaFuncSetAttribute(s3_kernel<Sched, false, SFSource::kUniformFp8>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    s3_kernel<Sched, false, SFSource::kUniformFp8><<<grid, kNThreads, smem, stream>>>(p, sp);
  }
}

}  // namespace

void fwd(TensorView q, TensorView k, TensorView v, TensorView work_indptr, TensorView head_indices,
         TensorView qo_tile_indices, TensorView qo_indptr, TensorView kv_indptr, TensorView qo_lens,
         TensorView kv_lens, TensorView batch_indices, TensorView o, TensorView lse, TensorView l,
         double sm_scale, double o_scale, bool causal) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(work_indptr);
  CHECK_INPUT(head_indices);
  CHECK_INPUT(qo_tile_indices);
  CHECK_INPUT(qo_indptr);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(qo_lens);
  CHECK_INPUT(kv_lens);
  CHECK_INPUT(batch_indices);
  CHECK_INPUT(o);
  CHECK_INPUT(lse);
  CHECK_INPUT(l);

  CHECK_DIM(3, q);
  CHECK_DIM(3, k);
  CHECK_DIM(3, v);
  CHECK_DIM(1, work_indptr);
  CHECK_DIM(1, head_indices);
  CHECK_DIM(1, qo_tile_indices);
  CHECK_DIM(1, qo_indptr);
  CHECK_DIM(1, kv_indptr);
  CHECK_DIM(1, qo_lens);
  CHECK_DIM(1, kv_lens);
  CHECK_DIM(1, batch_indices);
  CHECK_DIM(3, o);
  CHECK_DIM(2, lse);
  CHECK_DIM(2, l);

  TVM_FFI_ICHECK(q.dtype() == dl_float8_e4m3fn) << "q must be float8_e4m3fn";
  TVM_FFI_ICHECK(k.dtype() == dl_float8_e4m3fn) << "k must be float8_e4m3fn";
  TVM_FFI_ICHECK(v.dtype() == dl_float8_e4m3fn) << "v must be float8_e4m3fn";
  TVM_FFI_ICHECK(o.dtype() == dl_float32) << "o must be float32";
  TVM_FFI_ICHECK(lse.dtype() == dl_float32) << "lse must be float32";
  TVM_FFI_ICHECK(l.dtype() == dl_float32) << "l must be float32";
  TVM_FFI_ICHECK(work_indptr.dtype() == dl_int32) << "work_indptr must be int32";
  TVM_FFI_ICHECK(head_indices.dtype() == dl_int32) << "head_indices must be int32";
  TVM_FFI_ICHECK(qo_tile_indices.dtype() == dl_int32) << "qo_tile_indices must be int32";
  TVM_FFI_ICHECK(qo_indptr.dtype() == dl_int32) << "qo_indptr must be int32";
  TVM_FFI_ICHECK(kv_indptr.dtype() == dl_int32) << "kv_indptr must be int32";
  TVM_FFI_ICHECK(qo_lens.dtype() == dl_int32) << "qo_lens must be int32";
  TVM_FFI_ICHECK(kv_lens.dtype() == dl_int32) << "kv_lens must be int32";
  TVM_FFI_ICHECK(batch_indices.dtype() == dl_int32) << "batch_indices must be int32";

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  int num_sm = 0;
  cudaError_t status =
      cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, q.device().device_id);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaDeviceGetAttribute failed: " << cudaGetErrorString(status);
  int major = 0, minor = 0;
  status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, q.device().device_id);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaDeviceGetAttribute failed: " << cudaGetErrorString(status);
  status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, q.device().device_id);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaDeviceGetAttribute failed: " << cudaGetErrorString(status);
  TVM_FFI_ICHECK(major == 12 && (minor == 0 || minor == 1))
      << "MXFP8 attention SM120 kernel requires compute capability 12.0/12.1";

  const int64_t Hq = q.size(1), Hkv = k.size(1), D = q.size(2);
  TVM_FFI_ICHECK(D == kHeadDim) << "head_dim must be " << kHeadDim << ", got " << D;
  TVM_FFI_ICHECK(v.size(0) == Hkv && v.size(1) == D && v.size(2) == k.size(0))
      << "v must be transposed [num_kv_heads, head_dim, Sk_pad]";
  TVM_FFI_ICHECK(k.size(2) == D) << "k head_dim must match q";
  TVM_FFI_ICHECK(Hkv >= 1 && Hq % Hkv == 0) << "GQA group must divide evenly";
  TVM_FFI_ICHECK(q.size(0) % 128 == 0 && k.size(0) % 128 == 0)
      << "padded token totals must be 128-multiples";
  TVM_FFI_ICHECK(o.size(0) == q.size(0) && o.size(1) == Hq && o.size(2) == D)
      << "o must be [Sq_pad, Hq, D]";
  TVM_FFI_ICHECK(lse.size(0) == Hq && lse.size(1) == q.size(0))
      << "lse must be [Hq, Sq_pad] (head-major)";
  TVM_FFI_ICHECK(l.size(0) == Hq && l.size(1) == q.size(0)) << "l must be [Hq, Sq_pad]";
  TVM_FFI_ICHECK(work_indptr.size(0) == num_sm + 1)
      << "work_indptr must have num_sm+1 = " << (num_sm + 1) << " entries";
  const int64_t n_work = head_indices.size(0);
  TVM_FFI_ICHECK(qo_tile_indices.size(0) == n_work && qo_indptr.size(0) == n_work &&
                 kv_indptr.size(0) == n_work && qo_lens.size(0) == n_work &&
                 kv_lens.size(0) == n_work && batch_indices.size(0) == n_work)
      << "per-work-item arrays must have the same length as head_indices";

  if (n_work == 0) {
    return;
  }

  cudaStream_t stream = get_stream(q.device());
  run_fwd(q, k, v, work_indptr, head_indices, qo_tile_indices, qo_indptr, kv_indptr, qo_lens,
          kv_lens, batch_indices, o, lse, l, Hq, Hkv, sm_scale, o_scale, causal, num_sm, stream);
}

}  // namespace mxfp8_attention_sm120
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fwd, flashinfer::mxfp8_attention_sm120::fwd);
