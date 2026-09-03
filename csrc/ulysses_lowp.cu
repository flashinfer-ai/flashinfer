/*
 * Copyright (c) 2026 by NVIDIA Corporation.
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

// TVM-FFI host bindings for the low-precision Ulysses A2A stack (stats
// protocol 3 / ALIGN-128): fused K-sum/V-amax statistics, V2-G global-grid
// INT8/FP8 quant-and-pack, and the receiver-side unpack. Kernel logic lives
// in include/flashinfer/comm/ulysses_lowp.cuh.
//
// V2-G global-grid (payload ABI v3): V2-G keeps ordinary Sage2's global
// 32/64-token quantization grids across rank boundaries.  Under ALIGN-128
// (local_sequence % 128 == 0) no quantization group straddles a rank
// boundary, so every rank's locally computed grouped amax is already the
// FINAL per-group scale -- no cross-rank scale merge exists.

#include <cstdint>

#include "flashinfer/comm/ulysses_lowp.cuh"
#include "tvm_ffi_utils.h"

namespace lowp = flashinfer::ulysses_lowp;

namespace {

void check_shape_2d(const TensorView& t, const char* name, int64_t s0, int64_t s1) {
  TVM_FFI_ICHECK(t.size(0) == s0 && t.size(1) == s1)
      << name << " has shape (" << t.size(0) << ", " << t.size(1) << "), expected (" << s0 << ", "
      << s1 << ")";
}

void check_shape_3d(const TensorView& t, const char* name, int64_t s0, int64_t s1, int64_t s2) {
  TVM_FFI_ICHECK(t.size(0) == s0 && t.size(1) == s1 && t.size(2) == s2)
      << name << " has shape (" << t.size(0) << ", " << t.size(1) << ", " << t.size(2)
      << "), expected (" << s0 << ", " << s1 << ", " << s2 << ")";
}

void check_shape_4d(const TensorView& t, const char* name, int64_t s0, int64_t s1, int64_t s2,
                    int64_t s3) {
  TVM_FFI_ICHECK(t.size(0) == s0 && t.size(1) == s1 && t.size(2) == s2 && t.size(3) == s3)
      << name << " has shape (" << t.size(0) << ", " << t.size(1) << ", " << t.size(2) << ", "
      << t.size(3) << "), expected (" << s0 << ", " << s1 << ", " << s2 << ", " << s3 << ")";
}

void check_v2g_shard(const TensorView& tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK_EQ(tensor.stride(-1), 1) << name << " must be contiguous at the last dimension";
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 4) << name << " must be a 4D tensor";
  TVM_FFI_ICHECK(tensor.dtype() == dl_float16 || tensor.dtype() == dl_bfloat16)
      << name << " must have dtype float16 or bfloat16";
  // Head count is parametric (e.g. 28 under TP2); D=128 stays a hard
  // requirement (kernel tiles, V permutation and scale shapes depend on it).
  TVM_FFI_ICHECK(tensor.size(0) > 0 && tensor.size(1) > 0 && tensor.size(2) > 0 &&
                 tensor.size(3) == 128)
      << name << " must be a non-empty [B,L,H,128] tensor";
}

void check_v2g_rank(int64_t rank, int64_t world_size) {
  // {2,4,6,8} aligns with the Ulysses SUPPORTED_WORLD_SIZES; the payload
  // layout is fully world_size-parameterized.  P=6 is admissible but untested.
  TVM_FFI_ICHECK(world_size == 2 || world_size == 4 || world_size == 6 || world_size == 8)
      << "V2-G requires world_size in {2,4,6,8}";
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size) << "V2-G requires 0 <= rank < world_size";
}

}  // namespace

// Quantize canonical NHD V with an externally supplied per-channel scale. The
// output intentionally remains canonical uint8 FP8 bits for low-precision
// Ulysses communication; Sage's sequence permutation is a separate operation.
void ulysses_lowp_quant_v_fp8_with_scale(TensorView input, TensorView scale, TensorView output) {
  CHECK_CUDA(input);
  CHECK_CUDA(scale);
  CHECK_CUDA(output);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(scale);
  CHECK_CONTIGUOUS(output);
  CHECK_DIM(4, input);
  CHECK_DIM(3, scale);
  CHECK_DIM(4, output);
  CHECK_INPUT_TYPE(scale, dl_float32);
  CHECK_INPUT_TYPE(output, dl_uint8);
  CHECK_DEVICE(input, scale);
  CHECK_DEVICE(input, output);
  TVM_FFI_ICHECK(input.size(0) > 0 && input.size(1) > 0 && input.size(2) > 0)
      << "input batch, sequence, and head dimensions must be non-zero";
  TVM_FFI_ICHECK_EQ(input.size(3), 128) << "input head dimension must be 128";
  check_shape_3d(scale, "scale", input.size(0), input.size(2), input.size(3));
  check_shape_4d(output, "output", input.size(0), input.size(1), input.size(2), input.size(3));

  constexpr uint32_t THREADS = 256;
  constexpr uint32_t PACK_SIZE = 8;
  const uint64_t num_packs = static_cast<uint64_t>(input.numel()) / PACK_SIZE;
  const uint32_t blocks = static_cast<uint32_t>((num_packs + THREADS - 1) / THREADS);
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  const cudaStream_t stream = get_stream(input.device());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
    lowp::QuantVFP8WithScaleKernel<c_type><<<blocks, THREADS, 0, stream>>>(
        static_cast<const c_type*>(input.data_ptr()), static_cast<float*>(scale.data_ptr()),
        static_cast<int8_t*>(output.data_ptr()), num_packs, static_cast<uint32_t>(input.size(1)),
        static_cast<uint32_t>(input.size(2)), static_cast<uint32_t>(input.size(3)));
    cudaError_t status = cudaGetLastError();
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "QuantVFP8WithScaleKernel failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

// Fused per-(batch, head, channel) K-sum and V-amax statistics, two-stage
// sequence-parallel form. Unlike the fork this launcher does NOT allocate the
// chunk-partial workspaces: the Python wrapper allocates k_partial/v_partial
// [B, H, ceil(L/256), 128] fp32 and passes them through.
void ulysses_lowp_k_sum_v_amax(TensorView k, TensorView v, TensorView k_sum, TensorView v_amax,
                               TensorView k_partial, TensorView v_partial) {
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  CHECK_CUDA(k_sum);
  CHECK_CUDA(v_amax);
  CHECK_CUDA(k_partial);
  CHECK_CUDA(v_partial);
  CHECK_LAST_DIM_CONTIGUOUS(k);
  CHECK_LAST_DIM_CONTIGUOUS(v);
  CHECK_CONTIGUOUS(k_sum);
  CHECK_CONTIGUOUS(v_amax);
  CHECK_CONTIGUOUS(k_partial);
  CHECK_CONTIGUOUS(v_partial);
  CHECK_DIM(4, k);
  CHECK_DIM(4, v);
  CHECK_DIM(3, k_sum);
  CHECK_DIM(3, v_amax);
  CHECK_DIM(4, k_partial);
  CHECK_DIM(4, v_partial);
  CHECK_INPUT_TYPE(k_sum, dl_float32);
  CHECK_INPUT_TYPE(v_amax, dl_float32);
  CHECK_INPUT_TYPE(k_partial, dl_float32);
  CHECK_INPUT_TYPE(v_partial, dl_float32);

  TVM_FFI_ICHECK(k.dtype() == dl_float16 || k.dtype() == dl_bfloat16)
      << "k must have dtype float16 or bfloat16";
  CHECK_SAME_DTYPE(k, v);
  CHECK_DEVICE(k, v);
  CHECK_DEVICE(k, k_sum);
  CHECK_DEVICE(k, v_amax);
  CHECK_DEVICE(k, k_partial);
  CHECK_DEVICE(k, v_partial);
  CHECK_SHAPE(k, v);
  TVM_FFI_ICHECK(k.size(0) > 0 && k.size(1) > 0) << "batch and sequence must be non-zero";
  // Head count is parametric (TP shards heads before attention, e.g. 56/TP2 = 28);
  // the reduction below is per-(batch, head, channel) and never mixes heads.
  TVM_FFI_ICHECK_GT(k.size(2), 0) << "head count must be positive";
  TVM_FFI_ICHECK_EQ(k.size(3), 128) << "head dimension must be 128";
  check_shape_3d(k_sum, "k_sum", k.size(0), k.size(2), k.size(3));
  check_shape_3d(v_amax, "v_amax", k.size(0), k.size(2), k.size(3));

  ffi::CUDADeviceGuard device_guard(k.device().device_id);
  const cudaStream_t stream = get_stream(k.device());
  constexpr uint32_t HEAD_DIM = 128;
  constexpr uint32_t TOKEN_LANES = 2;
  constexpr uint32_t CHUNK_TOKENS = 256;
  const int64_t batch = k.size(0);
  const int64_t tokens = k.size(1);
  const int64_t heads = k.size(2);
  const int64_t chunks = (tokens + CHUNK_TOKENS - 1) / CHUNK_TOKENS;
  check_shape_4d(k_partial, "k_partial", batch, heads, chunks, HEAD_DIM);
  check_shape_4d(v_partial, "v_partial", batch, heads, chunks, HEAD_DIM);
  dim3 grid(heads, batch, chunks);
  dim3 block(HEAD_DIM * TOKEN_LANES);
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(k.dtype(), c_type, [&] {
    lowp::KSumVAmaxPartialKernel<c_type, HEAD_DIM, CHUNK_TOKENS><<<grid, block, 0, stream>>>(
        static_cast<const c_type*>(k.data_ptr()), static_cast<const c_type*>(v.data_ptr()),
        static_cast<float*>(k_partial.data_ptr()), static_cast<float*>(v_partial.data_ptr()),
        static_cast<uint32_t>(tokens), static_cast<uint32_t>(heads),
        static_cast<uint32_t>(chunks), k.stride(0), k.stride(1), k.stride(2), v.stride(0),
        v.stride(1), v.stride(2));
    cudaError_t status = cudaGetLastError();
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "KSumVAmaxPartialKernel failed with error code " << cudaGetErrorString(status);
    return true;
  });
  // Stage 2 combines the chunk partials in FIXED ascending chunk order, so
  // results are deterministic (bit-identical run to run). See the kernel
  // comment for the one deliberate ULP-level k_sum association change of the
  // two-stage form.
  dim3 grid2(heads, batch);
  dim3 block2(HEAD_DIM);
  lowp::KSumVAmaxCombineKernel<HEAD_DIM><<<grid2, block2, 0, stream>>>(
      static_cast<const float*>(k_partial.data_ptr()),
      static_cast<const float*>(v_partial.data_ptr()), static_cast<float*>(k_sum.data_ptr()),
      static_cast<float*>(v_amax.data_ptr()), static_cast<uint32_t>(heads),
      static_cast<uint32_t>(chunks));
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "KSumVAmaxCombineKernel failed with error code " << cudaGetErrorString(status);
}

// Per-touched-group partial Q amax on this rank's shard, on the GLOBAL
// 32-token grid.
void ulysses_lowp_q_grouped_amax(TensorView q, TensorView amax_out, int64_t rank,
                                 int64_t world_size) {
  check_v2g_shard(q, "q");
  check_v2g_rank(rank, world_size);
  CHECK_CUDA(amax_out);
  CHECK_CONTIGUOUS(amax_out);
  CHECK_DIM(3, amax_out);
  CHECK_INPUT_TYPE(amax_out, dl_float32);
  CHECK_DEVICE(q, amax_out);

  const int64_t local_sequence = q.size(1);
  const int64_t slots_alloc = lowp::grid::slots(local_sequence, 32);
  const int64_t group_first = lowp::grid::group_first(rank, local_sequence, 32);
  const int64_t touched = lowp::grid::group_last(rank, local_sequence, 32) - group_first + 1;
  check_shape_3d(amax_out, "amax_out", q.size(0), q.size(2), slots_alloc);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q.dtype(), c_type, [&] {
    constexpr uint32_t HEAD_DIM = 128;
    constexpr uint32_t GROUP = 32;
    dim3 grid(touched, q.size(2), q.size(0));
    dim3 block(GROUP * (HEAD_DIM / 8));
    lowp::GroupedAmaxKernel<HEAD_DIM, GROUP, false, c_type><<<grid, block, 0, stream>>>(
        static_cast<const c_type*>(q.data_ptr()), nullptr,
        static_cast<float*>(amax_out.data_ptr()), static_cast<uint32_t>(local_sequence),
        static_cast<uint32_t>(rank * local_sequence), static_cast<uint32_t>(q.size(2)),
        static_cast<uint32_t>(slots_alloc), static_cast<uint32_t>(group_first), q.stride(0),
        q.stride(1), q.stride(2));
    cudaError_t status = cudaGetLastError();
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "GroupedAmaxKernel failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

// Per-touched-group partial K amax (mean-subtracted) on this rank's shard, on
// the GLOBAL 64-token grid.
void ulysses_lowp_k_grouped_amax(TensorView k, TensorView k_mean, TensorView amax_out, int64_t rank,
                                 int64_t world_size) {
  check_v2g_shard(k, "k");
  check_v2g_rank(rank, world_size);
  CHECK_CUDA(k_mean);
  CHECK_CONTIGUOUS(k_mean);
  CHECK_DIM(3, k_mean);
  CHECK_CUDA(amax_out);
  CHECK_CONTIGUOUS(amax_out);
  CHECK_DIM(3, amax_out);
  CHECK_INPUT_TYPE(amax_out, dl_float32);
  TVM_FFI_ICHECK(k.dtype() == k_mean.dtype()) << "k_mean must have the same dtype as k";
  CHECK_DEVICE(k, k_mean);
  CHECK_DEVICE(k, amax_out);
  check_shape_3d(k_mean, "k_mean", k.size(0), k.size(2), k.size(3));

  const int64_t local_sequence = k.size(1);
  const int64_t slots_alloc = lowp::grid::slots(local_sequence, 64);
  const int64_t group_first = lowp::grid::group_first(rank, local_sequence, 64);
  const int64_t touched = lowp::grid::group_last(rank, local_sequence, 64) - group_first + 1;
  check_shape_3d(amax_out, "amax_out", k.size(0), k.size(2), slots_alloc);

  ffi::CUDADeviceGuard device_guard(k.device().device_id);
  const cudaStream_t stream = get_stream(k.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(k.dtype(), c_type, [&] {
    constexpr uint32_t HEAD_DIM = 128;
    constexpr uint32_t GROUP = 64;
    dim3 grid(touched, k.size(2), k.size(0));
    dim3 block(GROUP * (HEAD_DIM / 8));
    lowp::GroupedAmaxKernel<HEAD_DIM, GROUP, true, c_type><<<grid, block, 0, stream>>>(
        static_cast<const c_type*>(k.data_ptr()), static_cast<const c_type*>(k_mean.data_ptr()),
        static_cast<float*>(amax_out.data_ptr()), static_cast<uint32_t>(local_sequence),
        static_cast<uint32_t>(rank * local_sequence), static_cast<uint32_t>(k.size(2)),
        static_cast<uint32_t>(slots_alloc), static_cast<uint32_t>(group_first), k.stride(0),
        k.stride(1), k.stride(2));
    cudaError_t status = cudaGetLastError();
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "GroupedAmaxKernel failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

// Callers must zero the scale-and-padding region [3*main_bytes, chunk_bytes)
// of every destination chunk before the first V2-G pack launch; the kernels
// write only the touched slots.
void ulysses_lowp_quant_q_int8_pack(TensorView q, TensorView q_amax_final, TensorView output,
                                    int64_t rank, int64_t world_size) {
  check_v2g_shard(q, "q");
  check_v2g_rank(rank, world_size);
  CHECK_CUDA(q_amax_final);
  CHECK_CONTIGUOUS(q_amax_final);
  CHECK_DIM(3, q_amax_final);
  CHECK_INPUT_TYPE(q_amax_final, dl_float32);
  CHECK_CUDA(output);
  CHECK_CONTIGUOUS(output);
  CHECK_DIM(2, output);
  CHECK_INPUT_TYPE(output, dl_uint8);
  CHECK_DEVICE(q, q_amax_final);
  CHECK_DEVICE(q, output);

  const int64_t batch_size = q.size(0);
  const int64_t local_sequence = q.size(1);
  const int64_t num_heads = q.size(2);
  const int64_t head_dim = q.size(3);
  const int64_t local_heads = num_heads / world_size;
  const lowp::grid::ChunkSpec spec =
      lowp::grid::chunk_spec(batch_size, local_sequence, local_heads, head_dim);
  const int64_t group_first = lowp::grid::group_first(rank, local_sequence, 32);
  const int64_t touched = lowp::grid::group_last(rank, local_sequence, 32) - group_first + 1;
  check_shape_3d(q_amax_final, "q_amax_final", batch_size, num_heads, spec.q_slots);
  check_shape_2d(output, "output", world_size, spec.chunk_bytes);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q.dtype(), c_type, [&] {
    constexpr uint32_t HEAD_DIM = 128;
    constexpr uint32_t GROUP = 32;
    dim3 grid(touched, num_heads, batch_size);
    dim3 block(GROUP * (HEAD_DIM / 8));
    lowp::QuantInt8GroupScalePackKernel<HEAD_DIM, GROUP, false, c_type>
        <<<grid, block, 0, stream>>>(
            static_cast<const c_type*>(q.data_ptr()), nullptr,
            static_cast<float*>(q_amax_final.data_ptr()),
            static_cast<uint8_t*>(output.data_ptr()), static_cast<uint32_t>(local_sequence),
            static_cast<uint32_t>(rank * local_sequence), static_cast<uint32_t>(num_heads),
            static_cast<uint32_t>(local_heads), static_cast<uint32_t>(batch_size),
            static_cast<uint64_t>(spec.chunk_bytes), 0,
            static_cast<uint64_t>(3 * spec.main_bytes), static_cast<uint32_t>(spec.q_slots),
            static_cast<uint32_t>(group_first), q.stride(0), q.stride(1), q.stride(2));
    cudaError_t status = cudaGetLastError();
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "QuantInt8GroupScalePackKernel failed with error code "
        << cudaGetErrorString(status);
    return true;
  });
}

// Same zero-fill contract as ulysses_lowp_quant_q_int8_pack: the caller zeroes
// [3*main_bytes, chunk_bytes) of every destination chunk before the first pack
// launch of a payload.
void ulysses_lowp_quant_kv_int8_fp8_pack(TensorView k, TensorView v, TensorView k_mean,
                                         TensorView k_amax_final, TensorView v_scale,
                                         TensorView output, int64_t rank, int64_t world_size) {
  check_v2g_shard(k, "k");
  check_v2g_shard(v, "v");
  check_v2g_rank(rank, world_size);
  CHECK_CUDA(k_mean);
  CHECK_CONTIGUOUS(k_mean);
  CHECK_DIM(3, k_mean);
  CHECK_CUDA(k_amax_final);
  CHECK_CONTIGUOUS(k_amax_final);
  CHECK_DIM(3, k_amax_final);
  CHECK_INPUT_TYPE(k_amax_final, dl_float32);
  CHECK_CUDA(v_scale);
  CHECK_CONTIGUOUS(v_scale);
  CHECK_DIM(3, v_scale);
  CHECK_INPUT_TYPE(v_scale, dl_float32);
  CHECK_CUDA(output);
  CHECK_CONTIGUOUS(output);
  CHECK_DIM(2, output);
  CHECK_INPUT_TYPE(output, dl_uint8);
  CHECK_SHAPE(k, v);
  TVM_FFI_ICHECK(k.dtype() == v.dtype() && k.dtype() == k_mean.dtype())
      << "k, v, and k_mean must have the same dtype";
  CHECK_DEVICE(k, v);
  CHECK_DEVICE(k, k_mean);
  CHECK_DEVICE(k, k_amax_final);
  CHECK_DEVICE(k, v_scale);
  CHECK_DEVICE(k, output);
  check_shape_3d(k_mean, "k_mean", k.size(0), k.size(2), k.size(3));
  check_shape_3d(v_scale, "v_scale", k.size(0), k.size(2), k.size(3));

  const int64_t batch_size = k.size(0);
  const int64_t local_sequence = k.size(1);
  const int64_t num_heads = k.size(2);
  const int64_t head_dim = k.size(3);
  const int64_t local_heads = num_heads / world_size;
  const lowp::grid::ChunkSpec spec =
      lowp::grid::chunk_spec(batch_size, local_sequence, local_heads, head_dim);
  const int64_t group_first = lowp::grid::group_first(rank, local_sequence, 64);
  const int64_t touched = lowp::grid::group_last(rank, local_sequence, 64) - group_first + 1;
  check_shape_3d(k_amax_final, "k_amax_final", batch_size, num_heads, spec.k_slots);
  check_shape_2d(output, "output", world_size, spec.chunk_bytes);

  ffi::CUDADeviceGuard device_guard(k.device().device_id);
  const cudaStream_t stream = get_stream(k.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(k.dtype(), c_type, [&] {
    constexpr uint32_t HEAD_DIM = 128;
    constexpr uint32_t GROUP = 64;
    dim3 k_grid(touched, num_heads, batch_size);
    dim3 k_block(GROUP * (HEAD_DIM / 8));
    lowp::QuantInt8GroupScalePackKernel<HEAD_DIM, GROUP, true, c_type>
        <<<k_grid, k_block, 0, stream>>>(
            static_cast<const c_type*>(k.data_ptr()), static_cast<const c_type*>(k_mean.data_ptr()),
            static_cast<float*>(k_amax_final.data_ptr()),
            static_cast<uint8_t*>(output.data_ptr()), static_cast<uint32_t>(local_sequence),
            static_cast<uint32_t>(rank * local_sequence), static_cast<uint32_t>(num_heads),
            static_cast<uint32_t>(local_heads), static_cast<uint32_t>(batch_size),
            static_cast<uint64_t>(spec.chunk_bytes), static_cast<uint64_t>(spec.main_bytes),
            static_cast<uint64_t>(3 * spec.main_bytes + spec.q_scale_bytes),
            static_cast<uint32_t>(spec.k_slots), static_cast<uint32_t>(group_first), k.stride(0),
            k.stride(1), k.stride(2));
    constexpr uint32_t V_THREADS = 256;
    const uint64_t v_packs =
        static_cast<uint64_t>(batch_size) * local_sequence * local_heads * head_dim / 8;
    dim3 v_grid((v_packs + V_THREADS - 1) / V_THREADS, world_size);
    dim3 v_block(V_THREADS);
    lowp::QuantVFP8WithScalePackKernel<c_type><<<v_grid, v_block, 0, stream>>>(
        static_cast<const c_type*>(v.data_ptr()), static_cast<float*>(v_scale.data_ptr()),
        static_cast<uint8_t*>(output.data_ptr()), v_packs, static_cast<uint32_t>(local_sequence),
        static_cast<uint32_t>(num_heads), static_cast<uint32_t>(local_heads),
        static_cast<uint32_t>(head_dim), static_cast<uint32_t>(batch_size),
        static_cast<uint64_t>(spec.main_bytes), static_cast<uint64_t>(spec.chunk_bytes),
        v.stride(0), v.stride(1), v.stride(2));
    cudaError_t status = cudaGetLastError();
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "quant_kv_int8_fp8_pack kernels failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

// Fused amax+quant fast path (ALIGN-128 only): one kernel reads Q once,
// reduces the per-group amax in-block, and quantizes from registers.  Byte-
// identical to the split GroupedAmax + QuantInt8GroupScalePack path.  Same
// zero-fill contract as the split packers.
void ulysses_lowp_quant_q_int8_pack_fused(TensorView q, TensorView output, int64_t rank,
                                          int64_t world_size) {
  check_v2g_shard(q, "q");
  check_v2g_rank(rank, world_size);
  CHECK_CUDA(output);
  CHECK_CONTIGUOUS(output);
  CHECK_DIM(2, output);
  CHECK_INPUT_TYPE(output, dl_uint8);
  CHECK_DEVICE(q, output);
  TVM_FFI_ICHECK_EQ(q.size(1) % 128, 0)
      << "fused amax+quant is an ALIGN-128 fast path: local_sequence must be a "
         "whole number of 128-token blocks (protocol 2 keeps the split kernels)";

  const int64_t batch_size = q.size(0);
  const int64_t local_sequence = q.size(1);
  const int64_t num_heads = q.size(2);
  const int64_t head_dim = q.size(3);
  const int64_t local_heads = num_heads / world_size;
  const lowp::grid::ChunkSpec spec =
      lowp::grid::chunk_spec(batch_size, local_sequence, local_heads, head_dim);
  const int64_t group_first = lowp::grid::group_first(rank, local_sequence, 32);
  const int64_t touched = lowp::grid::group_last(rank, local_sequence, 32) - group_first + 1;
  check_shape_2d(output, "output", world_size, spec.chunk_bytes);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q.dtype(), c_type, [&] {
    constexpr uint32_t HEAD_DIM = 128;
    constexpr uint32_t GROUP = 32;
    dim3 grid(touched, num_heads, batch_size);
    dim3 block(GROUP * (HEAD_DIM / 8));
    lowp::QuantInt8FusedAmaxPackKernel<HEAD_DIM, GROUP, false, c_type>
        <<<grid, block, 0, stream>>>(
            static_cast<const c_type*>(q.data_ptr()), nullptr,
            static_cast<uint8_t*>(output.data_ptr()), static_cast<uint32_t>(local_sequence),
            static_cast<uint32_t>(rank * local_sequence), static_cast<uint32_t>(num_heads),
            static_cast<uint32_t>(local_heads), static_cast<uint32_t>(batch_size),
            static_cast<uint64_t>(spec.chunk_bytes), 0,
            static_cast<uint64_t>(3 * spec.main_bytes), static_cast<uint32_t>(spec.q_slots),
            static_cast<uint32_t>(group_first), 0xFFFFFFFFu, 0u, q.stride(0), q.stride(1),
            q.stride(2));
    cudaError_t status = cudaGetLastError();
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "QuantInt8FusedAmaxPackKernel(Q) failed with error code "
        << cudaGetErrorString(status);
    return true;
  });
}

// Fused K (with in-kernel used_sequence tail repair) + packed V.
// used_sequence <= 0 means "no padding".
void ulysses_lowp_quant_kv_int8_fp8_pack_fused(TensorView k, TensorView v, TensorView k_mean,
                                               TensorView v_scale, TensorView output,
                                               int64_t rank, int64_t world_size,
                                               int64_t used_sequence) {
  check_v2g_shard(k, "k");
  check_v2g_shard(v, "v");
  check_v2g_rank(rank, world_size);
  CHECK_CUDA(k_mean);
  CHECK_CONTIGUOUS(k_mean);
  CHECK_DIM(3, k_mean);
  CHECK_CUDA(v_scale);
  CHECK_CONTIGUOUS(v_scale);
  CHECK_DIM(3, v_scale);
  CHECK_INPUT_TYPE(v_scale, dl_float32);
  CHECK_CUDA(output);
  CHECK_CONTIGUOUS(output);
  CHECK_DIM(2, output);
  CHECK_INPUT_TYPE(output, dl_uint8);
  CHECK_SHAPE(k, v);
  TVM_FFI_ICHECK(k.dtype() == v.dtype() && k.dtype() == k_mean.dtype())
      << "k, v, and k_mean must have the same dtype";
  CHECK_DEVICE(k, v);
  CHECK_DEVICE(k, k_mean);
  CHECK_DEVICE(k, v_scale);
  CHECK_DEVICE(k, output);
  check_shape_3d(k_mean, "k_mean", k.size(0), k.size(2), k.size(3));
  check_shape_3d(v_scale, "v_scale", k.size(0), k.size(2), k.size(3));
  TVM_FFI_ICHECK_EQ(k.size(1) % 128, 0)
      << "fused amax+quant is an ALIGN-128 fast path: local_sequence must be a "
         "whole number of 128-token blocks (protocol 2 keeps the split kernels)";

  const int64_t batch_size = k.size(0);
  const int64_t local_sequence = k.size(1);
  const int64_t num_heads = k.size(2);
  const int64_t head_dim = k.size(3);
  const int64_t local_heads = num_heads / world_size;
  const int64_t global_sequence = local_sequence * world_size;
  TVM_FFI_ICHECK(used_sequence <= global_sequence)
      << "used_sequence must not exceed local_sequence * world_size";
  const lowp::grid::ChunkSpec spec =
      lowp::grid::chunk_spec(batch_size, local_sequence, local_heads, head_dim);
  const int64_t group_first = lowp::grid::group_first(rank, local_sequence, 64);
  const int64_t touched = lowp::grid::group_last(rank, local_sequence, 64) - group_first + 1;
  check_shape_2d(output, "output", world_size, spec.chunk_bytes);

  // The ONE group mixing live and padded rows (matches the split path's
  // Python repair: only when the padding boundary is not group-aligned).
  uint32_t exclude_group = 0xFFFFFFFFu;
  uint32_t used_u32 = 0;
  if (used_sequence > 0 && used_sequence < global_sequence && (used_sequence % 64) != 0) {
    exclude_group = static_cast<uint32_t>((used_sequence - 1) / 64);
    used_u32 = static_cast<uint32_t>(used_sequence);
  }

  ffi::CUDADeviceGuard device_guard(k.device().device_id);
  const cudaStream_t stream = get_stream(k.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(k.dtype(), c_type, [&] {
    constexpr uint32_t HEAD_DIM = 128;
    constexpr uint32_t GROUP = 64;
    dim3 k_grid(touched, num_heads, batch_size);
    dim3 k_block(GROUP * (HEAD_DIM / 8));
    lowp::QuantInt8FusedAmaxPackKernel<HEAD_DIM, GROUP, true, c_type>
        <<<k_grid, k_block, 0, stream>>>(
            static_cast<const c_type*>(k.data_ptr()), static_cast<const c_type*>(k_mean.data_ptr()),
            static_cast<uint8_t*>(output.data_ptr()), static_cast<uint32_t>(local_sequence),
            static_cast<uint32_t>(rank * local_sequence), static_cast<uint32_t>(num_heads),
            static_cast<uint32_t>(local_heads), static_cast<uint32_t>(batch_size),
            static_cast<uint64_t>(spec.chunk_bytes), static_cast<uint64_t>(spec.main_bytes),
            static_cast<uint64_t>(3 * spec.main_bytes + spec.q_scale_bytes),
            static_cast<uint32_t>(spec.k_slots), static_cast<uint32_t>(group_first),
            exclude_group, used_u32, k.stride(0), k.stride(1), k.stride(2));
    constexpr uint32_t V_THREADS = 256;
    const uint64_t v_packs =
        static_cast<uint64_t>(batch_size) * local_sequence * local_heads * head_dim / 8;
    dim3 v_grid((v_packs + V_THREADS - 1) / V_THREADS, world_size);
    dim3 v_block(V_THREADS);
    lowp::QuantVFP8WithScalePackKernel<c_type><<<v_grid, v_block, 0, stream>>>(
        static_cast<const c_type*>(v.data_ptr()), static_cast<float*>(v_scale.data_ptr()),
        static_cast<uint8_t*>(output.data_ptr()), v_packs, static_cast<uint32_t>(local_sequence),
        static_cast<uint32_t>(num_heads), static_cast<uint32_t>(local_heads),
        static_cast<uint32_t>(head_dim), static_cast<uint32_t>(batch_size),
        static_cast<uint64_t>(spec.main_bytes), static_cast<uint64_t>(spec.chunk_bytes),
        v.stride(0), v.stride(1), v.stride(2));
    cudaError_t status = cudaGetLastError();
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "fused quant_kv pack kernels failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

// V2-G receiver: rebuild contiguous logical Q/K [B,S,h,128], the globally
// packed V [B,128,h,ceil(S/64)*64], and the global-grid Q/K scale tensors
// where slot g is written only by its canonical owner source (owner-only
// writes; unused tail slots are deterministically zeroed).
void ulysses_lowp_unpack_for_sage(TensorView input, TensorView q, TensorView k, TensorView v,
                                  TensorView q_scale, TensorView k_scale, int64_t local_sequence,
                                  int64_t world_size) {
  CHECK_CUDA(input);
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  CHECK_CUDA(q_scale);
  CHECK_CUDA(k_scale);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(q);
  CHECK_CONTIGUOUS(k);
  CHECK_CONTIGUOUS(v);
  CHECK_CONTIGUOUS(q_scale);
  CHECK_CONTIGUOUS(k_scale);
  CHECK_DIM(2, input);
  CHECK_DIM(4, q);
  CHECK_DIM(4, k);
  CHECK_DIM(4, v);
  CHECK_DIM(3, q_scale);
  CHECK_DIM(3, k_scale);
  CHECK_INPUT_TYPE(input, dl_uint8);
  CHECK_INPUT_TYPE(q, dl_int8);
  CHECK_INPUT_TYPE(k, dl_int8);
  CHECK_INPUT_TYPE(v, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(q_scale, dl_float32);
  CHECK_INPUT_TYPE(k_scale, dl_float32);
  TVM_FFI_ICHECK_GT(local_sequence, 0) << "local_sequence must be positive";
  TVM_FFI_ICHECK_EQ(local_sequence % 128, 0)
      << "ALIGN-128 (stats protocol 3): local_sequence must be a whole number of 128-token blocks";
  TVM_FFI_ICHECK(world_size == 2 || world_size == 4 || world_size == 6 || world_size == 8)
      << "V2-G requires world_size in {2,4,6,8}";
  CHECK_DEVICE(input, q);
  CHECK_DEVICE(input, k);
  CHECK_DEVICE(input, v);
  CHECK_DEVICE(input, q_scale);
  CHECK_DEVICE(input, k_scale);
  TVM_FFI_ICHECK_EQ(input.size(0), world_size) << "input source dimension must equal world_size";
  TVM_FFI_ICHECK(q.size(0) > 0 && q.size(2) > 0 && q.size(3) == 128)
      << "q must have non-empty [B,S,h,128] shape";

  const int64_t batch_size = q.size(0);
  const int64_t logical_sequence = local_sequence * world_size;
  const int64_t local_heads = q.size(2);
  const int64_t head_dim = q.size(3);
  const int64_t padded_sequence = (logical_sequence + 63) / 64 * 64;
  const int64_t q_scale_alloc = (logical_sequence + 127) / 128 * 4;
  const int64_t k_scale_alloc = (logical_sequence + 63) / 64;
  const lowp::grid::ChunkSpec spec =
      lowp::grid::chunk_spec(batch_size, local_sequence, local_heads, head_dim);
  TVM_FFI_ICHECK_EQ(q.size(1), logical_sequence) << "q logical sequence shape is incorrect";
  check_shape_4d(k, "k", batch_size, logical_sequence, local_heads, head_dim);
  check_shape_4d(v, "v", batch_size, head_dim, local_heads, padded_sequence);
  check_shape_3d(q_scale, "q_scale", batch_size, local_heads, q_scale_alloc);
  check_shape_3d(k_scale, "k_scale", batch_size, local_heads, k_scale_alloc);
  TVM_FFI_ICHECK_EQ(input.size(1), spec.chunk_bytes)
      << "input chunk size does not match the V2-G ABI";

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  const cudaStream_t stream = get_stream(input.device());
  constexpr uint32_t HEAD_DIM = 128;
  constexpr uint32_t CTA_SIZE = 64;
  constexpr uint32_t VECTOR_SIZE = 16;
  dim3 grid(padded_sequence / CTA_SIZE, local_heads, batch_size);
  dim3 block(CTA_SIZE * HEAD_DIM / VECTOR_SIZE);
  lowp::UnpackForSageKernel<HEAD_DIM, CTA_SIZE><<<grid, block, 0, stream>>>(
      static_cast<const uint8_t*>(input.data_ptr()), static_cast<uint8_t*>(q.data_ptr()),
      static_cast<uint8_t*>(k.data_ptr()), static_cast<uint8_t*>(v.data_ptr()),
      static_cast<uint8_t*>(q_scale.data_ptr()), static_cast<uint8_t*>(k_scale.data_ptr()),
      static_cast<uint64_t>(spec.main_bytes), static_cast<uint64_t>(spec.chunk_bytes),
      static_cast<uint32_t>(batch_size), static_cast<uint32_t>(local_sequence),
      static_cast<uint32_t>(logical_sequence), static_cast<uint32_t>(padded_sequence),
      static_cast<uint32_t>(spec.q_slots), static_cast<uint32_t>(spec.k_slots),
      static_cast<uint32_t>(q_scale_alloc), static_cast<uint32_t>(k_scale_alloc));
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "UnpackForSageKernel failed with error code " << cudaGetErrorString(status);
}

// UNALIGNED receiver (boundary-stats protocol 2, 64-aligned GLOBAL packing):
// identical tensor contract to ulysses_lowp_unpack_for_sage but without the
// ALIGN-128 local_sequence precondition; a 64-token tile may span two source
// chunks and unused scale tail slots are deterministically zeroed by the
// kernel itself.
void ulysses_lowp_unpack_for_sage_unaligned(TensorView input, TensorView q, TensorView k,
                                            TensorView v, TensorView q_scale,
                                            TensorView k_scale, int64_t local_sequence,
                                            int64_t world_size) {
  CHECK_CUDA(input);
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  CHECK_CUDA(q_scale);
  CHECK_CUDA(k_scale);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(q);
  CHECK_CONTIGUOUS(k);
  CHECK_CONTIGUOUS(v);
  CHECK_CONTIGUOUS(q_scale);
  CHECK_CONTIGUOUS(k_scale);
  CHECK_DIM(2, input);
  CHECK_DIM(4, q);
  CHECK_DIM(4, k);
  CHECK_DIM(4, v);
  CHECK_DIM(3, q_scale);
  CHECK_DIM(3, k_scale);
  CHECK_INPUT_TYPE(input, dl_uint8);
  CHECK_INPUT_TYPE(q, dl_int8);
  CHECK_INPUT_TYPE(k, dl_int8);
  CHECK_INPUT_TYPE(v, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(q_scale, dl_float32);
  CHECK_INPUT_TYPE(k_scale, dl_float32);
  TVM_FFI_ICHECK_GT(local_sequence, 0) << "local_sequence must be positive";
  TVM_FFI_ICHECK(world_size == 2 || world_size == 4 || world_size == 6 || world_size == 8)
      << "V2-G requires world_size in {2,4,6,8}";
  CHECK_DEVICE(input, q);
  CHECK_DEVICE(input, k);
  CHECK_DEVICE(input, v);
  CHECK_DEVICE(input, q_scale);
  CHECK_DEVICE(input, k_scale);
  TVM_FFI_ICHECK_EQ(input.size(0), world_size) << "input source dimension must equal world_size";
  TVM_FFI_ICHECK(q.size(0) > 0 && q.size(2) > 0 && q.size(3) == 128)
      << "q must have non-empty [B,S,h,128] shape";

  const int64_t batch_size = q.size(0);
  const int64_t logical_sequence = local_sequence * world_size;
  const int64_t local_heads = q.size(2);
  const int64_t head_dim = q.size(3);
  const int64_t padded_sequence = (logical_sequence + 63) / 64 * 64;
  const int64_t q_scale_alloc = (logical_sequence + 127) / 128 * 4;
  const int64_t k_scale_alloc = (logical_sequence + 63) / 64;
  const lowp::grid::ChunkSpec spec =
      lowp::grid::chunk_spec(batch_size, local_sequence, local_heads, head_dim);
  TVM_FFI_ICHECK_EQ(q.size(1), logical_sequence) << "q logical sequence shape is incorrect";
  check_shape_4d(k, "k", batch_size, logical_sequence, local_heads, head_dim);
  check_shape_4d(v, "v", batch_size, head_dim, local_heads, padded_sequence);
  check_shape_3d(q_scale, "q_scale", batch_size, local_heads, q_scale_alloc);
  check_shape_3d(k_scale, "k_scale", batch_size, local_heads, k_scale_alloc);
  TVM_FFI_ICHECK_EQ(input.size(1), spec.chunk_bytes)
      << "input chunk size does not match the V2-G ABI";

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  const cudaStream_t stream = get_stream(input.device());
  constexpr uint32_t HEAD_DIM = 128;
  constexpr uint32_t CTA_SIZE = 64;
  constexpr uint32_t VECTOR_SIZE = 16;
  dim3 grid(padded_sequence / CTA_SIZE, local_heads, batch_size);
  dim3 block(CTA_SIZE * HEAD_DIM / VECTOR_SIZE);
  lowp::UnpackForSageUnalignedKernel<HEAD_DIM, CTA_SIZE><<<grid, block, 0, stream>>>(
      static_cast<const uint8_t*>(input.data_ptr()), static_cast<uint8_t*>(q.data_ptr()),
      static_cast<uint8_t*>(k.data_ptr()), static_cast<uint8_t*>(v.data_ptr()),
      static_cast<uint8_t*>(q_scale.data_ptr()), static_cast<uint8_t*>(k_scale.data_ptr()),
      static_cast<uint64_t>(spec.main_bytes), static_cast<uint64_t>(spec.chunk_bytes),
      static_cast<uint32_t>(batch_size), static_cast<uint32_t>(local_sequence),
      static_cast<uint32_t>(logical_sequence), static_cast<uint32_t>(padded_sequence),
      static_cast<uint32_t>(spec.q_slots), static_cast<uint32_t>(spec.k_slots),
      static_cast<uint32_t>(q_scale_alloc), static_cast<uint32_t>(k_scale_alloc));
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "UnpackForSageUnalignedKernel failed with error code " << cudaGetErrorString(status);
}

// Payload ABI version consumed by the Python capability handshake
// (ABI v3 = chunk layout shared by both stats protocols).
int64_t ulysses_lowp_abi_version() { return 3; }

TVM_FFI_DLL_EXPORT_TYPED_FUNC(ulysses_lowp_k_sum_v_amax, ulysses_lowp_k_sum_v_amax);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ulysses_lowp_q_grouped_amax, ulysses_lowp_q_grouped_amax);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ulysses_lowp_k_grouped_amax, ulysses_lowp_k_grouped_amax);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ulysses_lowp_quant_q_int8_pack, ulysses_lowp_quant_q_int8_pack);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ulysses_lowp_quant_kv_int8_fp8_pack,
                              ulysses_lowp_quant_kv_int8_fp8_pack);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ulysses_lowp_quant_q_int8_pack_fused,
                              ulysses_lowp_quant_q_int8_pack_fused);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ulysses_lowp_quant_kv_int8_fp8_pack_fused,
                              ulysses_lowp_quant_kv_int8_fp8_pack_fused);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ulysses_lowp_unpack_for_sage, ulysses_lowp_unpack_for_sage);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ulysses_lowp_unpack_for_sage_unaligned,
                              ulysses_lowp_unpack_for_sage_unaligned);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ulysses_lowp_quant_v_fp8_with_scale,
                              ulysses_lowp_quant_v_fp8_with_scale);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ulysses_lowp_abi_version, ulysses_lowp_abi_version);
