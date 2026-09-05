/* Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0. */
#pragma once

#include <Python.h>

#if defined(FLASHKDA_GENERATED_DIRECT_SOURCE_ABI)
#include <ATen/ATen.h>
#include <torch/csrc/autograd/python_variable.h>
#endif

#include <exception>
#include <memory>

// The authoritative direct launcher uses the CUDA Runtime API.  Select the
// same cubin-launch backend so the exported copy-to-kernel submission path is
// identical while retaining the audited embedded kernel image.
#if defined(FLASHKDA_GENERATED_EMBEDDED_CUBIN)
#ifdef TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
#undef TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
#endif
#define TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API 0
#endif

#include "flashkda_generated_binding_common.cuh"

namespace flashinfer::flash_kda_generated {

#if FLASHKDA_GENERATED_ABI_VARIANT == FLASHKDA_GENERATED_VARIANT_SERVING
struct GeneratedCheckpointMapWords {
  uint64_t words[sizeof(CUtensorMap) / sizeof(uint64_t)];
};

static __global__ void PublishGeneratedCheckpointMap(
    uint64_t* destination, GeneratedCheckpointMapWords source) {
  if (threadIdx.x < sizeof(CUtensorMap) / sizeof(uint64_t)) {
    destination[threadIdx.x] = source.words[threadIdx.x];
  }
}

inline CUtensorMap EncodeGeneratedCheckpointValueTma(const TensorView& tensor) {
  TVM_FFI_ICHECK(tensor.ndim() >= 2 &&
                 tensor.stride(tensor.ndim() - 1) == 1)
      << "checkpoint v must have unit innermost stride";
  const int64_t d1 = tensor.size(tensor.ndim() - 1);
  const int64_t d2 = tensor.size(tensor.ndim() - 2);
  TVM_FFI_ICHECK(d1 > 0 && d2 > 0 && d1 % 64 == 0)
      << "v trailing dimensions cannot encode checkpoint N16 TMA";
  const int64_t outer2 = tensor.numel() / (d1 * d2);
  uint64_t global_dim[4] = {
      64, static_cast<uint64_t>(d2), static_cast<uint64_t>(outer2),
      static_cast<uint64_t>(d1 / 64)};
  uint64_t global_strides[3] = {
      static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
      static_cast<uint64_t>(d1 * d2 * sizeof(__nv_bfloat16)),
      64 * sizeof(__nv_bfloat16)};
  uint32_t box_dim[4] = {64, 1, 16, 1};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(), global_dim,
      global_strides, box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for checkpoint N16 v with CUresult="
      << int(result);
  return map;
}

inline CUtensorMap EncodeGeneratedCheckpointTma(const TensorView& tensor) {
  TVM_FFI_ICHECK(tensor.ndim() == 4 && tensor.size(2) == 128 &&
                 tensor.size(3) == 128)
      << "state_checkpoints must be [C,H,128,128]";
  const int64_t checkpoints = tensor.size(0);
  const int64_t heads = tensor.size(1);
  uint64_t global_dim[4] = {128, 128, static_cast<uint64_t>(heads),
                            static_cast<uint64_t>(checkpoints)};
  uint64_t global_strides[3] = {
      128 * sizeof(__nv_bfloat16), 128 * 128 * sizeof(__nv_bfloat16),
      static_cast<uint64_t>(heads) * 128 * 128 * sizeof(__nv_bfloat16)};
  uint32_t box_dim[4] = {64, 128, 1, 1};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(), global_dim,
      global_strides, box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for state checkpoints with CUresult="
      << int(result);
  return map;
}

struct DirectM128Args {
  void *q{}, *q_tma{}, *k{}, *k_tma{}, *v{}, *v_tma{}, *g{}, *g_tma{};
  void *beta{}, *beta_tma{}, *a_log{}, *dt_bias{}, *cu_seqlens{}, *seq_order{};
  void *initial_state{}, *out{}, *out_tma{}, *final_state{};
  int32_t num_heads{}, use_initial_state{}, store_final_state{};
  float scale{}, lower_bound{};
  uint64_t state_indices_addr{}, state_checkpoints_addr{}, checkpoint_cu_starts_addr{};
  int64_t beta_token_stride{}, state_slot_stride{};
  int32_t use_state_indices{}, checkpoint_every_n_tokens{};
  void *cu_chunk_offsets{}, *chunk_state{}, *state_checkpoint_needed{};
  void *tape_qd{}, *tape_kd{}, *tape_kr{}, *tape_j{}, *tape_restore_factor{};
  void *tape_e{}, *tape_x{}, *tape_r{}, *norm_inv_out{}, *decay_out{};
  void *beta_active_out{}, *initial_state_f32{}, *zero_workspace{};
  int32_t zero_words{}, num_sequences{};
  void *state_checkpoints_tma{}, *final_state_f32{};
};

struct PreparedDirectM128Launch {
  DirectM128Args args{};
  dim3 grid{};
  cudaStream_t stream{};
  int32_t device_id{};
  void* kernel_args[50]{};

  PreparedDirectM128Launch(DirectM128Args input,
                           const StatePointerSlots& state, dim3 launch_grid,
                           cudaStream_t launch_stream, int32_t launch_device)
      : args(input),
        grid(launch_grid),
        stream(launch_stream),
        device_id(launch_device) {
    args.initial_state = state.initial_state;
    args.final_state = state.final_state;
    args.initial_state_f32 = state.initial_state_f32;
    args.final_state_f32 = state.final_state_f32;
    void* bound_args[] = {
        &args.q, &args.q_tma, &args.k, &args.k_tma, &args.v, &args.v_tma,
        &args.g, &args.g_tma, &args.beta, &args.beta_tma, &args.a_log,
        &args.dt_bias, &args.cu_seqlens, &args.seq_order, &args.initial_state,
        &args.out, &args.out_tma, &args.final_state, &args.num_heads,
        &args.use_initial_state, &args.store_final_state, &args.scale,
        &args.lower_bound, &args.state_indices_addr,
        &args.state_checkpoints_addr, &args.checkpoint_cu_starts_addr,
        &args.beta_token_stride, &args.state_slot_stride,
        &args.use_state_indices, &args.checkpoint_every_n_tokens,
        &args.cu_chunk_offsets, &args.chunk_state,
        &args.state_checkpoint_needed, &args.tape_qd, &args.tape_kd,
        &args.tape_kr, &args.tape_j, &args.tape_restore_factor, &args.tape_e,
        &args.tape_x, &args.tape_r, &args.norm_inv_out, &args.decay_out,
        &args.beta_active_out, &args.initial_state_f32, &args.zero_workspace,
        &args.zero_words, &args.num_sequences, &args.state_checkpoints_tma,
        &args.final_state_f32};
    CheckArgumentCount<50>(bound_args);
    for (int index = 0; index < 50; ++index) {
      kernel_args[index] = bound_args[index];
    }
    // The optional beta pack is submitted after this object is constructed.
    // Finish device and dynamic-smem setup here so the later launch-only FFI
    // call can enqueue the dependent kernel without CUDA setup in between.
    ConfigureGeneratedKernelForDevice(FLASHKDA_GENERATED_KERNEL_ARGUMENT,
                                      device_id);
  }

  PreparedDirectM128Launch(const PreparedDirectM128Launch&) = delete;
  PreparedDirectM128Launch& operator=(const PreparedDirectM128Launch&) = delete;
  PreparedDirectM128Launch(PreparedDirectM128Launch&&) = delete;
  PreparedDirectM128Launch& operator=(PreparedDirectM128Launch&&) = delete;
};

// Keep the native callback chain DSO-local.  Each generated selector DSO
// embeds a different kernel, so coalescing these inline functions or the
// PyMethodDef below could route a later selector through the first loaded DSO.
static inline void LaunchPreparedDirectM128(
    PreparedDirectM128Launch* prepared) {
  LaunchConfiguredGeneratedKernel(
      FLASHKDA_GENERATED_KERNEL_ARGUMENT, prepared->grid, prepared->stream,
      prepared->kernel_args, "generated direct-M128 launch");
}

inline int64_t DirectM128PreparedHandle(PreparedDirectM128Launch* prepared) {
  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(prepared));
}

inline PreparedDirectM128Launch* DirectM128PreparedPointer(int64_t handle) {
  TVM_FFI_ICHECK(handle != 0) << "direct-M128 prepared handle must be nonzero";
  return reinterpret_cast<PreparedDirectM128Launch*>(
      static_cast<uintptr_t>(handle));
}

inline void LaunchPreparedDirectM128Handle(int64_t handle) {
  LaunchPreparedDirectM128(DirectM128PreparedPointer(handle));
}

inline constexpr char kDirectM128PythonCapsuleName[] =
    "flashinfer.flash_kda_generated.PreparedDirectM128Launch";

static inline PyObject* LaunchPreparedDirectM128Python(
    PyObject* capsule, PyObject* arguments) {
  void* pointer =
      PyCapsule_GetPointer(capsule, kDirectM128PythonCapsuleName);
  if (pointer == nullptr) {
    return nullptr;
  }
  PyObject* destination = nullptr;
  PyObject* source = nullptr;
  if (!PyArg_ParseTuple(arguments, "OO:_launch_direct_m128_prepared",
                        &destination, &source)) {
    return nullptr;
  }
  if (!THPVariable_Check(destination) || !THPVariable_Check(source)) {
    PyErr_SetString(PyExc_TypeError,
                    "direct-M128 beta pack operands must be tensors");
    return nullptr;
  }
  try {
    // Enter the same aten::copy_ dispatcher as Tensor.copy_ without another
    // Python call between the copy activity and the dependent kernel launch.
    // The argument tuple owns both tensors for the duration of this callback.
    at::Tensor destination_tensor = THPVariable_Unpack(destination);
    at::Tensor source_tensor = THPVariable_Unpack(source);
    destination_tensor.copy_(source_tensor, /*non_blocking=*/false);
    LaunchPreparedDirectM128(
        static_cast<PreparedDirectM128Launch*>(pointer));
  } catch (const c10::Error& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError,
                    "generated direct-M128 launch failed");
    return nullptr;
  }
  Py_RETURN_NONE;
}

static inline void DecrefDirectM128PythonObject(void* object) {
  const PyGILState_STATE gil_state = PyGILState_Ensure();
  Py_DECREF(static_cast<PyObject*>(object));
  PyGILState_Release(gil_state);
}

static inline ffi::ObjectRef MakeDirectM128PythonLauncher(int64_t handle) {
  auto* prepared = DirectM128PreparedPointer(handle);
  PyObject* capsule =
      PyCapsule_New(prepared, kDirectM128PythonCapsuleName, nullptr);
  if (capsule == nullptr) {
    PyErr_Clear();
  }
  TVM_FFI_ICHECK(capsule != nullptr)
      << "failed to create direct-M128 launch capsule";

  static PyMethodDef method = {
      "_launch_direct_m128_prepared",
      LaunchPreparedDirectM128Python,
      METH_VARARGS,
      nullptr,
  };
  PyObject* callable = PyCFunction_NewEx(&method, capsule, nullptr);
  Py_DECREF(capsule);
  if (callable == nullptr) {
    PyErr_Clear();
  }
  TVM_FFI_ICHECK(callable != nullptr)
      << "failed to create direct-M128 native launcher";

  TVMFFIObjectHandle opaque_handle = nullptr;
  const int status = TVMFFIObjectCreateOpaque(
      callable, kTVMFFIOpaquePyObject, DecrefDirectM128PythonObject,
      &opaque_handle);
  if (status != 0 || opaque_handle == nullptr) {
    Py_DECREF(callable);
  }
  TVM_FFI_ICHECK(status == 0 && opaque_handle != nullptr)
      << "failed to wrap direct-M128 native launcher";
  return ffi::ObjectRef(
      ffi::details::ObjectUnsafe::ObjectPtrFromOwned<ffi::Object>(
          reinterpret_cast<TVMFFIObject*>(opaque_handle)));
}

inline void DisposePreparedDirectM128Handle(int64_t handle) {
  delete DirectM128PreparedPointer(handle);
}

inline int64_t PrepareDirectM128(
    TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
    TensorView beta_tma, TensorView a_log, TensorView dt_bias,
    TensorView cu_seqlens, TensorView seq_order, TensorView state_indices,
    TensorView initial_state, TensorView out, TensorView final_state,
    TensorView state_checkpoints, TensorView checkpoint_cu_starts,
    TensorView cu_chunk_offsets, TensorView chunk_state,
    TensorView state_checkpoint_needed, TensorView tape_qd, TensorView tape_kd,
    TensorView tape_kr, TensorView tape_j, TensorView tape_restore_factor,
    TensorView tape_e, TensorView tape_x, TensorView tape_r,
    TensorView norm_inv_out, TensorView decay_out, TensorView beta_active_out,
    TensorView zero_workspace, TensorView state_checkpoints_tma,
    TensorView descriptor_storage, int64_t prepare_descriptors,
    int64_t num_heads, int64_t beta_token_stride,
    int64_t state_slot_stride, int64_t use_state_indices,
    int64_t use_initial_state, int64_t store_final_state,
    int64_t checkpoint_every_n_tokens, int64_t zero_words,
    int64_t num_sequences, double scale, double lower_bound,
    int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t cuda_stream) {
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA);
  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const auto prepared = [&]() {
#if FLASHKDA_GENERATED_AFFINE_DEPENDENCY == \
    FLASHKDA_GENERATED_AFFINE_BF16_INDEXED_INITIAL_FP32_FINAL
    CheckCudaTensorDevice(initial_state, "initial_state", q.device().device_id);
    CheckCudaTensorDevice(final_state, "final_state_f32", q.device().device_id);
    CheckDtype(initial_state, "initial_state", dl_bfloat16);
    CheckDtype(final_state, "final_state_f32", dl_float32);
    StatePointerSlots raw_state{initial_state.data_ptr(), nullptr, nullptr,
                                final_state.data_ptr(), dl_bfloat16, 0};
    return PrepareCommonInputsWithRawState<
        FLASHKDA_GENERATED_VALUE_ROWS, FLASHKDA_GENERATED_TMA_TILE_TOKENS,
        FLASHKDA_GENERATED_PAIR_PACKED_BETA != 0, false, true>(
        q, k, v, g, beta, beta_tma, a_log, dt_bias, cu_seqlens, seq_order,
        initial_state, out, final_state, descriptor_storage,
        prepare_descriptors, num_heads, beta_token_stride, scale, lower_bound,
        cuda_stream, raw_state);
#elif FLASHKDA_GENERATED_AFFINE_DEPENDENCY == \
    FLASHKDA_GENERATED_AFFINE_FP32_SPLIT_STATE || \
    FLASHKDA_GENERATED_AFFINE_DEPENDENCY == \
    FLASHKDA_GENERATED_AFFINE_FP32_CARRY_DEPENDENCY
    CheckCudaTensorDevice(initial_state, "initial_state_f32", q.device().device_id);
    CheckCudaTensorDevice(final_state, "final_state_f32", q.device().device_id);
    CheckDtype(initial_state, "initial_state_f32", dl_float32);
    CheckDtype(final_state, "final_state_f32", dl_float32);
    StatePointerSlots raw_state{nullptr, nullptr, initial_state.data_ptr(),
                                final_state.data_ptr(), dl_float32, 0};
    return PrepareCommonInputsWithRawState<
        FLASHKDA_GENERATED_VALUE_ROWS, FLASHKDA_GENERATED_TMA_TILE_TOKENS,
        FLASHKDA_GENERATED_PAIR_PACKED_BETA != 0, false>(
        q, k, v, g, beta, beta_tma, a_log, dt_bias, cu_seqlens, seq_order,
        initial_state, out, final_state, descriptor_storage,
        prepare_descriptors, num_heads, beta_token_stride, scale, lower_bound,
        cuda_stream, raw_state);
#elif FLASHKDA_GENERATED_AFFINE_DEPENDENCY == \
    FLASHKDA_GENERATED_AFFINE_BF16_STATE_WITH_FP32_SPLIT_DEPENDENCY
    CheckCudaTensorDevice(initial_state, "initial_state", q.device().device_id);
    CheckCudaTensorDevice(final_state, "final_state", q.device().device_id);
    CheckCudaTensorDevice(tape_restore_factor, "initial_state_f32_dependency",
                          q.device().device_id);
    CheckCudaTensorDevice(norm_inv_out, "final_state_f32_dummy",
                          q.device().device_id);
    CheckDtype(initial_state, "initial_state", dl_bfloat16);
    CheckDtype(final_state, "final_state", dl_bfloat16);
    CheckDtype(tape_restore_factor, "initial_state_f32_dependency", dl_float32);
    CheckDtype(norm_inv_out, "final_state_f32_dummy", dl_float32);
    StatePointerSlots raw_state{initial_state.data_ptr(), final_state.data_ptr(),
                                tape_restore_factor.data_ptr(),
                                norm_inv_out.data_ptr(), dl_bfloat16, 0};
    return PrepareCommonInputsWithRawState<
        FLASHKDA_GENERATED_VALUE_ROWS, FLASHKDA_GENERATED_TMA_TILE_TOKENS,
        FLASHKDA_GENERATED_PAIR_PACKED_BETA != 0, false>(
        q, k, v, g, beta, beta_tma, a_log, dt_bias, cu_seqlens, seq_order,
        initial_state, out, final_state, descriptor_storage,
        prepare_descriptors, num_heads, beta_token_stride, scale, lower_bound,
        cuda_stream, raw_state);
#else
    return PrepareCommonInputs<
        FLASHKDA_GENERATED_VALUE_ROWS, FLASHKDA_GENERATED_TMA_TILE_TOKENS,
        FLASHKDA_GENERATED_PAIR_PACKED_BETA != 0, false>(
        q, k, v, g, beta, beta_tma, a_log, dt_bias, cu_seqlens, seq_order,
        state_indices, initial_state, out, final_state, descriptor_storage,
        prepare_descriptors, num_heads, beta_token_stride, state_slot_stride,
        use_state_indices, use_initial_state, store_final_state, scale,
        lower_bound, cuda_stream);
#endif
  }();
  TVM_FFI_ICHECK(num_sequences == prepared.num_sequences)
      << "num_sequences must match cu_seqlens";
  if (checkpoint_every_n_tokens != 0) {
    flash_kda::CheckServingCheckpointInputsForStateDtype(
        state_checkpoints, checkpoint_cu_starts, prepared.device_id,
        prepared.num_sequences, num_heads, checkpoint_every_n_tokens,
        GeneratedStateDtype(), FLASHKDA_GENERATED_TMA_TILE_TOKENS);
    flash_kda::CheckServingAuxiliaryNoOverlap(
        state_indices, state_checkpoints, checkpoint_cu_starts, q, k, v, g,
        beta, beta_tma, a_log, dt_bias, cu_seqlens, seq_order, initial_state,
        out, final_state, descriptor_storage, use_state_indices,
        checkpoint_every_n_tokens);
  }
  DirectM128Args args{};
  args.q=q.data_ptr(); args.q_tma=prepared.tma.q; args.k=k.data_ptr(); args.k_tma=prepared.tma.k;
  args.v=v.data_ptr(); args.v_tma=prepared.tma.v; args.g=g.data_ptr(); args.g_tma=prepared.tma.g;
  args.beta=beta.data_ptr(); args.beta_tma=prepared.tma.beta; args.a_log=a_log.data_ptr();
  args.dt_bias=dt_bias.data_ptr(); args.cu_seqlens=cu_seqlens.data_ptr();
  args.seq_order=seq_order.data_ptr(); args.out=out.data_ptr(); args.out_tma=prepared.tma.out;
  args.num_heads=CheckedInt32(num_heads,"num_heads");
  args.use_initial_state=CheckedInt32(use_initial_state,"use_initial_state");
  args.store_final_state=CheckedInt32(store_final_state,"store_final_state");
  args.scale=static_cast<float>(scale); args.lower_bound=static_cast<float>(lower_bound);
  args.state_indices_addr=reinterpret_cast<uintptr_t>(state_indices.data_ptr());
  args.state_checkpoints_addr=reinterpret_cast<uintptr_t>(CheckedBufferPointer(
      state_checkpoints,"state_checkpoints",prepared.device_id,GeneratedStateDtype(),true));
  args.checkpoint_cu_starts_addr=reinterpret_cast<uintptr_t>(CheckedBufferPointer(
      checkpoint_cu_starts,"checkpoint_cu_starts",prepared.device_id,dl_int64,true));
  args.beta_token_stride=beta_token_stride; args.state_slot_stride=state_slot_stride;
  args.use_state_indices=CheckedInt32(use_state_indices,"use_state_indices");
  args.checkpoint_every_n_tokens=CheckedInt32(checkpoint_every_n_tokens,"checkpoint_every_n_tokens");
#define FLASHKDA_OPTIONAL(field, dtype) \
  args.field=CheckedBufferPointer(field,#field,prepared.device_id,dtype,true)
  FLASHKDA_OPTIONAL(cu_chunk_offsets,dl_int64); FLASHKDA_OPTIONAL(chunk_state,dl_bfloat16);
  FLASHKDA_OPTIONAL(state_checkpoint_needed,dl_uint32); FLASHKDA_OPTIONAL(tape_qd,dl_bfloat16);
  FLASHKDA_OPTIONAL(tape_kd,dl_bfloat16); FLASHKDA_OPTIONAL(tape_kr,dl_bfloat16);
  FLASHKDA_OPTIONAL(tape_j,dl_bfloat16); FLASHKDA_OPTIONAL(tape_restore_factor,dl_float32);
  FLASHKDA_OPTIONAL(tape_e,dl_bfloat16); FLASHKDA_OPTIONAL(tape_x,dl_bfloat16);
  FLASHKDA_OPTIONAL(tape_r,dl_bfloat16); FLASHKDA_OPTIONAL(norm_inv_out,dl_float32);
  FLASHKDA_OPTIONAL(decay_out,dl_bfloat16); FLASHKDA_OPTIONAL(beta_active_out,dl_float32);
  FLASHKDA_OPTIONAL(zero_workspace,dl_uint32);
#undef FLASHKDA_OPTIONAL
#if FLASHKDA_GENERATED_AFFINE_DEPENDENCY == \
    FLASHKDA_GENERATED_AFFINE_BF16_STATE_WITH_FP32_SPLIT_DEPENDENCY
  // The public 49-argument ABI has no independent mixed-state dependency
  // carrier.  The map adapter receives that raw pointer through the otherwise
  // inactive tape_restore_factor input, while the generated kernel must still
  // see the role's typed FP32 dummy for its inactive tape slot.
  args.tape_restore_factor = norm_inv_out.data_ptr();
#endif
  args.zero_words=CheckedInt32(zero_words,"zero_words");
  args.num_sequences=CheckedInt32(num_sequences,"num_sequences");
  if (checkpoint_every_n_tokens != 0) {
    TVM_FFI_ICHECK(descriptor_storage.numel() >=
                   7 * static_cast<int64_t>(sizeof(CUtensorMap)))
        << "checkpoint descriptor storage must hold seven TensorMaps";
    auto* descriptor_bytes =
        static_cast<unsigned char*>(descriptor_storage.data_ptr());
    args.state_checkpoints_tma =
        descriptor_bytes + 6 * sizeof(CUtensorMap);
    if (prepare_descriptors != 0) {
#if FLASHKDA_GENERATED_TMA_TILE_TOKENS == 16
      // Only the checkpoint-specialized N16 kernel expects V through the
      // four-dimensional, 128-byte-swizzled descriptor.  N32 kernels retain
      // the three-dimensional descriptor prepared by PrepareCommonInputs.
      const CUtensorMap value_map = EncodeGeneratedCheckpointValueTma(v);
      GeneratedCheckpointMapWords value_words{};
      std::memcpy(value_words.words, &value_map, sizeof(value_map));
      PublishGeneratedCheckpointMap<<<
          1, sizeof(CUtensorMap) / sizeof(uint64_t), 0, prepared.stream>>>(
          reinterpret_cast<uint64_t*>(descriptor_bytes +
                                      2 * sizeof(CUtensorMap)),
          value_words);
      CheckCuda(cudaGetLastError(),
                "PublishGeneratedCheckpointValueMap launch");
#endif
      const CUtensorMap checkpoint_map =
          EncodeGeneratedCheckpointTma(state_checkpoints);
      GeneratedCheckpointMapWords checkpoint_words{};
      std::memcpy(checkpoint_words.words, &checkpoint_map,
                  sizeof(checkpoint_map));
      PublishGeneratedCheckpointMap<<<
          1, sizeof(CUtensorMap) / sizeof(uint64_t), 0, prepared.stream>>>(
          reinterpret_cast<uint64_t*>(args.state_checkpoints_tma),
          checkpoint_words);
      CheckCuda(cudaGetLastError(),
                "PublishGeneratedCheckpointMap launch");
    }
  } else {
    args.state_checkpoints_tma=CheckedDescriptorPointer(
        state_checkpoints_tma,"state_checkpoints_tma",prepared.device_id,prepared.tma.q);
  }
  auto launch = std::make_unique<PreparedDirectM128Launch>(
      args, prepared.state, CheckedGrid(grid_x, grid_y, grid_z),
      prepared.stream, prepared.device_id);
  return DirectM128PreparedHandle(launch.release());
}

inline void RunDirectM128(
    TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
    TensorView beta_tma, TensorView a_log, TensorView dt_bias,
    TensorView cu_seqlens, TensorView seq_order, TensorView state_indices,
    TensorView initial_state, TensorView out, TensorView final_state,
    TensorView state_checkpoints, TensorView checkpoint_cu_starts,
    TensorView cu_chunk_offsets, TensorView chunk_state,
    TensorView state_checkpoint_needed, TensorView tape_qd, TensorView tape_kd,
    TensorView tape_kr, TensorView tape_j, TensorView tape_restore_factor,
    TensorView tape_e, TensorView tape_x, TensorView tape_r,
    TensorView norm_inv_out, TensorView decay_out, TensorView beta_active_out,
    TensorView zero_workspace, TensorView state_checkpoints_tma,
    TensorView descriptor_storage, int64_t prepare_descriptors,
    int64_t num_heads, int64_t beta_token_stride,
    int64_t state_slot_stride, int64_t use_state_indices,
    int64_t use_initial_state, int64_t store_final_state,
    int64_t checkpoint_every_n_tokens, int64_t zero_words,
    int64_t num_sequences, double scale, double lower_bound,
    int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t cuda_stream) {
  std::unique_ptr<PreparedDirectM128Launch> launch(DirectM128PreparedPointer(
      PrepareDirectM128(
          q, k, v, g, beta, beta_tma, a_log, dt_bias, cu_seqlens, seq_order,
          state_indices, initial_state, out, final_state, state_checkpoints,
          checkpoint_cu_starts, cu_chunk_offsets, chunk_state,
          state_checkpoint_needed, tape_qd, tape_kd, tape_kr, tape_j,
          tape_restore_factor, tape_e, tape_x, tape_r, norm_inv_out, decay_out,
          beta_active_out, zero_workspace, state_checkpoints_tma,
          descriptor_storage, prepare_descriptors, num_heads,
          beta_token_stride, state_slot_stride, use_state_indices,
          use_initial_state, store_final_state, checkpoint_every_n_tokens,
          zero_words, num_sequences, scale, lower_bound, grid_x, grid_y,
          grid_z, cuda_stream)));
  LaunchPreparedDirectM128(launch.get());
}

#elif FLASHKDA_GENERATED_ABI_VARIANT == FLASHKDA_GENERATED_VARIANT_VTILE

struct DirectM128VtileArgs {
  void *q{}, *q_tma{}, *k{}, *k_tma{}, *v{}, *v_tma{}, *g{}, *g_tma{};
  void *beta{}, *beta_tma{}, *a_log{}, *dt_bias{}, *cu_seqlens{}, *seq_order{};
  void *initial_state{}, *out{}, *out_tma{}, *final_state{};
  uint64_t state_indices_addr{}; int64_t state_slot_stride{}; int32_t use_state_indices{};
  void *initial_state_f32{}, *final_state_f32{};
  int32_t uniform_seq_len{}, persistent_tasks{}, persistent_stride{};
  int32_t num_heads{}, use_initial_state{}, store_final_state{}; float scale{}, lower_bound{};
};

inline void LaunchDirectM128Vtile(DirectM128VtileArgs args,const StatePointerSlots& state,
                                  dim3 grid,cudaStream_t stream) {
  args.initial_state=state.initial_state; args.final_state=state.final_state;
  args.initial_state_f32=state.initial_state_f32; args.final_state_f32=state.final_state_f32;
  void* kernel_args[]={&args.q,&args.q_tma,&args.k,&args.k_tma,&args.v,&args.v_tma,
    &args.g,&args.g_tma,&args.beta,&args.beta_tma,&args.a_log,&args.dt_bias,
    &args.cu_seqlens,&args.seq_order,&args.initial_state,&args.out,&args.out_tma,
    &args.final_state,&args.state_indices_addr,&args.state_slot_stride,&args.use_state_indices,
    &args.initial_state_f32,&args.final_state_f32,&args.uniform_seq_len,&args.persistent_tasks,
    &args.persistent_stride,&args.num_heads,&args.use_initial_state,&args.store_final_state,
    &args.scale,&args.lower_bound};
  CheckArgumentCount<31>(kernel_args);
  ConfigureAndLaunch(FLASHKDA_GENERATED_KERNEL_ARGUMENT,grid,stream,
                     kernel_args,"generated direct-M128 vtile launch");
}

inline void RunDirectM128Vtile(
    TensorView q,TensorView k,TensorView v,TensorView g,TensorView beta,TensorView beta_tma,
    TensorView a_log,TensorView dt_bias,TensorView cu_seqlens,TensorView seq_order,
    TensorView state_indices,TensorView initial_state,TensorView out,TensorView final_state,
    TensorView descriptor_storage,int64_t prepare_descriptors,int64_t uniform_seq_len,
    int64_t persistent_tasks,int64_t persistent_stride,int64_t num_heads,
    int64_t beta_token_stride,int64_t state_slot_stride,int64_t use_state_indices,
    int64_t use_initial_state,int64_t store_final_state,double scale,double lower_bound,
    int64_t grid_x,int64_t grid_y,int64_t grid_z,int64_t cuda_stream) {
  TVM_FFI_ICHECK(q.device().device_type==kDLCUDA); ffi::CUDADeviceGuard guard(q.device().device_id);
  const auto p=PrepareCommonInputs<FLASHKDA_GENERATED_VALUE_ROWS,
      FLASHKDA_GENERATED_TMA_TILE_TOKENS,FLASHKDA_GENERATED_PAIR_PACKED_BETA!=0,
      FLASHKDA_GENERATED_VALUE_TMA_RANK==4>(
      q,k,v,g,beta,beta_tma,a_log,dt_bias,cu_seqlens,seq_order,state_indices,initial_state,
      out,final_state,descriptor_storage,prepare_descriptors,num_heads,beta_token_stride,
      state_slot_stride,use_state_indices,use_initial_state,store_final_state,scale,lower_bound,
      cuda_stream);
  DirectM128VtileArgs a{}; a.q=q.data_ptr();a.q_tma=p.tma.q;a.k=k.data_ptr();a.k_tma=p.tma.k;
  a.v=v.data_ptr();a.v_tma=p.tma.v;a.g=g.data_ptr();a.g_tma=p.tma.g;
  a.beta=beta.data_ptr();a.beta_tma=p.tma.beta;a.a_log=a_log.data_ptr();a.dt_bias=dt_bias.data_ptr();
  a.cu_seqlens=cu_seqlens.data_ptr();a.seq_order=seq_order.data_ptr();a.out=out.data_ptr();a.out_tma=p.tma.out;
  a.state_indices_addr=reinterpret_cast<uintptr_t>(state_indices.data_ptr());a.state_slot_stride=state_slot_stride;
  a.use_state_indices=CheckedInt32(use_state_indices,"use_state_indices");
  a.uniform_seq_len=CheckedInt32(uniform_seq_len,"uniform_seq_len");
  a.persistent_tasks=CheckedInt32(persistent_tasks,"persistent_tasks");
  a.persistent_stride=CheckedInt32(persistent_stride,"persistent_stride");
  a.num_heads=CheckedInt32(num_heads,"num_heads");a.use_initial_state=CheckedInt32(use_initial_state,"use_initial_state");
  a.store_final_state=CheckedInt32(store_final_state,"store_final_state");a.scale=static_cast<float>(scale);a.lower_bound=static_cast<float>(lower_bound);
  TVM_FFI_ICHECK(grid_x==persistent_stride && grid_y==1 && grid_z==1)
      << "vtile grid_x must equal persistent_stride";
  LaunchDirectM128Vtile(a,p.state,CheckedGrid(grid_x,grid_y,grid_z),p.stream);
}

#else
#error "unsupported generated direct-M128 ABI variant"
#endif

}  // namespace flashinfer::flash_kda_generated

#if FLASHKDA_GENERATED_ABI_VARIANT == FLASHKDA_GENERATED_VARIANT_SERVING
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run,flashinfer::flash_kda_generated::RunDirectM128);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_direct,flashinfer::flash_kda_generated::PrepareDirectM128);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_direct,flashinfer::flash_kda_generated::LaunchPreparedDirectM128Handle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dispose_direct,flashinfer::flash_kda_generated::DisposePreparedDirectM128Handle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_direct_python_launcher,flashinfer::flash_kda_generated::MakeDirectM128PythonLauncher);
#else
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run,flashinfer::flash_kda_generated::RunDirectM128Vtile);
#endif
