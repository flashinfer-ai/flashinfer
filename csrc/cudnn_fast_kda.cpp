// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Reuse the compiled KDA host with a checked, native argument frame.
#include <cuda.h>
#include <dlfcn.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace ffi = tvm::ffi;
constexpr size_t kInputs = 10;
constexpr size_t kArgs = 39;

struct Driver {
  void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  template <typename T>
  T load(const char* name) {
    if (!library) throw std::runtime_error("CUDA driver is unavailable");
    auto value = reinterpret_cast<T>(dlsym(library, name));
    if (!value) throw std::runtime_error(std::string("Missing CUDA entry: ") + name);
    return value;
  }
  decltype(&cuCtxGetCurrent) get_context = load<decltype(&cuCtxGetCurrent)>("cuCtxGetCurrent");
  decltype(&cuCtxSetCurrent) set_context = load<decltype(&cuCtxSetCurrent)>("cuCtxSetCurrent");
  decltype(&cuStreamIsCapturing) capturing =
      load<decltype(&cuStreamIsCapturing)>("cuStreamIsCapturing");
  decltype(&cuStreamSynchronize) synchronize =
      load<decltype(&cuStreamSynchronize)>("cuStreamSynchronize");
  decltype(&cuEventCreate) event_create = load<decltype(&cuEventCreate)>("cuEventCreate");
  decltype(&cuEventRecord) event_record = load<decltype(&cuEventRecord)>("cuEventRecord");
  decltype(&cuEventQuery) event_query = load<decltype(&cuEventQuery)>("cuEventQuery");
  decltype(&cuEventDestroy) event_destroy = load<decltype(&cuEventDestroy)>("cuEventDestroy_v2");
};

Driver& driver() {
  static Driver value;
  return value;
}

void check_cuda(CUresult result, const char* operation) {
  if (result != CUDA_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(result));
}

struct ContextGuard {
  CUcontext previous = nullptr, target;
  explicit ContextGuard(CUcontext context) : target(context) {
    check_cuda(driver().get_context(&previous), "cuCtxGetCurrent");
    if (previous != target) check_cuda(driver().set_context(target), "cuCtxSetCurrent");
  }
  ~ContextGuard() {
    if (previous != target) driver().set_context(previous);
  }
};

struct Exchange {
  py::object type;
  DLPackExchangeAPI* api;
};

DLPackExchangeAPI* exchange(PyObject* object) {
  // Owned type references prevent address reuse in this small process cache.
  static auto* cache = new std::vector<Exchange>();
  PyObject* type = reinterpret_cast<PyObject*>(Py_TYPE(object));
  for (const auto& item : *cache)
    if (item.type.ptr() == type) return item.api;
  py::object capsule =
      py::reinterpret_steal<py::object>(PyObject_GetAttrString(type, "__dlpack_c_exchange_api__"));
  if (!capsule) {
    PyErr_Clear();
    return nullptr;
  }
  auto* api =
      static_cast<DLPackExchangeAPI*>(PyCapsule_GetPointer(capsule.ptr(), "dlpack_exchange_api"));
  if (!api) {
    PyErr_Clear();
    return nullptr;
  }
  for (int i = 0; api && i < 8; ++i) {
    if (api->header.version.major == DLPACK_MAJOR_VERSION) {
      if (!api->dltensor_from_py_object_no_sync) return nullptr;
      cache->push_back({py::reinterpret_borrow<py::object>(type), api});
      return api;
    }
    api = reinterpret_cast<DLPackExchangeAPI*>(api->header.prev_api);
  }
  return nullptr;
}

bool observe(py::handle object, DLTensor& tensor) {
  auto* api = exchange(object.ptr());
  if (!api) return false;
  if (api->dltensor_from_py_object_no_sync(object.ptr(), &tensor) != 0)
    throw py::error_already_set();
  return true;
}

struct Metadata {
  uintptr_t pointer = 0;
  uint64_t bytes = 0;
  int32_t device = -1;
  DLDataType dtype{};
  int rank = 0;
  std::array<int64_t, 4> shape{};
  std::array<int64_t, 4> strides{};
};

bool metadata(const DLTensor& t, size_t role, Metadata& m) {
  if (t.device.device_type != kDLCUDA || t.ndim < 1 || t.ndim > 4 || !t.shape || !t.data)
    return false;
  if (t.dtype.lanes != 1 || !t.dtype.bits) return false;
  m.device = t.device.device_id;
  m.dtype = t.dtype;
  auto base = reinterpret_cast<uintptr_t>(t.data);
  if (t.byte_offset > std::numeric_limits<uintptr_t>::max() - base) return false;
  m.pointer = base + t.byte_offset;
  uint64_t size = (uint64_t(t.dtype.bits) + 7) / 8;
  int64_t stride = 1;
  std::array<int64_t, 4> actual{};
  for (int i = t.ndim - 1; i >= 0; --i) {
    if (t.shape[i] <= 0 || t.shape[i] > std::numeric_limits<int64_t>::max() / stride) return false;
    actual[i] = t.strides ? t.strides[i] : stride;
    if (t.shape[i] > 1 && actual[i] != stride) return false;
    if (actual[i] < 0) return false;
    stride *= t.shape[i];
  }
  if (uint64_t(stride) > std::numeric_limits<uint64_t>::max() / size) return false;
  m.bytes = uint64_t(stride) * size;
  if (m.bytes > std::numeric_limits<uintptr_t>::max() - m.pointer) return false;
  int first = 0;
  if (role <= 3 || role == 6) {
    if (t.ndim == 4 && t.shape[0] == 1)
      first = 1;
    else if (t.ndim != 3)
      return false;
  } else if (role == 4) {
    if (t.ndim == 3 && t.shape[0] == 1)
      first = 1;
    else if (t.ndim != 2)
      return false;
  } else if (role == 8) {
    if (t.ndim != 1 && t.ndim != 2) return false;
    m.rank = 1;
    m.shape[0] = stride;
    m.strides[0] = 1;
    return true;
  } else if ((role == 5 || role == 7) && t.ndim != 1)
    return false;
  else if (role == 9 && t.ndim != 4)
    return false;
  m.rank = t.ndim - first;
  for (int i = first; i < t.ndim; ++i) {
    m.shape[i - first] = t.shape[i];
    m.strides[i - first] = t.shape[i] == 1 ? 0 : actual[i];
  }
  return true;
}

bool same_contract(const Metadata& a, const Metadata& b) {
  return a.device == b.device && a.dtype.code == b.dtype.code && a.dtype.bits == b.dtype.bits &&
         a.dtype.lanes == b.dtype.lanes && a.rank == b.rank && a.shape == b.shape &&
         a.strides == b.strides && a.bytes == b.bytes;
}

bool overlaps(const Metadata& a, const Metadata& b) {
  return a.pointer < b.pointer + b.bytes && b.pointer < a.pointer + a.bytes;
}

struct TensorSlot {
  DLTensor tensor{};
  std::vector<int64_t> shape, strides;
  void set(const DLTensor& source) {
    tensor = source;
    shape.assign(source.shape, source.shape + source.ndim);
    strides.resize(source.ndim);
    int64_t compact = 1;
    for (int i = source.ndim - 1; i >= 0; --i) {
      strides[i] = source.strides ? source.strides[i] : compact;
      compact *= source.shape[i];
    }
    tensor.shape = shape.data();
    tensor.strides = strides.data();
    tensor.data = static_cast<char*>(source.data) + source.byte_offset;
    tensor.byte_offset = 0;
  }
};

class cuDNNFastKDA {
  struct Retired {
    CUevent event;
    py::tuple inputs;
  };
  py::object owners_;
  ffi::Function function_;
  py::tuple frame_owner_, live_inputs_;
  std::array<Metadata, kInputs> expected_, current_;
  std::array<TensorSlot, kArgs> tensors_;
  std::array<ffi::AnyView, kArgs> args_;
  std::array<int, kArgs> input_at_;
  std::vector<Metadata> scratch_;
  std::deque<Retired> retired_;
  CUstream stream_ = nullptr;
  CUcontext context_ = nullptr;
  uint64_t hits_ = 0;
  bool have_inputs_ = false, launched_ = false;

  bool valid_aliases(const std::array<Metadata, kInputs>& values) const {
    for (size_t output : {size_t(6), size_t(9)})
      for (size_t i = 0; i < kInputs; ++i)
        if (i != output && overlaps(values[output], values[i])) return false;
    for (const auto& region : scratch_)
      for (const auto& value : values)
        if (overlaps(region, value)) return false;
    return true;
  }

  void retain_inputs(py::tuple inputs, const std::array<Metadata, kInputs>& values) {
    bool changed = !have_inputs_;
    if (have_inputs_)
      for (size_t i = 0; i < kInputs; ++i)
        changed |=
            inputs[i].ptr() != live_inputs_[i].ptr() || values[i].pointer != current_[i].pointer;
    if (!changed) return;
    auto& d = driver();
    while (!retired_.empty()) {
      auto status = d.event_query(retired_.front().event);
      if (status == CUDA_ERROR_NOT_READY) break;
      check_cuda(status, "cuEventQuery");
      check_cuda(d.event_destroy(retired_.front().event), "cuEventDestroy");
      retired_.pop_front();
    }
    if (have_inputs_ && launched_) {
      CUevent event;
      check_cuda(d.event_create(&event, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
      auto status = d.event_record(event, stream_);
      if (status != CUDA_SUCCESS) {
        d.event_destroy(event);
        check_cuda(status, "cuEventRecord");
      }
      retired_.push_back({event, live_inputs_});
    }
    live_inputs_ = std::move(inputs);
    current_ = values;
    have_inputs_ = true;
  }

 public:
  cuDNNFastKDA(const std::string& name, py::tuple frame, py::tuple samples,
               std::vector<int> slot_inputs, py::object owners)
      : owners_(std::move(owners)),
        function_(ffi::Function::GetGlobalRequired(name)),
        frame_owner_(frame) {
    if (frame.size() != kArgs || samples.size() != kInputs || slot_inputs.size() != kArgs)
      throw py::value_error("KDA requires 39 frame slots and 10 input tensors");
    auto& d = driver();
    check_cuda(d.get_context(&context_), "cuCtxGetCurrent");
    if (!context_) throw py::value_error("KDA preparation requires the CUDA context");
    stream_ = reinterpret_cast<CUstream>(
        py::cast<int64_t>(py::module_::import("builtins").attr("int")(frame[kArgs - 1])));
    std::array<bool, kInputs> mapped{};
    for (size_t i = 0; i < kInputs; ++i) {
      DLTensor t{};
      if (!observe(samples[i], t) || !metadata(t, i, expected_[i]))
        throw py::value_error("KDA fast path requires contiguous CUDA tensor metadata");
      if (i && expected_[i].device != expected_[0].device)
        throw py::value_error("KDA inputs must share a device");
    }
    for (size_t i = 0; i < kArgs; ++i) {
      int input = slot_inputs[i];
      if (input < -1 || input >= int(kInputs)) throw py::value_error("Invalid KDA input slot");
      input_at_[i] = input;
      if (i == kArgs - 1) {
        if (input != -1) throw py::value_error("The final frame slot must be the stream");
        args_[i] = ffi::AnyView(reinterpret_cast<void*>(stream_));
      } else if (frame[i].is_none()) {
        if (input != -1) throw py::value_error("An absent operand cannot bind an input");
        args_[i] = ffi::AnyView(nullptr);
      } else if (py::isinstance<py::bool_>(frame[i])) {
        args_[i] = ffi::AnyView(py::cast<bool>(frame[i]));
      } else if (py::isinstance<py::int_>(frame[i])) {
        args_[i] = ffi::AnyView(py::cast<int64_t>(frame[i]));
      } else if (py::isinstance<py::float_>(frame[i])) {
        args_[i] = ffi::AnyView(py::cast<double>(frame[i]));
      } else {
        DLTensor t{};
        if (!observe(frame[i], t)) throw py::value_error("Unsupported KDA frame argument");
        tensors_[i].set(t);
        args_[i] = ffi::AnyView(&tensors_[i].tensor);
        if (input >= 0) {
          if (reinterpret_cast<uintptr_t>(tensors_[i].tensor.data) != expected_[input].pointer)
            throw py::value_error("KDA input mapping does not match its sample pointer");
          mapped[input] = true;
        } else if (t.data) {
          Metadata span;
          // Scratch slots have ranks 1 through 4 and compact layouts.
          span.pointer = reinterpret_cast<uintptr_t>(tensors_[i].tensor.data);
          span.bytes = (uint64_t(t.dtype.bits) * t.dtype.lanes + 7) / 8;
          for (int dim = 0; dim < t.ndim; ++dim) span.bytes *= t.shape[dim];
          scratch_.push_back(span);
        }
      }
    }
    for (bool present : mapped)
      if (!present) throw py::value_error("Every public input needs a native frame slot");
    if (!valid_aliases(expected_))
      throw py::value_error("KDA outputs/state must not overlap other inputs or scratch");
  }

  ~cuDNNFastKDA() {
    if (!launched_) return;
    auto& d = driver();
    CUcontext previous = nullptr;
    d.get_context(&previous);
    if (previous != context_) d.set_context(context_);
    // Cold teardown keeps queued GPU work from losing its owners.
    d.synchronize(stream_);
    for (const auto& old : retired_) d.event_destroy(old.event);
    if (previous != context_) d.set_context(previous);
  }

  bool execute(py::tuple inputs, int64_t stream = -1) {
    if (inputs.size() != kInputs) return false;
    std::array<Metadata, kInputs> values;
    for (size_t i = 0; i < kInputs; ++i) {
      DLTensor t{};
      if (!observe(inputs[i], t) || !metadata(t, i, values[i]) ||
          !same_contract(values[i], expected_[i]))
        return false;
      const size_t alignment = (i == 4 || i == 5 || i == 7) ? 4 : 16;
      if (values[i].pointer % alignment) return false;
    }
    if (!valid_aliases(values)) return false;
    if (stream == -1) {
      auto* api = exchange(inputs[0].ptr());
      if (!api || !api->current_work_stream) return false;
      void* value = nullptr;
      if (api->current_work_stream(kDLCUDA, expected_[0].device, &value) != 0)
        throw py::error_already_set();
      stream = reinterpret_cast<int64_t>(value);
    }
    if (reinterpret_cast<CUstream>(stream) != stream_) return false;
    auto& d = driver();
    ContextGuard context(context_);
    CUstreamCaptureStatus capture;
    check_cuda(d.capturing(stream_, &capture), "cuStreamIsCapturing");
    if (capture != CU_STREAM_CAPTURE_STATUS_NONE) return false;
    retain_inputs(inputs, values);
    for (size_t i = 0; i < kArgs; ++i)
      if (input_at_[i] >= 0)
        tensors_[i].tensor.data = reinterpret_cast<void*>(values[input_at_[i]].pointer);
    ffi::Any result;
    launched_ = true;
    function_.CallPacked(args_.data(), kArgs, &result);
    ++hits_;
    return true;
  }

  bool can_release() const {
    if (!launched_) return true;
    ContextGuard context(context_);
    CUstreamCaptureStatus capture;
    return driver().capturing(stream_, &capture) == CUDA_SUCCESS &&
           capture == CU_STREAM_CAPTURE_STATUS_NONE;
  }

  uint64_t hits() const { return hits_; }
};

PYBIND11_MODULE(_cudnn_fast_kda, module) {
  py::class_<cuDNNFastKDA>(module, "cuDNNFastKDA", py::module_local())
      .def(py::init<const std::string&, py::tuple, py::tuple, std::vector<int>, py::object>())
      .def("execute", &cuDNNFastKDA::execute, py::arg("inputs"), py::arg("stream") = -1)
      .def("can_release", &cuDNNFastKDA::can_release)
      .def_property_readonly("hits", &cuDNNFastKDA::hits);
}
