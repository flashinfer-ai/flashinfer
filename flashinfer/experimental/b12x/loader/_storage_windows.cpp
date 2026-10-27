#define PY_SSIZE_T_CLEAN
#define NOMINMAX
#include <Python.h>
#include <Windows.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <io.h>
#include <winternl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "_cuda_range.h"

extern "C" NTSYSAPI NTSTATUS NTAPI NtQueryInformationFile(HANDLE, PIO_STATUS_BLOCK, PVOID, ULONG,
                                                          FILE_INFORMATION_CLASS);

constexpr size_t alignment = 4096;
constexpr size_t metadata_bytes = 64 << 10;
constexpr size_t slot_bytes = 1 << 20;
constexpr unsigned slot_count = 16;
constexpr size_t staging_bytes = slot_count * slot_bytes;

static void win_check(BOOL success, const char* operation) {
  if (!success)
    throw std::runtime_error(std::string(operation) + ": Win32 " + std::to_string(GetLastError()));
}

static void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct WithoutGIL {
  PyThreadState* state = PyEval_SaveThread();
  ~WithoutGIL() { PyEval_RestoreThread(state); }
};

template <typename F>
static PyObject* python_call(F&& function) {
  try {
    return function();
  } catch (const std::bad_alloc&) {
    return PyErr_NoMemory();
  } catch (const std::exception& error) {
    return PyErr_Format(PyExc_RuntimeError, "%s", error.what());
  }
}

static HANDLE direct_file(int fd, uint64_t offset, uint64_t bytes) {
  HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (file == INVALID_HANDLE_VALUE || GetFileType(file) != FILE_TYPE_DISK)
    throw std::runtime_error("direct input must be a live disk file");
  IO_STATUS_BLOCK status{};
  ULONG mode = 0;
  auto result = NtQueryInformationFile(file, &status, &mode, sizeof(mode),
                                       static_cast<FILE_INFORMATION_CLASS>(16));
  if (result < 0 || !(mode & 8))
    throw std::runtime_error("direct reader requires FILE_FLAG_NO_BUFFERING");
  LARGE_INTEGER size{};
  win_check(GetFileSizeEx(file, &size), "query direct input size");
  if (offset > static_cast<uint64_t>(size.QuadPart) ||
      bytes > static_cast<uint64_t>(size.QuadPart) - offset)
    throw std::runtime_error("invalid direct input file range");
  return file;
}

static void begin_read(HANDLE file, OVERLAPPED& op, char* output, size_t bytes, uint64_t offset) {
  ResetEvent(op.hEvent);
  op.Offset = static_cast<DWORD>(offset);
  op.OffsetHigh = static_cast<DWORD>(offset >> 32);
  if (!ReadFile(file, output, static_cast<DWORD>(bytes), nullptr, &op) &&
      GetLastError() != ERROR_IO_PENDING)
    win_check(FALSE, "submit unbuffered checkpoint read");
}

static DWORD finish_read(HANDLE file, OVERLAPPED& op, size_t required) {
  DWORD bytes = 0;
  win_check(GetOverlappedResult(file, &op, &bytes, TRUE), "complete unbuffered checkpoint read");
  if (bytes < required) throw std::runtime_error("short unbuffered checkpoint read");
  return bytes;
}

struct Reader {
  std::mutex mutex;
  char* scratch = nullptr;
  OVERLAPPED op{};
  uint64_t physical_bytes = 0, reads = 0;

  void initialize() {
    scratch = static_cast<char*>(_aligned_malloc(metadata_bytes, alignment));
    if (!scratch) throw std::bad_alloc();
    op.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    win_check(op.hEvent != nullptr, "create metadata read event");
  }
  ~Reader() {
    if (op.hEvent) CloseHandle(op.hEvent);
    _aligned_free(scratch);
  }
  void read(int fd, uint64_t offset, size_t bytes, char* output) {
    HANDLE file = direct_file(fd, offset, bytes);
    while (bytes) {
      size_t delta = offset % alignment;
      size_t payload = std::min(bytes, metadata_bytes - delta);
      size_t length = (payload + delta + alignment - 1) / alignment * alignment;
      begin_read(file, op, scratch, length, offset - delta);
      physical_bytes += finish_read(file, op, delta + payload);
      reads++;
      std::memcpy(output, scratch + delta, payload);
      offset += payload;
      bytes -= payload;
      output += payload;
    }
  }
};

struct Job {
  int fd;
  uint64_t offset, bytes;
  char* destination;
  unsigned transform;
  uint64_t rows, source_stride, destination_stride;
};

struct Slot {
  OVERLAPPED op{};
  HANDLE file = INVALID_HANDLE_VALUE;
  cudaEvent_t copied = nullptr;
  bool reading = false, copying = false;
  Job job{};
  size_t delta = 0, required = 0;
};

struct Batch {
  std::mutex mutex;
  int device = 0;
  unsigned depth = 0;
  char* buffers = nullptr;
  cudaStream_t stream = nullptr;
  std::array<Slot, slot_count> slots{};
  bool poisoned = false;
  uint64_t physical_bytes = 0, reads = 0, bounced_bytes = 0, strided_bytes = 0;
  uint64_t submits = 0, batches = 0, descriptors = 0;
  unsigned max_inflight = 0;
  double execution_seconds = 0;

  void initialize(int ordinal, unsigned concurrency) {
    device = ordinal;
    depth = concurrency;
    cuda_check(cudaSetDevice(device), "select checkpoint device");
    cuda_check(
        cudaHostAlloc(reinterpret_cast<void**>(&buffers), staging_bytes, cudaHostAllocDefault),
        "allocate checkpoint staging");
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create checkpoint stream");
    for (Slot& slot : slots) {
      slot.op.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
      win_check(slot.op.hEvent != nullptr, "create checkpoint read event");
      cuda_check(cudaEventCreateWithFlags(&slot.copied, cudaEventDisableTiming),
                 "create copy event");
    }
  }

  void drain() noexcept {
    for (Slot& slot : slots) {
      if (slot.reading) {
        CancelIoEx(slot.file, &slot.op);
        DWORD ignored = 0;
        GetOverlappedResult(slot.file, &slot.op, &ignored, TRUE);
        slot.reading = false;
      }
    }
    if (stream) cudaStreamSynchronize(stream);
  }

  ~Batch() {
    int previous = device;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    drain();
    for (Slot& slot : slots) {
      if (slot.op.hEvent) CloseHandle(slot.op.hEvent);
      if (slot.copied) cudaEventDestroy(slot.copied);
    }
    if (stream) cudaStreamDestroy(stream);
    if (buffers) cudaFreeHost(buffers);
    cudaSetDevice(previous);
  }

  void copy(unsigned index) {
    Slot& slot = slots[index];
    if (!slot.reading) return;
    physical_bytes += finish_read(slot.file, slot.op, slot.required);
    slot.reading = false;
    reads++;
    char* buffer = buffers + index * slot_bytes;
    char* source = buffer + slot.delta;
    size_t width = slot.job.bytes;
    size_t pitch = slot.job.rows > 1 ? slot.job.source_stride : width;
    if (slot.job.transform == 1) {
      for (uint64_t row = 0; row < slot.job.rows; row++)
        std::memmove(buffer + row * width, source + row * pitch, width);
      for (uint64_t i = slot.job.rows * width / 2; i > 0;) {
        --i;
        uint16_t bf16;
        std::memcpy(&bf16, buffer + 2 * i, 2);
        uint32_t fp32 = static_cast<uint32_t>(bf16) << 16;
        std::memcpy(buffer + 4 * i, &fp32, 4);
      }
      source = buffer;
      width *= 2;
      pitch = width;
    }
    cuda_check(cudaMemcpy2DAsync(slot.job.destination,
                                 slot.job.rows > 1 ? slot.job.destination_stride : width, source,
                                 pitch, width, slot.job.rows, cudaMemcpyHostToDevice, stream),
               "copy checkpoint staging to device");
    cuda_check(cudaEventRecord(slot.copied, stream), "record checkpoint copy");
    slot.copying = true;
    bounced_bytes += slot.job.rows * slot.job.bytes;
    if (slot.job.rows > 1) strided_bytes += slot.job.rows * slot.job.bytes;
  }

  void run(const std::vector<Job>& jobs, uint64_t descriptor_count, cudaStream_t producer) {
    if (poisoned) throw std::runtime_error("checkpoint executor is unusable after an I/O failure");
    for (const Job& job : jobs) {
      if (job.transform != 2)
        direct_file(job.fd, job.offset, (job.rows - 1) * job.source_stride + job.bytes);
    }
    cuda_check(cudaSetDevice(device), "select checkpoint device");
    cuda_check(cudaStreamSynchronize(producer), "synchronize checkpoint destinations");
    auto start = std::chrono::steady_clock::now();
    try {
      unsigned cursor = 0, outstanding = 0;
      for (const Job& job : jobs) {
        if (job.transform == 2) {
          cuda_check(cudaMemcpyAsync(job.destination, reinterpret_cast<void*>(job.offset),
                                     job.bytes, cudaMemcpyHostToDevice, stream),
                     "copy checkpoint metadata");
          continue;
        }
        HANDLE file =
            direct_file(job.fd, job.offset, (job.rows - 1) * job.source_stride + job.bytes);
        uint64_t row = 0, within = 0;
        while (row < job.rows && job.bytes) {
          unsigned index = cursor++ % depth;
          Slot& slot = slots[index];
          if (slot.reading) {
            copy(index);
            outstanding--;
          }
          if (slot.copying) {
            cuda_check(cudaEventSynchronize(slot.copied), "recycle checkpoint staging slot");
            slot.copying = false;
          }
          slot.file = file;
          slot.job = job;
          uint64_t offset = job.offset + row * job.source_stride + within;
          slot.delta = offset % alignment;
          slot.job.destination +=
              row * job.destination_stride + within * (1 + (job.transform == 1));
          slot.job.bytes = std::min(job.bytes - within, slot_bytes - slot.delta);
          slot.job.rows = 1;
          if (job.transform == 1)
            slot.job.bytes = std::min(slot.job.bytes, slot_bytes / 2) & ~uint64_t(1);
          if (!within && job.bytes < alignment && job.source_stride >= job.bytes) {
            slot.job.rows = std::min(job.rows - row,
                                     1 + (slot_bytes - slot.delta - job.bytes) / job.source_stride);
          }
          if (job.transform == 1)
            slot.job.rows = std::min(slot.job.rows, slot_bytes / (2 * slot.job.bytes));
          slot.required = slot.delta + (slot.job.rows - 1) * job.source_stride + slot.job.bytes;
          size_t length = (slot.required + alignment - 1) / alignment * alignment;
          begin_read(file, slot.op, buffers + index * slot_bytes, length, offset - slot.delta);
          slot.reading = true;
          submits++;
          max_inflight = std::max(max_inflight, ++outstanding);
          within += slot.job.bytes;
          if (within == job.bytes) {
            row += slot.job.rows;
            within = 0;
          }
        }
      }
      for (unsigned i = 0; i < depth; i++) copy(i);
      cuda_check(cudaStreamSynchronize(stream), "complete checkpoint copies");
      for (Slot& slot : slots) slot.copying = false;
      batches++;
      descriptors += descriptor_count;
      execution_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    } catch (...) {
      poisoned = true;
      drain();
      throw;
    }
  }
};

static PyObject* direct_open(PyObject*, PyObject* path) {
  return python_call([&]() -> PyObject* {
    wchar_t* text = PyUnicode_AsWideCharString(path, nullptr);
    if (!text) return nullptr;
    HANDLE file =
        CreateFileW(text, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    PyMem_Free(text);
    win_check(file != INVALID_HANDLE_VALUE, "open unbuffered checkpoint");
    int fd =
        _open_osfhandle(reinterpret_cast<intptr_t>(file), _O_RDONLY | _O_BINARY | _O_NOINHERIT);
    if (fd < 0) {
      CloseHandle(file);
      throw std::runtime_error("cannot own checkpoint descriptor");
    }
    return PyLong_FromLong(fd);
  });
}

static PyObject* direct_reader(PyObject*, PyObject* args) {
  int device;
  if (!PyArg_ParseTuple(args, "i", &device)) return nullptr;
  return python_call([&]() -> PyObject* {
    cuda_check(cudaSetDevice(device), "select metadata device");
    auto reader = std::make_unique<Reader>();
    reader->initialize();
    auto capsule = PyCapsule_New(reader.get(), "b12x.direct_reader", [](PyObject* p) {
      delete static_cast<Reader*>(PyCapsule_GetPointer(p, "b12x.direct_reader"));
    });
    if (capsule) reader.release();
    return capsule;
  });
}

static PyObject* direct_bytes(PyObject*, PyObject* args) {
  PyObject* capsule;
  int fd;
  long long offset, bytes;
  if (!PyArg_ParseTuple(args, "OiLL", &capsule, &fd, &offset, &bytes)) return nullptr;
  auto reader = static_cast<Reader*>(PyCapsule_GetPointer(capsule, "b12x.direct_reader"));
  if (!reader) return nullptr;
  if (offset < 0 || bytes < 0 || bytes > 100 * (1 << 20))
    return PyErr_Format(PyExc_ValueError,
                        "invalid header/metadata range or length exceeds 100 MiB");
  return python_call([&]() -> PyObject* {
    std::string output(static_cast<size_t>(bytes), '\0');
    {
      WithoutGIL released;
      std::lock_guard<std::mutex> locked(reader->mutex);
      reader->read(fd, offset, static_cast<size_t>(bytes), output.data());
    }
    return PyBytes_FromStringAndSize(output.data(), bytes);
  });
}

static PyObject* direct_stats(PyObject*, PyObject* capsule) {
  auto reader = static_cast<Reader*>(PyCapsule_GetPointer(capsule, "b12x.direct_reader"));
  if (!reader) return nullptr;
  std::lock_guard<std::mutex> locked(reader->mutex);
  return Py_BuildValue("{s:K,s:K,s:K}", "physical_bytes", reader->physical_bytes, "reads",
                       reader->reads, "scratch_bytes", uint64_t(metadata_bytes));
}

static PyObject* batch_executor(PyObject*, PyObject* args) {
  int device, workers;
  if (!PyArg_ParseTuple(args, "ii", &device, &workers)) return nullptr;
  if (workers < 1 || workers > 16)
    return PyErr_Format(PyExc_ValueError, "io_threads must be between 1 and 16");
  return python_call([&]() -> PyObject* {
    auto batch = std::make_unique<Batch>();
    batch->initialize(device, static_cast<unsigned>(workers));
    auto capsule = PyCapsule_New(batch.get(), "b12x.batch_executor", [](PyObject* p) {
      delete static_cast<Batch*>(PyCapsule_GetPointer(p, "b12x.batch_executor"));
    });
    if (capsule) batch.release();
    return capsule;
  });
}

static PyObject* batch_execute(PyObject*, PyObject* args) {
  PyObject* capsule;
  Py_buffer records{};
  unsigned long long stream;
  if (!PyArg_ParseTuple(args, "Oy*K", &capsule, &records, &stream)) return nullptr;
  auto batch = static_cast<Batch*>(PyCapsule_GetPointer(capsule, "b12x.batch_executor"));
  if (!batch) {
    PyBuffer_Release(&records);
    return nullptr;
  }
  auto result = python_call([&]() -> PyObject* {
    if (records.len % (8 * sizeof(uint64_t)))
      throw std::runtime_error("invalid read descriptor byte size");
    std::vector<Job> jobs;
    size_t count = static_cast<size_t>(records.len) / (8 * sizeof(uint64_t));
    for (size_t i = 0; i < count; i++) {
      uint64_t r[8];
      std::memcpy(r, static_cast<char*>(records.buf) + i * sizeof(r), sizeof(r));
      auto [fd, offset, bytes, pointer, transform, rows, source_stride, destination_stride] =
          std::array<uint64_t, 8>{r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]};
      uint64_t factor = 1 + (transform == 1);
      if (fd > INT_MAX || transform > 2 || offset > INT64_MAX || bytes > INT64_MAX - offset ||
          (transform == 1 && (bytes % 2 || bytes > INT64_MAX / 2)) ||
          (transform == 2 && ((!offset && bytes) || rows != 1)) || !rows || rows > INT64_MAX ||
          source_stride > INT64_MAX || destination_stride > INT64_MAX ||
          (source_stride && rows - 1 > (INT64_MAX - offset - bytes) / source_stride) ||
          (destination_stride && rows - 1 > (INT64_MAX - bytes * factor) / destination_stride))
        throw std::runtime_error("invalid read descriptor");
      if (rows > 1 && destination_stride < bytes * factor)
        throw std::runtime_error("overlapping batch destinations need an explicit dependency");
      if (!device_range(pointer, (rows - 1) * destination_stride + bytes * factor, batch->device))
        throw std::runtime_error("batch destination exceeds CUDA device allocation");
      if (bytes)
        jobs.push_back({static_cast<int>(fd), offset, bytes, reinterpret_cast<char*>(pointer),
                        static_cast<unsigned>(transform), rows, source_stride, destination_stride});
    }
    std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
      return reinterpret_cast<uintptr_t>(a.destination) <
             reinterpret_cast<uintptr_t>(b.destination);
    });
    for (size_t i = 1; i < jobs.size(); i++) {
      const Job& p = jobs[i - 1];
      if (reinterpret_cast<uintptr_t>(p.destination) + (p.rows - 1) * p.destination_stride +
              p.bytes * (1 + (p.transform == 1)) >
          reinterpret_cast<uintptr_t>(jobs[i].destination))
        throw std::runtime_error("overlapping batch destinations need an explicit dependency");
    }
    {
      WithoutGIL released;
      std::lock_guard<std::mutex> locked(batch->mutex);
      batch->run(jobs, count, reinterpret_cast<cudaStream_t>(stream));
    }
    Py_RETURN_NONE;
  });
  PyBuffer_Release(&records);
  return result;
}

static PyObject* batch_stats(PyObject*, PyObject* capsule) {
  auto b = static_cast<Batch*>(PyCapsule_GetPointer(capsule, "b12x.batch_executor"));
  if (!b) return nullptr;
  std::lock_guard<std::mutex> locked(b->mutex);
  return Py_BuildValue("{s:K,s:K,s:K,s:K,s:K,s:K,s:d,s:K,s:K,s:K,s:I}", "physical_bytes",
                       b->physical_bytes, "strided_copy_bytes", b->strided_bytes, "reads", b->reads,
                       "scratch_bytes", uint64_t(staging_bytes), "batches", b->batches,
                       "descriptors", b->descriptors, "execution_seconds", b->execution_seconds,
                       "bounce_buffer_bytes", uint64_t(staging_bytes), "bounced_bytes",
                       b->bounced_bytes, "overlapped_submits", b->submits, "max_inflight_reads",
                       b->max_inflight);
}

static PyObject* capabilities(PyObject*, PyObject* args) {
  int device;
  if (!PyArg_ParseTuple(args, "i", &device)) return nullptr;
  static const std::pair<const char*, cudaDeviceAttr> keys[] = {
      {"integrated", cudaDevAttrIntegrated},
      {"can_map_host_memory", cudaDevAttrCanMapHostMemory},
      {"managed_memory", cudaDevAttrManagedMemory},
      {"concurrent_managed_access", cudaDevAttrConcurrentManagedAccess},
      {"pageable_memory_access", cudaDevAttrPageableMemoryAccess},
      {"host_page_tables", cudaDevAttrPageableMemoryAccessUsesHostPageTables},
      {"host_register_supported", cudaDevAttrHostRegisterSupported},
      {"registered_host_pointer", cudaDevAttrCanUseHostPointerForRegisteredMem}};
  PyObject* result = PyDict_New();
  if (!result) return nullptr;
  for (const auto& key : keys) {
    int value;
    auto status = cudaDeviceGetAttribute(&value, key.second, device);
    if (status != cudaSuccess) {
      Py_DECREF(result);
      return PyErr_Format(PyExc_RuntimeError, "%s", cudaGetErrorString(status));
    }
    PyObject* number = PyLong_FromLong(value);
    if (!number || PyDict_SetItemString(result, key.first, number) < 0) {
      Py_XDECREF(number);
      Py_DECREF(result);
      return nullptr;
    }
    Py_DECREF(number);
  }
  return result;
}

static PyMethodDef methods[] = {{"direct_open", direct_open, METH_O, nullptr},
                                {"direct_reader", direct_reader, METH_VARARGS, nullptr},
                                {"direct_bytes", direct_bytes, METH_VARARGS, nullptr},
                                {"direct_stats", direct_stats, METH_O, nullptr},
                                {"batch_executor", batch_executor, METH_VARARGS, nullptr},
                                {"batch_execute", batch_execute, METH_VARARGS, nullptr},
                                {"batch_stats", batch_stats, METH_O, nullptr},
                                {"capabilities", capabilities, METH_VARARGS, nullptr},
                                {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT,
                             "_b12x_loader_storage",
                             nullptr,
                             -1,
                             methods,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr};

PyMODINIT_FUNC PyInit__b12x_loader_storage() {
  PyObject* result = PyModule_Create(&module);
  if (result && PyModule_AddIntConstant(result, "ABI_VERSION", 1) < 0) {
    Py_DECREF(result);
    return nullptr;
  }
  return result;
}
