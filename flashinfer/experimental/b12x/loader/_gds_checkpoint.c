/* Included by _gds_reader.c. Payloads enter device memory through strict cuFile. */
#include <limits.h>
#include "_cuda_range.h"

#define CHECKPOINT_CAPSULE "b12x.gds_checkpoint"
#define CHECKPOINT_SCRATCH (8u << 20)
#define CHECKPOINT_CHUNK (64u << 20)
#define CHECKPOINT_PAGE 65536u

typedef struct {
    int fd;
    int64_t size;
    CUfileHandle_t handle;
} checkpoint_file_t;

typedef struct {
    int fd;
    CUfileHandle_t handle;
    int64_t file_size, offset, bytes, rows, source_stride, destination_stride;
    char *destination;
    bool expand, host_copy;
} checkpoint_job_t;

typedef struct checkpoint_executor checkpoint_executor_t;
typedef struct {
    checkpoint_executor_t *executor;
    pthread_t thread;
    void *allocation, *scratch;
    bool registered;
    cudaStream_t stream;
    uint64_t physical_bytes, destination_bytes, realigned_bytes, strided_copy_bytes;
    uint64_t reads, device_copy_bytes, expanded_bytes, host_copy_bytes;
} checkpoint_worker_t;

struct checkpoint_executor {
    int device, count, running, version;
    pthread_mutex_t mutex;
    pthread_cond_t ready, done;
    bool stopping, active, poisoned, closed;
    uint64_t generation, batches, descriptors;
    double execution_seconds;
    checkpoint_worker_t workers[16];
    checkpoint_file_t *files;
    size_t file_count;
    checkpoint_job_t *jobs;
    size_t job_count, next_job;
    CUfunction copy, expand;
    failure_t failure;
};

static bool checkpoint_read(checkpoint_worker_t *w, const checkpoint_job_t *job,
                            void *target, int64_t offset, size_t length, failure_t *failure) {
    ssize_t expected = job->file_size - offset < (int64_t)length ? job->file_size - offset : (ssize_t)length;
    ssize_t result = cuFileRead(job->handle, target, length, offset, 0);
    w->reads++;
    if (result > 0) w->physical_bytes += result;
    if (result == expected) return true;
    if (result == -1) system_error(failure, "cuFileRead checkpoint");
    else snprintf(failure->message, sizeof(failure->message),
        "short or failed GDS checkpoint read at offset %lld: expected %lld, received %lld",
        (long long)offset, (long long)expected, (long long)result);
    return false;
}

static bool checkpoint_copy(checkpoint_worker_t *w, void *source, void *destination,
        int64_t bytes, int64_t rows, int64_t source_stride, int64_t destination_stride,
        bool expand, failure_t *failure) {
    if (!expand && rows == 1) {
        if (!gpu_ok(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, w->stream),
                    "copy GDS checkpoint edge", failure)) return false;
    } else {
        void *scratch = NULL;
        void *args[] = {&source, &destination, &bytes, &rows, &source_stride,
                       &destination_stride, &scratch, &scratch};
        uint64_t elements = (uint64_t)rows * bytes / (expand ? 2 : 1);
        if (!driver_ok(cuLaunchKernel(expand ? w->executor->expand : w->executor->copy,
                (unsigned)((elements + 1023) / 1024), 1, 1, 128, 1, 1, 0,
                (CUstream)w->stream, args, NULL), "launch GDS checkpoint copy", failure)) return false;
    }
    w->device_copy_bytes += rows * bytes * (expand ? 2 : 1);
    if (expand) w->expanded_bytes += rows * bytes;
    return gpu_ok(cudaStreamSynchronize(w->stream), "complete GDS checkpoint copy", failure);
}

static bool checkpoint_range(checkpoint_worker_t *w, const checkpoint_job_t *job,
        int64_t offset, int64_t bytes, char *destination, failure_t *failure) {
    while (bytes) {
        bool direct = !job->expand && !((uintptr_t)destination % CHECKPOINT_PAGE) &&
                      !(offset % PLE_BLOCK) && bytes >= CHECKPOINT_PAGE;
        if (direct) {
            size_t length = bytes < CHECKPOINT_SCRATCH ? (size_t)bytes : CHECKPOINT_SCRATCH;
            length &= ~(size_t)(CHECKPOINT_PAGE - 1);
            if (!gds_ok(cuFileBufRegister(destination, length, 0),
                        "register GDS final weight range", failure)) return false;
            bool ok = checkpoint_read(w, job, destination, offset, length, failure);
            if (!gds_ok(cuFileBufDeregister(destination), "deregister GDS final weight range", failure)) ok = false;
            if (!ok) return false;
            w->destination_bytes += length;
            offset += length; bytes -= length; destination += length;
            continue;
        }
        int64_t aligned = offset & ~(int64_t)(PLE_BLOCK - 1);
        size_t delta = offset - aligned;
        size_t payload = (uint64_t)bytes < CHECKPOINT_SCRATCH - delta ? (size_t)bytes : CHECKPOINT_SCRATCH - delta;
        if (!job->expand && !(offset % PLE_BLOCK) && (uintptr_t)destination % CHECKPOINT_PAGE) {
            size_t prefix = CHECKPOINT_PAGE - (uintptr_t)destination % CHECKPOINT_PAGE;
            if (payload > prefix) payload = prefix;
        }
        if (job->expand) payload &= ~(size_t)1;
        size_t length = (delta + payload + PLE_BLOCK - 1) & ~(size_t)(PLE_BLOCK - 1);
        if (!checkpoint_read(w, job, w->scratch, aligned, length, failure) ||
            !checkpoint_copy(w, (char *)w->scratch + delta, destination, payload, 1, 0, 0,
                             job->expand, failure)) return false;
        w->realigned_bytes += payload;
        offset += payload; bytes -= payload; destination += payload * (job->expand ? 2 : 1);
    }
    return true;
}

static bool checkpoint_rows(checkpoint_worker_t *w, const checkpoint_job_t *job, failure_t *failure) {
    if (job->host_copy) {
        if (!gpu_ok(cudaMemcpyAsync(job->destination, (void *)(uintptr_t)job->offset, job->bytes,
                    cudaMemcpyHostToDevice, w->stream), "copy checkpoint control metadata", failure)) return false;
        w->host_copy_bytes += job->bytes;
        return gpu_ok(cudaStreamSynchronize(w->stream), "complete checkpoint control metadata", failure);
    }
    int64_t offset = job->offset, rows = job->rows;
    char *destination = job->destination;
    while (rows) {
        int64_t aligned = offset & ~(int64_t)(PLE_BLOCK - 1);
        size_t delta = offset - aligned;
        if (rows == 1 || job->bytes >= PLE_BLOCK || !job->source_stride ||
            job->source_stride > CHECKPOINT_SCRATCH - (int64_t)delta - job->bytes) {
            if (!checkpoint_range(w, job, offset, job->bytes, destination, failure)) return false;
            rows--;
            if (rows) { offset += job->source_stride; destination += job->destination_stride; }
            continue;
        }
        int64_t count = 1 + (CHECKPOINT_SCRATCH - (int64_t)delta - job->bytes) / job->source_stride;
        if (count > rows) count = rows;
        size_t payload = (count - 1) * job->source_stride + job->bytes;
        size_t length = (delta + payload + PLE_BLOCK - 1) & ~(size_t)(PLE_BLOCK - 1);
        if (!checkpoint_read(w, job, w->scratch, aligned, length, failure) ||
            !checkpoint_copy(w, (char *)w->scratch + delta, destination, job->bytes, count,
                             job->source_stride, job->destination_stride, job->expand, failure)) return false;
        w->strided_copy_bytes += count * job->bytes;
        rows -= count;
        if (rows) { offset += count * job->source_stride; destination += count * job->destination_stride; }
    }
    return true;
}

static void *checkpoint_worker_main(void *opaque) {
    checkpoint_worker_t *w = opaque;
    checkpoint_executor_t *e = w->executor;
    failure_t initialization = {{0}};
    gpu_ok(cudaSetDevice(e->device), "select GDS checkpoint device", &initialization);
    uint64_t generation = 0;
    pthread_mutex_lock(&e->mutex);
    for (;;) {
        while (!e->stopping && generation == e->generation) pthread_cond_wait(&e->ready, &e->mutex);
        if (e->stopping) break;
        generation = e->generation;
        if (initialization.message[0]) e->failure = initialization;
        while (e->next_job < e->job_count && !e->failure.message[0]) {
            checkpoint_job_t job = e->jobs[e->next_job++];
            pthread_mutex_unlock(&e->mutex);
            failure_t failure = {{0}};
            bool ok = checkpoint_rows(w, &job, &failure);
            pthread_mutex_lock(&e->mutex);
            if (!ok && !e->failure.message[0]) e->failure = failure;
        }
        if (--e->running == 0) pthread_cond_signal(&e->done);
    }
    pthread_mutex_unlock(&e->mutex);
    return NULL;
}

static void checkpoint_release(checkpoint_executor_t *e) {
    if (e->closed) return;
    pthread_mutex_lock(&e->mutex);
    e->stopping = true;
    pthread_cond_broadcast(&e->ready);
    pthread_mutex_unlock(&e->mutex);
    for (int i = 0; i < e->count; i++) pthread_join(e->workers[i].thread, NULL);
    int previous = e->device;
    cudaGetDevice(&previous); cudaSetDevice(e->device);
    for (int i = 0; i < 16; i++) {
        checkpoint_worker_t *w = &e->workers[i];
        if (w->stream) { cudaStreamSynchronize(w->stream); cudaStreamDestroy(w->stream); }
        if (w->registered) cuFileBufDeregister(w->scratch);
        if (w->allocation) cudaFree(w->allocation);
    }
    for (size_t i = 0; i < e->file_count; i++) cuFileHandleDeregister(e->files[i].handle);
    free(e->files);
    cudaSetDevice(previous);
    e->closed = true;
}

static void checkpoint_delete(PyObject *capsule) {
    checkpoint_executor_t *e = PyCapsule_GetPointer(capsule, CHECKPOINT_CAPSULE);
    if (!e) return;
    Py_BEGIN_ALLOW_THREADS
    checkpoint_release(e);
    Py_END_ALLOW_THREADS
    pthread_mutex_destroy(&e->mutex);
    pthread_cond_destroy(&e->ready); pthread_cond_destroy(&e->done);
    free(e);
}

static PyObject *checkpoint_create(PyObject *self, PyObject *args) {
    (void)self;
    int device, workers;
    unsigned long long copy, expand;
    if (!PyArg_ParseTuple(args, "iiKK", &device, &workers, &copy, &expand)) return NULL;
    if (workers < 1 || workers > 16) return PyErr_Format(PyExc_ValueError, "io_threads must be between 1 and 16");
    checkpoint_executor_t *e = calloc(1, sizeof(*e));
    if (!e) return PyErr_NoMemory();
    e->device = device;
    e->copy = (CUfunction)(uintptr_t)copy; e->expand = (CUfunction)(uintptr_t)expand;
    pthread_mutex_init(&e->mutex, NULL);
    pthread_cond_init(&e->ready, NULL); pthread_cond_init(&e->done, NULL);
    failure_t failure = {{0}};
    int previous = device;
    Py_BEGIN_ALLOW_THREADS
    gpu_ok(cudaGetDevice(&previous), "query checkpoint device", &failure);
    if (!failure.message[0]) gpu_ok(cudaSetDevice(device), "select checkpoint device", &failure);
    if (!failure.message[0]) open_driver(&failure, &e->version);
    CUfunction kernels[] = {e->copy, e->expand};
    for (int k = 0; k < 2 && !failure.message[0]; k++) {
        for (unsigned i = 0; i <= 8; i++) {
            size_t offset, size;
            CUresult status = cuFuncGetParamInfo(kernels[k], i, &offset, &size);
            if (i >= 6 && status == CUDA_ERROR_INVALID_VALUE) break;
            if (!driver_ok(status, "query checkpoint copy parameter", &failure)) break;
            if (i == 8 || size != 8 || offset != i * 8) {
                snprintf(failure.message, sizeof(failure.message), "GDS checkpoint copy parameter ABI is unsupported");
                break;
            }
        }
    }
    for (int i = 0; i < workers && !failure.message[0]; i++) {
        checkpoint_worker_t *w = &e->workers[i];
        w->executor = e;
        if (!gpu_ok(cudaMalloc(&w->allocation, CHECKPOINT_SCRATCH + CHECKPOINT_PAGE),
                    "allocate GDS checkpoint scratch", &failure)) break;
        w->scratch = (void *)(((uintptr_t)w->allocation + CHECKPOINT_PAGE - 1) & ~(uintptr_t)(CHECKPOINT_PAGE - 1));
        if (!gds_ok(cuFileBufRegister(w->scratch, CHECKPOINT_SCRATCH, 0),
                    "register GDS checkpoint scratch", &failure)) break;
        w->registered = true;
        if (!gpu_ok(cudaStreamCreateWithFlags(&w->stream, cudaStreamNonBlocking),
                    "create GDS checkpoint stream", &failure)) break;
        int status = pthread_create(&w->thread, NULL, checkpoint_worker_main, w);
        if (status) { snprintf(failure.message, sizeof(failure.message), "GDS pthread_create: %s", strerror(status)); break; }
        e->count++;
    }
    cudaSetDevice(previous);
    Py_END_ALLOW_THREADS
    PyObject *capsule = PyCapsule_New(e, CHECKPOINT_CAPSULE, checkpoint_delete);
    if (!capsule) {
        checkpoint_release(e);
        pthread_mutex_destroy(&e->mutex);
        pthread_cond_destroy(&e->ready); pthread_cond_destroy(&e->done);
        free(e); return NULL;
    }
    if (failure.message[0]) { Py_DECREF(capsule); return PyErr_Format(PyExc_RuntimeError, "%s", failure.message); }
    return capsule;
}

static checkpoint_file_t *checkpoint_file(checkpoint_executor_t *e, int fd, failure_t *failure) {
    struct stat s;
    int flags = fcntl(fd, F_GETFL);
    if (flags < 0 || !(flags & O_DIRECT)) {
        snprintf(failure->message, sizeof(failure->message), "GDS checkpoint reader requires O_DIRECT");
        return NULL;
    }
    if (fstat(fd, &s) || !S_ISREG(s.st_mode)) {
        snprintf(failure->message, sizeof(failure->message), "GDS checkpoint input must be a regular file");
        return NULL;
    }
    for (size_t i = 0; i < e->file_count; i++) if (e->files[i].fd == fd) {
        if (e->files[i].size == s.st_size) return &e->files[i];
        snprintf(failure->message, sizeof(failure->message), "GDS checkpoint file size changed");
        return NULL;
    }
    checkpoint_file_t *files = realloc(e->files, (e->file_count + 1) * sizeof(*files));
    if (!files) { snprintf(failure->message, sizeof(failure->message), "GDS file metadata allocation failed"); return NULL; }
    e->files = files;
    checkpoint_file_t *file = &e->files[e->file_count];
    CUfileDescr_t desc = {.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD};
    desc.handle.fd = fd;
    if (!gds_ok(cuFileHandleRegister(&file->handle, &desc), "register GDS checkpoint file", failure)) return NULL;
    file->fd = fd; file->size = s.st_size; e->file_count++;
    return file;
}

static int checkpoint_destination_order(const void *a, const void *b) {
    uintptr_t x = (uintptr_t)((const checkpoint_job_t *)a)->destination;
    uintptr_t y = (uintptr_t)((const checkpoint_job_t *)b)->destination;
    return (x > y) - (x < y);
}
static int checkpoint_source_order(const void *a, const void *b) {
    const checkpoint_job_t *x = a, *y = b;
    if (x->fd != y->fd) return (x->fd > y->fd) - (x->fd < y->fd);
    return (x->offset > y->offset) - (x->offset < y->offset);
}

static PyObject *checkpoint_execute(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule; Py_buffer records;
    unsigned long long stream;
    if (!PyArg_ParseTuple(args, "Oy*K", &capsule, &records, &stream)) return NULL;
    checkpoint_executor_t *e = PyCapsule_GetPointer(capsule, CHECKPOINT_CAPSULE);
    if (!e) { PyBuffer_Release(&records); return NULL; }
    failure_t failure = {{0}};
    checkpoint_job_t *jobs = NULL;
    size_t count = 0, capacity = 0, descriptors = records.len / 64;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&e->mutex);
    if (e->closed || e->poisoned || e->active) {
        snprintf(failure.message, sizeof(failure.message), "GDS checkpoint executor is closed, failed, or active");
        goto done;
    }
    if (records.len % 64) { snprintf(failure.message, sizeof(failure.message), "invalid read descriptor byte size"); goto done; }
    for (size_t i = 0; i < descriptors; i++) {
        uint64_t r[8]; memcpy(r, (char *)records.buf + i * sizeof(r), sizeof(r));
        uint64_t fd = r[0], offset = r[1], bytes = r[2], pointer = r[3], op = r[4];
        uint64_t rows = r[5], source_stride = r[6], destination_stride = r[7];
        bool expand = op == 1, host_copy = op == 2;
        if (fd > INT_MAX || op > 2 || offset > INT64_MAX || bytes > (uint64_t)INT64_MAX - offset ||
            (expand && (bytes % 2 || bytes > INT64_MAX / 2)) ||
            (host_copy && ((!offset && bytes) || rows != 1)) || !rows || rows > INT64_MAX ||
            source_stride > INT64_MAX || destination_stride > INT64_MAX ||
            (source_stride && rows - 1 > ((uint64_t)INT64_MAX - offset - bytes) / source_stride) ||
            (destination_stride && rows - 1 > ((uint64_t)INT64_MAX - bytes * (1 + expand)) / destination_stride)) {
            snprintf(failure.message, sizeof(failure.message), "invalid read descriptor"); goto done;
        }
        if (rows > 1 && destination_stride < bytes * (1 + expand)) {
            snprintf(failure.message, sizeof(failure.message), "overlapping batch destinations need an explicit dependency"); goto done;
        }
        uint64_t extent = (rows - 1) * destination_stride + bytes * (1 + expand);
        if (!device_range(pointer, extent, e->device)) {
            snprintf(failure.message, sizeof(failure.message), "GDS destination is outside its CUDA device allocation"); goto done;
        }
        checkpoint_file_t *file = NULL;
        if (!host_copy) {
            file = checkpoint_file(e, (int)fd, &failure);
            if (!file) goto done;
            uint64_t source_end = offset + (rows - 1) * source_stride + bytes;
            if (source_end > (uint64_t)file->size) {
                snprintf(failure.message, sizeof(failure.message), "invalid GDS checkpoint file range"); goto done;
            }
        }
        while (rows && bytes) {
            if (count == capacity) {
                size_t next = capacity ? capacity * 2 : 1024;
                if (next > SIZE_MAX / sizeof(*jobs)) { snprintf(failure.message, sizeof(failure.message), "too many GDS descriptors"); goto done; }
                checkpoint_job_t *grown = realloc(jobs, next * sizeof(*jobs));
                if (!grown) { snprintf(failure.message, sizeof(failure.message), "GDS descriptor allocation failed"); goto done; }
                jobs = grown; capacity = next;
            }
            uint64_t chunk = bytes < CHECKPOINT_CHUNK ? bytes : CHECKPOINT_CHUNK;
            uint64_t chunk_rows = 1;
            if (bytes <= CHECKPOINT_CHUNK && destination_stride == bytes * (1 + expand)) {
                uint64_t stride = source_stride > destination_stride ? source_stride : destination_stride;
                chunk_rows = stride ? CHECKPOINT_CHUNK / stride : 1;
                if (!chunk_rows) chunk_rows = 1;
                if (chunk_rows > rows) chunk_rows = rows;
            }
            jobs[count++] = (checkpoint_job_t){.fd = (int)fd, .handle = file ? file->handle : NULL,
                .file_size = file ? file->size : 0, .offset = offset, .bytes = chunk,
                .rows = chunk_rows, .source_stride = source_stride, .destination_stride = destination_stride,
                .destination = (char *)(uintptr_t)pointer, .expand = expand, .host_copy = host_copy};
            if (chunk < bytes) {
                if (rows != 1) { snprintf(failure.message, sizeof(failure.message), "strided rows exceed batch chunk size"); goto done; }
                offset += chunk; bytes -= chunk; pointer += chunk * (1 + expand);
            } else {
                rows -= chunk_rows;
                if (rows) { offset += chunk_rows * source_stride; pointer += chunk_rows * destination_stride; }
            }
        }
    }
    qsort(jobs, count, sizeof(*jobs), checkpoint_destination_order);
    for (size_t i = 1; i < count; i++) {
        checkpoint_job_t *previous = &jobs[i - 1];
        uint64_t end = (uintptr_t)previous->destination + previous->bytes * (1 + previous->expand) +
                       (previous->rows - 1) * previous->destination_stride;
        if (end > (uintptr_t)jobs[i].destination) {
            snprintf(failure.message, sizeof(failure.message), "overlapping batch destinations need an explicit dependency"); goto done;
        }
    }
    qsort(jobs, count, sizeof(*jobs), checkpoint_source_order);
    if (!gpu_ok(cudaStreamSynchronize((cudaStream_t)(uintptr_t)stream), "synchronize GDS checkpoint destinations", &failure)) goto done;
    e->active = true; e->jobs = jobs; e->job_count = count; e->next_job = 0;
    e->running = e->count; e->failure.message[0] = 0;
    double start = seconds();
    e->generation++; pthread_cond_broadcast(&e->ready);
    while (e->running) pthread_cond_wait(&e->done, &e->mutex);
    e->execution_seconds += seconds() - start;
    failure = e->failure; e->jobs = NULL; e->active = false;
    e->batches++; e->descriptors += descriptors;
    if (failure.message[0]) e->poisoned = true;
done:
    pthread_mutex_unlock(&e->mutex);
    free(jobs);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&records);
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    Py_RETURN_NONE;
}

static PyObject *checkpoint_stats(PyObject *self, PyObject *capsule) {
    (void)self;
    checkpoint_executor_t *e = PyCapsule_GetPointer(capsule, CHECKPOINT_CAPSULE);
    if (!e) return NULL;
    checkpoint_worker_t total = {0};
    pthread_mutex_lock(&e->mutex);
    if (e->active) {
        pthread_mutex_unlock(&e->mutex);
        return PyErr_Format(PyExc_RuntimeError, "GDS checkpoint executor is active");
    }
    for (int i = 0; i < e->count; i++) {
        checkpoint_worker_t *w = &e->workers[i];
        total.physical_bytes += w->physical_bytes; total.reads += w->reads;
        total.destination_bytes += w->destination_bytes; total.realigned_bytes += w->realigned_bytes;
        total.strided_copy_bytes += w->strided_copy_bytes; total.device_copy_bytes += w->device_copy_bytes;
        total.expanded_bytes += w->expanded_bytes; total.host_copy_bytes += w->host_copy_bytes;
    }
    PyObject *result = Py_BuildValue("{s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:d,s:i,s:i}",
        "physical_bytes", total.physical_bytes, "reads", total.reads,
        "destination_bytes", total.destination_bytes, "realigned_bytes", total.realigned_bytes,
        "strided_copy_bytes", total.strided_copy_bytes, "device_copy_bytes", total.device_copy_bytes,
        "gds_expanded_bytes", total.expanded_bytes, "metadata_h2d_bytes", total.host_copy_bytes,
        "gpu_scratch_bytes", (uint64_t)e->count * (CHECKPOINT_SCRATCH + CHECKPOINT_PAGE),
        "gds_physical_bytes", total.physical_bytes,
        "batches", e->batches, "descriptors", e->descriptors,
        "execution_seconds", e->execution_seconds, "gds_enabled", 1, "gds_version", e->version);
    pthread_mutex_unlock(&e->mutex);
    return result;
}
