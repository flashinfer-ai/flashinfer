#define _GNU_SOURCE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime_api.h>
#include <cufile.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct { char message[512]; } failure_t;
static void system_error(failure_t *failure, const char *operation) {
    snprintf(failure->message, sizeof(failure->message), "%s: %s", operation, strerror(errno));
}
#include "_row_plan.h"

#define GDS_CAPSULE "b12x.gds_reader"
typedef struct {
    CUfileBatchHandle_t handle;
    size_t first;
    unsigned offset, depth, count;
    bool active;
    cudaEvent_t gathered;
} gds_batch_t;
typedef struct {
    ple_plan_t plan;
    int device, version;
    unsigned depth;
    gds_batch_t batches[2];
    unsigned batch_count;
    CUfunction gather;
    unsigned gather_threads, gather_shared;
    void *weight, *scale;
    CUfileHandle_t *handles;
    size_t handle_count;
    CUfileIOParams_t *params;
    CUfileIOEvents_t *events;
    bool *completed;
    void *allocation, *arena;
    int64_t *host_desc, *device_desc;
    size_t arena_bytes;
    uint64_t status_calls;
    double submit_seconds, status_seconds;
    unsigned registered;
    bool closed, poisoned, unsafe;
    size_t next_job;
    pthread_mutex_t mutex;
} gds_reader_t;

static pthread_mutex_t driver_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool driver_ready;
static void drain_all(gds_reader_t *r);

static bool gds_ok(CUfileError_t status, const char *op, failure_t *failure) {
    if (status.err == CU_FILE_SUCCESS) return true;
    if (!failure->message[0]) snprintf(failure->message, sizeof(failure->message),
        "%s: cuFile error %d, CUDA error %d", op, status.err, status.cu_err);
    return false;
}
static bool gpu_ok(cudaError_t status, const char *op, failure_t *failure) {
    if (status == cudaSuccess) return true;
    if (!failure->message[0]) snprintf(failure->message, sizeof(failure->message),
        "%s: %s", op, cudaGetErrorString(status));
    return false;
}
static bool driver_ok(CUresult status, const char *op, failure_t *failure) {
    if (status == CUDA_SUCCESS) return true;
    const char *message = NULL;
    cuGetErrorString(status, &message);
    if (!failure->message[0]) snprintf(failure->message, sizeof(failure->message),
        "%s: %s", op, message ? message : "CUDA driver failure");
    return false;
}
static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static bool open_driver(failure_t *failure, int *version) {
    pthread_mutex_lock(&driver_mutex);
    bool ok = gds_ok(cuFileGetVersion(version), "cuFileGetVersion", failure);
    if (ok && *version < 1140) {
        snprintf(failure->message, sizeof(failure->message), "GDS disk backend requires cuFile >= 1.14");
        ok = false;
    }
    if (ok && !driver_ready) {
        CUfileError_t status = cuFileDriverOpen();
        if (status.err != CU_FILE_DRIVER_ALREADY_OPEN)
            ok = gds_ok(status, "cuFileDriverOpen", failure);
    }
    CUfileDrvProps_t props = {0};
    if (ok) ok = gds_ok(cuFileDriverGetProperties(&props), "initialize cuFile properties", failure);
    if (ok) driver_ready = true;
    /* The driver is process-global and may be shared with another library. */
    pthread_mutex_unlock(&driver_mutex);
    return ok;
}

static void release(gds_reader_t *r) {
    if (r->closed || r->unsafe) return;
    int previous;
    cudaGetDevice(&previous);
    cudaSetDevice(r->device);
    for (unsigned i = 0; i < r->batch_count; i++) {
        if (r->batches[i].gathered) {
            cudaEventSynchronize(r->batches[i].gathered);
            cudaEventDestroy(r->batches[i].gathered);
        }
        if (r->batches[i].handle) cuFileBatchIODestroy(r->batches[i].handle);
    }
    for (size_t i = 0; i < r->handle_count; i++) cuFileHandleDeregister(r->handles[i]);
    for (unsigned i = 0; i < r->registered; i++)
        cuFileBufDeregister((char *)r->arena + (size_t)i * PLE_READ_MAX);
    if (r->allocation) cudaFree(r->allocation);
    if (r->host_desc) cudaFreeHost(r->host_desc);
    if (r->device_desc) cudaFree(r->device_desc);
    ple_plan_release(&r->plan);
    free(r->handles); free(r->params); free(r->events); free(r->completed);
    r->closed = true;
    cudaSetDevice(previous);
}
static void destroy(PyObject *capsule) {
    gds_reader_t *r = PyCapsule_GetPointer(capsule, GDS_CAPSULE);
    if (!r) return;
    Py_BEGIN_ALLOW_THREADS
    drain_all(r);
    release(r);
    /* Never return DMA targets to an allocator if completion could not be proven. */
    if (!r->unsafe) { pthread_mutex_destroy(&r->mutex); free(r); }
    Py_END_ALLOW_THREADS
}
static gds_reader_t *get_reader(PyObject *capsule) {
    gds_reader_t *r = PyCapsule_GetPointer(capsule, GDS_CAPSULE);
    if (r && (r->closed || r->poisoned)) {
        PyErr_SetString(PyExc_RuntimeError, r->closed ? "GDS reader is closed" :
                        "GDS reader is unusable after a transport failure; create a new reader");
        return NULL;
    }
    return r;
}

static PyObject *create(PyObject *self, PyObject *args) {
    (void)self;
    long long shard, rows, start, end, weight, scale, count;
    int depth, device;
    if (!PyArg_ParseTuple(args, "LLLLLLLii", &shard, &rows, &start, &end, &weight, &scale,
                           &count, &depth, &device)) return NULL;
    if (depth <= 0 || depth > 32768) return PyErr_Format(PyExc_ValueError, "invalid GDS queue depth");
    gds_reader_t *r = calloc(1, sizeof(*r));
    if (!r) return PyErr_NoMemory();
    r->device = device;
    int error = pthread_mutex_init(&r->mutex, NULL);
    if (error) { free(r); return PyErr_Format(PyExc_RuntimeError, "GDS mutex: %s", strerror(error)); }
    error = ple_plan_init(&r->plan, shard, rows, start, end, weight, scale, count);
    failure_t failure = {{0}};
    if (error) snprintf(failure.message, sizeof(failure.message), "invalid GDS geometry or metadata allocation: %s", strerror(error));
    Py_BEGIN_ALLOW_THREADS
    CUfileDrvProps_t props = {0};
    if (!failure.message[0] && gpu_ok(cudaSetDevice(device), "select GDS device", &failure) &&
        open_driver(&failure, &r->version) &&
        gds_ok(cuFileDriverGetProperties(&props), "cuFileDriverGetProperties", &failure)) {
        r->depth = (unsigned)depth < props.max_batch_io_size ? (unsigned)depth : props.max_batch_io_size;
        if (!r->depth) snprintf(failure.message, sizeof(failure.message), "cuFile reports no batch capacity");
        r->arena_bytes = (size_t)r->depth * PLE_READ_MAX;
        r->params = calloc(r->depth, sizeof(*r->params));
        r->events = calloc(r->depth, sizeof(*r->events));
        r->completed = calloc(r->depth, sizeof(*r->completed));
        if (!r->params || !r->events || !r->completed)
            snprintf(failure.message, sizeof(failure.message), "GDS batch allocation failed");
        if (!failure.message[0] && gpu_ok(cudaMalloc(&r->allocation, r->arena_bytes + PLE_READ_MAX), "allocate GDS arena", &failure)) {
            r->arena = (void *)(((uintptr_t)r->allocation + PLE_READ_MAX - 1) & ~(uintptr_t)(PLE_READ_MAX - 1));
            /* Batch entries use separately registered, nonoverlapping GPU pages. */
            while (r->registered < r->depth && gds_ok(cuFileBufRegister(
                    (char *)r->arena + (size_t)r->registered * PLE_READ_MAX, PLE_READ_MAX, 0),
                    "cuFileBufRegister", &failure)) r->registered++;
        }
        size_t metadata = r->plan.capacity * 4 * sizeof(int64_t);
        if (!failure.message[0]) gpu_ok(cudaHostAlloc((void **)&r->host_desc, metadata, cudaHostAllocDefault), "allocate GDS host descriptors", &failure);
        if (!failure.message[0]) gpu_ok(cudaMalloc((void **)&r->device_desc, metadata), "allocate GDS device descriptors", &failure);
        r->batch_count = r->depth > 1 ? 2 : 1;
        unsigned offset = 0;
        for (unsigned i = 0; i < r->batch_count && !failure.message[0]; i++) {
            gds_batch_t *b = &r->batches[i];
            b->offset = offset;
            b->depth = (r->depth + r->batch_count - 1 - i) / r->batch_count;
            offset += b->depth;
            gds_ok(cuFileBatchIOSetUp(&b->handle, b->depth), "cuFileBatchIOSetUp", &failure);
            if (!failure.message[0]) gpu_ok(cudaEventCreateWithFlags(&b->gathered, cudaEventDisableTiming),
                                             "create GDS gather event", &failure);
        }
    }
    Py_END_ALLOW_THREADS
    if (failure.message[0]) {
        release(r); pthread_mutex_destroy(&r->mutex); free(r);
        return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    }
    PyObject *capsule = PyCapsule_New(r, GDS_CAPSULE, destroy);
    if (!capsule) { release(r); pthread_mutex_destroy(&r->mutex); free(r); }
    return capsule;
}
static int register_files(ple_plan_t *plan, void *owner) {
    gds_reader_t *r = owner;
    CUfileHandle_t *handles = realloc(r->handles, plan->file_count * sizeof(*handles));
    if (!handles) return -ENOMEM;
    r->handles = handles;
    while (r->handle_count < plan->file_count) {
        /* Source validation uses O_NONBLOCK to reject FIFOs without waiting. */
        int fd = plan->files[r->handle_count].fd;
        int flags = fcntl(fd, F_GETFL);
        if (flags < 0 || fcntl(fd, F_SETFL, flags & ~O_NONBLOCK) < 0) {
            system_error(&plan->failure, "clear nonblocking source flag");
            r->poisoned = true;
            return -EIO;
        }
        CUfileDescr_t desc = {.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD};
        desc.handle.fd = fd;
        if (!gds_ok(cuFileHandleRegister(&r->handles[r->handle_count], &desc),
                    "cuFileHandleRegister (direct GDS required)", &plan->failure)) {
            r->poisoned = true;
            return -EIO;
        }
        r->handle_count++;
    }
    return 0;
}
static PyObject *add(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule; long long shard, offset; const char *path; int scale;
    if (!PyArg_ParseTuple(args, "OLsLp", &capsule, &shard, &path, &offset, &scale)) return NULL;
    gds_reader_t *r = get_reader(capsule);
    if (!r) return NULL;
    int error; failure_t failure;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&r->mutex);
    r->plan.failure.message[0] = 0;
    error = ple_add_source(&r->plan, shard, path, offset, scale != 0, register_files, r);
    failure = r->plan.failure;
    pthread_mutex_unlock(&r->mutex);
    Py_END_ALLOW_THREADS
    if (error) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message[0] ? failure.message : strerror(error));
    Py_RETURN_NONE;
}
static PyObject *begin(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule, *ids_object; Py_ssize_t count; unsigned long long stream_pointer;
    if (!PyArg_ParseTuple(args, "OOnK", &capsule, &ids_object, &count, &stream_pointer)) return NULL;
    gds_reader_t *r = get_reader(capsule);
    if (!r) return NULL;
    if (count < 0 || (size_t)count > r->plan.max_lookups)
        return PyErr_Format(PyExc_ValueError, "GDS lookup count exceeds batch capacity");
    for (unsigned i = 0; i < r->batch_count; i++)
        if (r->batches[i].active) return PyErr_Format(PyExc_RuntimeError, "GDS batch is still active");
    Py_buffer ids;
    if (PyObject_GetBuffer(ids_object, &ids, PyBUF_CONTIG_RO) < 0) return NULL;
    if ((size_t)ids.len < (size_t)count * 8) { PyBuffer_Release(&ids); return PyErr_Format(PyExc_ValueError, "GDS IDs do not cover count"); }
    bool ok; failure_t failure;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&r->mutex);
    ple_plan_t *p = &r->plan;
    p->failure.message[0] = 0;
    p->lookups = count; p->requested_bytes = p->read_bytes = p->read_calls = 0;
    p->unique_blocks = p->coalesced_reads = p->submit_calls = 0;
    p->execution_seconds = 0;
    r->status_calls = 0;
    r->submit_seconds = r->status_seconds = 0;
    double start = seconds();
    ok = ple_plan(p, ids.buf, NULL, NULL, count);
    p->planning_seconds = seconds() - start;
    r->next_job = 0;
    start = seconds();
    if (ok) {
        for (size_t i = 0; i < p->job_count; i++) {
            ple_job_t *job = &p->jobs[i];
            for (size_t j = job->begin; j < job->end; j++) {
                ple_fragment_t *f = &p->fragments[j];
                int64_t *d = r->host_desc + j * 4;
                d[0] = (int64_t)(i % r->depth) * PLE_READ_MAX + f->offset - job->offset;
                d[1] = (int64_t)f->destination; d[2] = f->length; d[3] = f->scale;
            }
        }
        if (p->fragment_count) ok = gpu_ok(cudaMemcpyAsync(r->device_desc, r->host_desc,
            p->fragment_count * 4 * sizeof(int64_t), cudaMemcpyHostToDevice,
            (cudaStream_t)(uintptr_t)stream_pointer), "copy GDS gather descriptors", &p->failure);
    }
    p->execution_seconds += seconds() - start;
    failure = p->failure;
    pthread_mutex_unlock(&r->mutex);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&ids);
    if (!ok) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    return PyLong_FromSize_t(r->plan.job_count);
}

static bool drain(gds_reader_t *r, gds_batch_t *b, bool cancelling) {
    if (!b->active) return true;
    unsigned left = b->count;
    double deadline = seconds() + 30;
    if (cancelling) cuFileBatchIOCancel(b->handle);
    while (left) {
        unsigned nr = r->depth;
        struct timespec timeout = {.tv_sec = 0, .tv_nsec = 100000000};
        double wait_start = seconds();
        r->status_calls++;
        CUfileError_t status = cuFileBatchIOGetStatus(b->handle, left, &nr, r->events, &timeout);
        r->status_seconds += seconds() - wait_start;
        if (status.err != CU_FILE_SUCCESS || seconds() > deadline) {
            gds_ok(status, "cuFileBatchIOGetStatus", &r->plan.failure);
            if (!r->plan.failure.message[0]) snprintf(r->plan.failure.message, sizeof(r->plan.failure.message), "GDS completion deadline exceeded");
            r->poisoned = true;
            r->unsafe = true;
            cuFileBatchIOCancel(b->handle);
            b->active = false;
            return false;
        }
        for (unsigned i = 0; i < nr; i++) {
            CUfileIOEvents_t *event = &r->events[i];
            size_t slot = (uintptr_t)event->cookie;
            if (slot >= b->count || r->completed[b->offset + slot]) {
                snprintf(r->plan.failure.message, sizeof(r->plan.failure.message), "invalid or duplicate GDS completion cookie");
                r->poisoned = r->unsafe = true;
                cuFileBatchIOCancel(b->handle);
                b->active = false;
                return false;
            }
            if (event->status == CUFILE_PENDING || event->status == CUFILE_WAITING) continue;
            r->completed[b->offset + slot] = true;
            left--;
            ple_job_t *job = &r->plan.jobs[b->first + slot];
            r->plan.read_calls++;
            if (event->status == CUFILE_COMPLETE && (ssize_t)event->ret >= 0) r->plan.read_bytes += event->ret;
            if (!r->plan.failure.message[0] && (event->status != CUFILE_COMPLETE || event->ret != job->expected))
                snprintf(r->plan.failure.message, sizeof(r->plan.failure.message),
                         "short or failed GDS read at offset %lld: expected %u bytes, received %lld (status %d)",
                         (long long)job->offset, job->expected, (long long)event->ret, event->status);
        }
    }
    b->active = false;
    return true;
}
static void drain_all(gds_reader_t *r) {
    /* Submitted batches retain their completion records until they are drained. */
    for (unsigned i = 0; i < r->batch_count; i++) drain(r, &r->batches[i], false);
}
static bool submit_batch(gds_reader_t *r, gds_batch_t *b) {
    ple_plan_t *p = &r->plan;
    b->first = r->next_job;
    unsigned count = p->job_count - r->next_job < b->depth ? (unsigned)(p->job_count - r->next_job) : b->depth;
    b->count = count;
    for (unsigned i = 0; i < count; i++) {
        ple_job_t *job = &p->jobs[r->next_job + i];
        CUfileIOParams_t *param = &r->params[b->offset + i];
        *param = (CUfileIOParams_t){.mode = CUFILE_BATCH, .fh = r->handles[job->file],
            .opcode = CUFILE_READ, .cookie = (void *)(uintptr_t)i};
        param->u.batch.devPtr_base = (char *)r->arena + (size_t)(b->offset + i) * PLE_READ_MAX;
        param->u.batch.devPtr_offset = 0;
        param->u.batch.file_offset = job->offset;
        param->u.batch.size = job->length;
        r->completed[b->offset + i] = false;
    }
    if (count && gpu_ok(cudaSetDevice(r->device), "select GDS device", &p->failure)) {
        p->submit_calls++;
        double submit_start = seconds();
        b->active = true;
        bool submitted = gds_ok(cuFileBatchIOSubmit(b->handle, count, r->params + b->offset, 0), "cuFileBatchIOSubmit", &p->failure);
        r->submit_seconds += seconds() - submit_start;
        if (!submitted) {
            r->poisoned = true;
            drain(r, b, true);
            drain_all(r);
        }
    }
    r->next_job += count;
    return !p->failure.message[0];
}
static PyObject *configure_gather(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    unsigned long long function, weight, scale;
    unsigned threads, shared;
    if (!PyArg_ParseTuple(args, "OKIIKK", &capsule, &function, &threads, &shared, &weight, &scale)) return NULL;
    gds_reader_t *r = get_reader(capsule);
    if (!r) return NULL;
    if (threads != 128 || shared || !function || !weight || !scale)
        return PyErr_Format(PyExc_ValueError, "invalid GDS gather launch contract");
    CUfunction kernel = (CUfunction)(uintptr_t)function;
    failure_t failure = {{0}};
    /* Four data pointers, followed by up to two unused Triton scratch pointers. */
    unsigned parameters = 0;
    for (; parameters <= 6; parameters++) {
        size_t offset, size;
        CUresult status = cuFuncGetParamInfo(kernel, parameters, &offset, &size);
        if (parameters >= 4 && status == CUDA_ERROR_INVALID_VALUE) break;
        if (!driver_ok(status, "query GDS gather parameter", &failure)) break;
        if (parameters == 6 || size != sizeof(void *) || offset != parameters * sizeof(void *)) {
            snprintf(failure.message, sizeof(failure.message), "GDS gather kernel parameter ABI is unsupported");
            break;
        }
    }
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    r->gather = kernel; r->gather_threads = threads; r->gather_shared = shared;
    r->weight = (void *)(uintptr_t)weight; r->scale = (void *)(uintptr_t)scale;
    Py_RETURN_NONE;
}
static PyObject *execute(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule; unsigned long long stream_pointer;
    if (!PyArg_ParseTuple(args, "OK", &capsule, &stream_pointer)) return NULL;
    gds_reader_t *r = get_reader(capsule);
    if (!r) return NULL;
    if (!r->gather) return PyErr_Format(PyExc_RuntimeError, "GDS gather is not configured");
    failure_t failure;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&r->mutex);
    double start = seconds();
    ple_plan_t *p = &r->plan;
    cudaStream_t stream = (cudaStream_t)(uintptr_t)stream_pointer;
    unsigned pending = 0;
    for (unsigned i = 0; i < r->batch_count && !p->failure.message[0]; i++) {
        if (r->next_job < p->job_count && submit_batch(r, &r->batches[i])) pending++;
    }
    unsigned lane = 0;
    while (pending && !p->failure.message[0]) {
        gds_batch_t *b = &r->batches[lane];
        if (!drain(r, b, false) || p->failure.message[0]) break;
        pending--;
        size_t begin = p->jobs[b->first].begin;
        size_t end = p->jobs[b->first + b->count - 1].end;
        void *descriptors = r->device_desc + begin * 4;
        void *scratch = NULL;
        void *params[] = {&r->arena, &descriptors, &r->weight, &r->scale, &scratch, &scratch};
        if (!driver_ok(cuLaunchKernel(r->gather, (unsigned)(end - begin), 1, 1,
                r->gather_threads, 1, 1, r->gather_shared, (CUstream)stream, params, NULL),
                "launch GDS byte gather", &p->failure) ||
            !gpu_ok(cudaEventRecord(b->gathered, stream), "record GDS gather completion", &p->failure)) break;
        if (r->next_job < p->job_count) {
            if (!gpu_ok(cudaEventSynchronize(b->gathered), "wait for GDS gather", &p->failure) ||
                !submit_batch(r, b)) break;
            pending++;
        }
        lane = (lane + 1) % r->batch_count;
    }
    if (p->failure.message[0]) drain_all(r);
    p->execution_seconds += seconds() - start;
    failure = p->failure;
    pthread_mutex_unlock(&r->mutex);
    Py_END_ALLOW_THREADS
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s%s", failure.message,
        r->unsafe ? "; DMA completion unproven; registered storage retained until process exit" : "");
    Py_RETURN_NONE;
}
static PyObject *info(PyObject *self, PyObject *capsule) {
    (void)self;
    gds_reader_t *r = get_reader(capsule);
    if (!r) return NULL;
    return Py_BuildValue("KKnni", (unsigned long long)(uintptr_t)r->arena,
        (unsigned long long)(uintptr_t)r->device_desc, (Py_ssize_t)r->arena_bytes,
        (Py_ssize_t)r->plan.capacity, r->batch_count);
}
static PyObject *stats(PyObject *self, PyObject *capsule) {
    (void)self;
    gds_reader_t *r = get_reader(capsule);
    if (!r) return NULL;
    ple_plan_t *p = &r->plan;
    return Py_BuildValue("{s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:d,s:d,s:i,s:i,s:K,s:d,s:d}",
        "lookups", p->lookups, "requested_bytes", p->requested_bytes, "read_bytes", p->read_bytes,
        "read_calls", p->read_calls, "unique_blocks", p->unique_blocks, "coalesced_reads", p->coalesced_reads,
        "submit_calls", p->submit_calls, "staging_bytes", (uint64_t)(r->arena_bytes + PLE_READ_MAX),
        "descriptor_bytes", (uint64_t)(p->capacity * 4 * sizeof(int64_t)),
        "metadata_bytes", (uint64_t)(sizeof(*r) + p->capacity * (sizeof(ple_fragment_t) + sizeof(ple_job_t)) +
            r->depth * (sizeof(CUfileIOParams_t) + sizeof(CUfileIOEvents_t) + sizeof(bool)) +
            p->source_count * sizeof(ple_source_t) + p->file_count * (sizeof(ple_file_t) + sizeof(CUfileHandle_t))),
        "planning_seconds", p->planning_seconds, "execution_seconds", p->execution_seconds + p->planning_seconds,
        "gds_version", r->version, "gds_enabled", 1, "status_calls", r->status_calls,
        "submit_seconds", r->submit_seconds, "status_seconds", r->status_seconds);
}
static PyObject *close_reader(PyObject *self, PyObject *capsule) {
    (void)self;
    gds_reader_t *r = PyCapsule_GetPointer(capsule, GDS_CAPSULE);
    if (!r) return NULL;
    if (r->unsafe) return PyErr_Format(PyExc_RuntimeError, "cannot release GDS buffers while DMA completion is unproven");
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&r->mutex);
    drain_all(r);
    release(r);
    pthread_mutex_unlock(&r->mutex);
    Py_END_ALLOW_THREADS
    if (r->unsafe) return PyErr_Format(PyExc_RuntimeError, "cannot release GDS buffers while DMA completion is unproven");
    Py_RETURN_NONE;
}
#ifdef B12X_GDS_STATS
static PyObject *start_stats(PyObject *self, PyObject *args) {
    (void)self;
    int level = 1;
    if (!PyArg_ParseTuple(args, "|i", &level)) return NULL;
    failure_t failure = {{0}};
    if (!gds_ok(cuFileSetStatsLevel(level), "cuFileSetStatsLevel", &failure) ||
        !gds_ok(cuFileStatsStart(), "cuFileStatsStart", &failure))
        return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    Py_RETURN_NONE;
}
static PyObject *synchronous_transport_stats(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    CUfileStatsLevel3_t s = {0}; failure_t failure = {{0}};
    if (!gds_ok(cuFileGetStatsL3(&s), "cuFileGetStatsL3", &failure))
        return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    uint64_t nvfs = 0, p2p = 0, posix = 0, errors = 0;
    if (s.num_gpus > sizeof(s.per_gpu_stats) / sizeof(s.per_gpu_stats[0]))
        return PyErr_Format(PyExc_RuntimeError, "cuFile reported too many GPU statistics entries");
    for (unsigned i = 0; i < s.num_gpus; i++) {
        nvfs += s.per_gpu_stats[i].n_nvfs_reads;
        p2p += s.per_gpu_stats[i].n_p2p_reads;
        posix += s.per_gpu_stats[i].n_posix_reads;
        errors += s.per_gpu_stats[i].n_reads_err;
    }
    return Py_BuildValue("{s:K,s:K,s:K,s:K,s:K}",
        "nvfs_reads", nvfs, "p2p_reads", p2p, "posix_reads", posix,
        "read_errors", errors, "read_bytes", s.detailed.basic.read_bytes);
}
static PyObject *transport_stats(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    CUfileStatsLevel1_t s = {0}; failure_t failure = {{0}};
    if (!gds_ok(cuFileGetStatsL1(&s), "cuFileGetStatsL1", &failure))
        return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    return Py_BuildValue("{s:K,s:K,s:K,s:K,s:K,s:K,s:K}",
        "nvfs_ops", s.batch_nvfs_submit_ops.ok, "p2p_ops", s.batch_p2p_submit_ops.ok,
        "posix_ops", s.batch_posix_enqueued_ops.ok, "aio_ops", s.batch_aio_submit_ops.ok,
        "iouring_ops", s.batch_iouring_submit_ops.ok, "read_bytes", s.batch_read_bytes,
        "read_errors", s.batch_processed_ops.err);
}
#endif
#include "_gds_checkpoint.c"
#include "_gds_owner.c"

static PyMethodDef methods[] = {
    {"owner_create", owner_create, METH_VARARGS, NULL},
    {"owner_execute", owner_execute, METH_VARARGS, NULL},
    {"owner_export", owner_export, METH_VARARGS, NULL},
    {"owner_import", owner_import, METH_VARARGS, NULL},
    {"owner_unmap", owner_unmap, METH_O, NULL},
    {"owner_close", owner_close, METH_O, NULL},
    {"checkpoint_create", checkpoint_create, METH_VARARGS, NULL},
    {"checkpoint_execute", checkpoint_execute, METH_VARARGS, NULL},
    {"checkpoint_stats", checkpoint_stats, METH_O, NULL},
    {"create", create, METH_VARARGS, NULL}, {"add", add, METH_VARARGS, NULL},
    {"begin", begin, METH_VARARGS, NULL}, {"execute", execute, METH_VARARGS, NULL},
    {"configure_gather", configure_gather, METH_VARARGS, NULL},
    {"info", info, METH_O, NULL}, {"stats", stats, METH_O, NULL},
    {"close", close_reader, METH_O, NULL},
#ifdef B12X_GDS_STATS
    {"start_stats", start_stats, METH_VARARGS, NULL},
    {"transport_stats", transport_stats, METH_NOARGS, NULL},
    {"synchronous_transport_stats", synchronous_transport_stats, METH_NOARGS, NULL},
#endif
    {NULL, NULL, 0, NULL},
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, .m_name = "_b12x_gds_reader", .m_size = -1, .m_methods = methods};
PyMODINIT_FUNC PyInit__b12x_gds_reader(void) { return PyModule_Create(&module); }
