/* Descriptor destinations are ordinary CUDA tensors retained by Python. */
#include <limits.h>
#include <time.h>
#include "_cuda_range.h"

#define BATCH_CHUNK_BYTES (64 << 20)
typedef struct {
    int fd;
    int64_t offset, bytes;
    char *destination;
    bool expand_bf16;
    bool host_copy;
    int64_t rows, source_stride, destination_stride;
} read_job_t;

#include "_bounce.c"

typedef struct {
    pthread_mutex_t mutex;
    bounce_ring_t *bounce;
    int device;
    uint64_t batches, descriptors;
    double execution_seconds;
} batch_executor_t;

static void delete_batch(PyObject *capsule) {
    batch_executor_t *executor = PyCapsule_GetPointer(capsule, "b12x.batch_executor");
    if (!executor) return;
    bounce_release(executor->bounce);
    pthread_mutex_destroy(&executor->mutex);
    free(executor);
}

static PyObject *py_batch_executor(PyObject *self, PyObject *args) {
    (void)self;
    int device, workers;
    if (!PyArg_ParseTuple(args, "ii", &device, &workers)) return NULL;
    if (workers < 1 || workers > 16)
        return PyErr_Format(PyExc_ValueError, "io_threads must be between 1 and 16");
    batch_executor_t *executor = calloc(1, sizeof(*executor));
    if (!executor) return PyErr_NoMemory();
    executor->device = device;
    pthread_mutex_init(&executor->mutex, NULL);
    failure_t failure = {{0}};
    executor->bounce = bounce_create(device, (unsigned)workers, &failure);
    if (!executor->bounce) {
        pthread_mutex_destroy(&executor->mutex);
        free(executor);
        return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    }
    PyObject *capsule = PyCapsule_New(executor, "b12x.batch_executor", delete_batch);
    if (!capsule) {
        bounce_release(executor->bounce);
        pthread_mutex_destroy(&executor->mutex);
        free(executor);
    }
    return capsule;
}

static int job_destination_order(const void *a, const void *b) {
    uintptr_t x = (uintptr_t)((const read_job_t *)a)->destination;
    uintptr_t y = (uintptr_t)((const read_job_t *)b)->destination;
    return (x > y) - (x < y);
}

static int job_source_order(const void *a, const void *b) {
    const read_job_t *x = a, *y = b;
    if (x->fd != y->fd) return (x->fd > y->fd) - (x->fd < y->fd);
    return (x->offset > y->offset) - (x->offset < y->offset);
}

static PyObject *py_batch_execute(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    Py_buffer records;
    unsigned long long stream;
    if (!PyArg_ParseTuple(args, "Oy*K", &capsule, &records, &stream)) return NULL;
    batch_executor_t *executor = PyCapsule_GetPointer(capsule, "b12x.batch_executor");
    if (!executor) { PyBuffer_Release(&records); return NULL; }
    failure_t failure = {{0}};
    read_job_t *jobs = NULL;
    size_t count = 0, capacity = 0;
    if (records.len % (8 * sizeof(uint64_t))) {
        snprintf(failure.message, sizeof(failure.message), "invalid read descriptor byte size");
        goto done;
    }
    size_t descriptors = records.len / (8 * sizeof(uint64_t));
    for (size_t i = 0; i < descriptors; i++) {
        uint64_t record[8];
        memcpy(record, (char *)records.buf + i * sizeof(record), sizeof(record));
        uint64_t fd = record[0], offset = record[1], bytes = record[2];
        uint64_t pointer = record[3], transform = record[4];
        uint64_t rows = record[5], source_stride = record[6], destination_stride = record[7];
        bool expand_bf16 = transform == 1;
        if (fd > INT_MAX || transform > 2 || offset > INT64_MAX ||
            bytes > (uint64_t)INT64_MAX - offset ||
            (expand_bf16 && (bytes % 2 || bytes > INT64_MAX / 2)) ||
            (transform == 2 && ((!offset && bytes) || rows != 1)) || !rows || rows > INT64_MAX ||
            source_stride > INT64_MAX || destination_stride > INT64_MAX ||
            (source_stride && rows - 1 > ((uint64_t)INT64_MAX - offset - bytes) / source_stride) ||
            (destination_stride && rows - 1 > ((uint64_t)INT64_MAX - bytes * (1 + expand_bf16)) / destination_stride)) {
            snprintf(failure.message, sizeof(failure.message), "invalid read descriptor");
            goto done;
        }
        if (rows > 1 && destination_stride < bytes * (1 + expand_bf16)) {
            snprintf(failure.message, sizeof(failure.message), "overlapping batch destinations need an explicit dependency");
            goto done;
        }
        uint64_t extent = (rows - 1) * destination_stride + bytes * (1 + expand_bf16);
        if (!device_range(pointer, extent, executor->device)) {
            snprintf(failure.message, sizeof(failure.message), "batch destination exceeds CUDA device allocation");
            goto done;
        }
        char *destination = (char *)(uintptr_t)pointer;
        while (rows && bytes) {
            if (count == capacity) {
                size_t next = capacity ? capacity * 2 : 1024;
                read_job_t *grown = realloc(jobs, next * sizeof(*jobs));
                if (!grown) { snprintf(failure.message, sizeof(failure.message), "read descriptor allocation failed"); goto done; }
                jobs = grown;
                capacity = next;
            }
            int64_t chunk = bytes > BATCH_CHUNK_BYTES ? BATCH_CHUNK_BYTES : (int64_t)bytes;
            uint64_t chunk_rows = 1;
            if (bytes <= BATCH_CHUNK_BYTES && destination_stride == bytes * (1 + expand_bf16)) {
                uint64_t stride = source_stride > destination_stride ? source_stride : destination_stride;
                chunk_rows = stride ? BATCH_CHUNK_BYTES / stride : 1;
                if (!chunk_rows) chunk_rows = 1;
                if (chunk_rows > rows) chunk_rows = rows;
            }
            jobs[count++] = (read_job_t){(int)fd, (int64_t)offset, chunk, destination,
                expand_bf16, transform == 2, (int64_t)chunk_rows,
                (int64_t)source_stride, (int64_t)destination_stride};
            if (chunk < (int64_t)bytes) {
                /* Large contiguous rows can still be divided into independent jobs. */
                if (rows != 1) {
                    snprintf(failure.message, sizeof(failure.message), "strided rows exceed batch chunk size");
                    goto done;
                }
                offset += chunk;
                bytes -= chunk;
                destination += chunk * (1 + expand_bf16);
            } else {
                rows -= chunk_rows;
                if (rows) {
                    offset += chunk_rows * source_stride;
                    destination += chunk_rows * destination_stride;
                }
            }
        }
    }
    qsort(jobs, count, sizeof(*jobs), job_destination_order);
    for (size_t i = 1; i < count; i++) {
        read_job_t *previous = &jobs[i - 1];
        if ((uintptr_t)previous->destination + previous->bytes * (1 + previous->expand_bf16) +
            (previous->rows - 1) * previous->destination_stride >
            (uintptr_t)jobs[i].destination) {
            snprintf(failure.message, sizeof(failure.message), "overlapping batch destinations need an explicit dependency");
            goto done;
        }
    }
    qsort(jobs, count, sizeof(*jobs), job_source_order);
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&executor->mutex);
    if (cuda_ok(cudaSetDevice(executor->device), "select batch device", &failure) &&
        cuda_ok(cudaStreamSynchronize((cudaStream_t)(uintptr_t)stream),
                "synchronize batch destinations", &failure)) {
        struct timespec start, stop;
        clock_gettime(CLOCK_MONOTONIC, &start);
        bounce_execute(executor->bounce, jobs, count, &failure);
        clock_gettime(CLOCK_MONOTONIC, &stop);
        executor->execution_seconds += stop.tv_sec - start.tv_sec +
                                       (stop.tv_nsec - start.tv_nsec) * 1e-9;
        executor->batches++;
        executor->descriptors += descriptors;
    }
    pthread_mutex_unlock(&executor->mutex);
    Py_END_ALLOW_THREADS
done:
    free(jobs);
    PyBuffer_Release(&records);
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    Py_RETURN_NONE;
}

static PyObject *py_batch_stats(PyObject *self, PyObject *capsule) {
    (void)self;
    batch_executor_t *executor = PyCapsule_GetPointer(capsule, "b12x.batch_executor");
    if (!executor) return NULL;
    pthread_mutex_lock(&executor->mutex);
    bounce_ring_t *ring = executor->bounce;
    PyObject *result = Py_BuildValue("{s:K,s:K,s:K,s:i,s:K,s:K,s:d,s:i,s:K,s:K,s:I}",
        "physical_bytes", (unsigned long long)ring->physical_bytes,
        "strided_copy_bytes", (unsigned long long)ring->strided_bytes,
        "reads", (unsigned long long)ring->reads,
        "scratch_bytes", BOUNCE_BYTES,
        "batches", (unsigned long long)executor->batches,
        "descriptors", (unsigned long long)executor->descriptors,
        "execution_seconds", executor->execution_seconds,
        "bounce_buffer_bytes", BOUNCE_BYTES,
        "bounced_bytes", (unsigned long long)ring->bounced_bytes,
        "io_uring_submits", (unsigned long long)ring->submits,
        "max_inflight_reads", ring->max_inflight);
    pthread_mutex_unlock(&executor->mutex);
    return result;
}
