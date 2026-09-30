/* Included by _gds_reader.c. One bounded reader owns each shared source span. */
#define OWNER_CAPSULE "b12x.gds_owner"
#define OWNER_SLOTS 4
#define OWNER_SLOT_BYTES 4653056u
#define OWNER_ALIGNMENT 65536u

typedef struct {
    uint64_t source, destination, bytes, rows, source_stride, destination_stride, expand;
} owner_fragment_t;
typedef struct {
    CUfileHandle_t handle;
    int64_t file_size;
    uint64_t offset, bytes, first, count;
} owner_chunk_t;
typedef struct {
    void *base;
    uint64_t bytes;
} owner_import_t;
typedef struct {
    unsigned state, pending;
    uint64_t generation;
    owner_chunk_t *chunk;
    cudaStream_t stream;
    cudaEvent_t event;
} owner_slot_t;
typedef struct owner_executor owner_executor_t;
typedef struct {
    owner_executor_t *owner;
    pthread_t thread;
    unsigned index;
    uint64_t reads, bytes;
    double read_seconds;
} owner_worker_t;
struct owner_executor {
    checkpoint_executor_t files;
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    bool stopping, active, poisoned, closed, registered, unsafe;
    unsigned workers, running, parts;
    void *allocation, *arena;
    owner_slot_t slots[OWNER_SLOTS];
    owner_worker_t threads[16];
    owner_import_t *imports;
    size_t import_count;
    failure_t failure;
};

static void *owner_worker(void *argument) {
    owner_worker_t *worker = argument;
    owner_executor_t *e = worker->owner;
    unsigned slot_index = worker->index / e->parts, part = worker->index % e->parts;
    owner_slot_t *slot = &e->slots[slot_index];
    failure_t failure = {{0}};
    gpu_ok(cudaSetDevice(e->files.device), "set shared read worker device", &failure);
    pthread_mutex_lock(&e->mutex);
    if (failure.message[0] && !e->failure.message[0]) e->failure = failure;
    uint64_t generation = 0;
    while (!e->stopping) {
        while (!e->stopping && slot->generation == generation)
            pthread_cond_wait(&e->ready, &e->mutex);
        if (e->stopping) break;
        generation = slot->generation;
        owner_chunk_t *chunk = slot->chunk;
        uint64_t step = ((chunk->bytes + e->parts - 1) / e->parts + 4095) & ~(uint64_t)4095;
        uint64_t delta = part * step;
        pthread_mutex_unlock(&e->mutex);
        failure.message[0] = 0;
        if (delta < chunk->bytes) {
            uint64_t bytes = chunk->bytes - delta < step ? chunk->bytes - delta : step;
            int64_t offset = chunk->offset + delta;
            int64_t expected = chunk->file_size - offset;
            if (expected > (int64_t)bytes) expected = bytes;
            double started = seconds();
            ssize_t received = cuFileRead(chunk->handle, e->arena, bytes, offset,
                                         slot_index * OWNER_SLOT_BYTES + delta);
            worker->read_seconds += seconds() - started;
            worker->reads++;
            if (received > 0) worker->bytes += received;
            if (received != expected) snprintf(failure.message, sizeof(failure.message),
                "shared cuFileRead at %lld: expected %lld, received %lld",
                (long long)offset, (long long)expected, (long long)received);
        }
        pthread_mutex_lock(&e->mutex);
        if (failure.message[0] && !e->failure.message[0]) e->failure = failure;
        if (!--slot->pending) slot->state = 2;
        pthread_cond_broadcast(&e->ready);
    }
    pthread_mutex_unlock(&e->mutex);
    return NULL;
}

static bool owner_unmap_all(owner_executor_t *e, failure_t *failure) {
    if (e->unsafe) {
        snprintf(failure->message, sizeof(failure->message), "shared scatter completion is unproven; retain IPC owners until process exit");
        return false;
    }
    bool ok = true;
    for (size_t i = 0; i < e->import_count; i++) {
        if (e->imports[i].base && !gpu_ok(cudaIpcCloseMemHandle(e->imports[i].base),
                "close shared weight IPC mapping", failure)) ok = false;
        else e->imports[i].base = NULL;
    }
    if (ok) { free(e->imports); e->imports = NULL; e->import_count = 0; }
    return ok;
}

static bool owner_release(owner_executor_t *e, failure_t *failure) {
    if (e->closed) return true;
    pthread_mutex_lock(&e->mutex);
    e->stopping = true;
    pthread_cond_broadcast(&e->ready);
    pthread_mutex_unlock(&e->mutex);
    for (unsigned i = 0; i < e->running; i++) pthread_join(e->threads[i].thread, NULL);
    e->running = 0;
    bool ok = gpu_ok(cudaSetDevice(e->files.device), "set shared reader cleanup device", failure);
    for (unsigned i = 0; i < OWNER_SLOTS; i++) {
        if (e->slots[i].stream && !gpu_ok(cudaStreamSynchronize(e->slots[i].stream),
                "drain shared weight scatter", failure)) ok = false;
    }
    /* Retain registrations and mappings if CUDA cannot prove work has drained. */
    if (!ok) { e->poisoned = true; e->unsafe = true; return false; }
    ok = owner_unmap_all(e, failure);
    if (e->registered) {
        if (gds_ok(cuFileBufDeregister(e->arena), "deregister shared read arena", failure))
            e->registered = false;
        else ok = false;
    }
    if (!e->registered && e->allocation) {
        if (gpu_ok(cudaFree(e->allocation), "free shared read arena", failure)) e->allocation = NULL;
        else ok = false;
    }
    for (unsigned i = 0; i < OWNER_SLOTS; i++) {
        if (e->slots[i].event) {
            if (!gpu_ok(cudaEventDestroy(e->slots[i].event), "destroy shared read event", failure)) ok = false;
            e->slots[i].event = NULL;
        }
        if (e->slots[i].stream) {
            if (!gpu_ok(cudaStreamDestroy(e->slots[i].stream), "destroy shared read stream", failure)) ok = false;
            e->slots[i].stream = NULL;
        }
    }
    for (size_t i = 0; i < e->files.file_count; i++) cuFileHandleDeregister(e->files.files[i].handle);
    free(e->files.files); e->files.files = NULL; e->files.file_count = 0;
    e->closed = ok;
    return ok;
}

static void owner_destruct(PyObject *capsule) {
    owner_executor_t *e = PyCapsule_GetPointer(capsule, OWNER_CAPSULE);
    if (!e) { PyErr_Clear(); return; }
    failure_t failure = {{0}};
    bool safe;
    Py_BEGIN_ALLOW_THREADS
    safe = owner_release(e, &failure);
    Py_END_ALLOW_THREADS
    if (!safe) {
        fprintf(stderr, "b12x shared reader retained unsafe CUDA resources: %s\n", failure.message);
        return;
    }
    pthread_cond_destroy(&e->ready); pthread_mutex_destroy(&e->mutex);
    free(e);
}

static PyObject *owner_create(PyObject *self, PyObject *args) {
    (void)self;
    int device; unsigned threads;
    unsigned long long copy, expand;
    if (!PyArg_ParseTuple(args, "iIKK", &device, &threads, &copy, &expand)) return NULL;
    if (threads < OWNER_SLOTS || threads > 16 || threads % OWNER_SLOTS || !copy || !expand)
        return PyErr_Format(PyExc_ValueError, "shared reader needs 4, 8, 12 or 16 workers and copy kernels");
    owner_executor_t *e = calloc(1, sizeof(*e));
    if (!e) return PyErr_NoMemory();
    e->files.device = device;
    e->files.copy = (CUfunction)(uintptr_t)copy; e->files.expand = (CUfunction)(uintptr_t)expand;
    e->workers = threads; e->parts = threads / OWNER_SLOTS;
    pthread_mutex_init(&e->mutex, NULL); pthread_cond_init(&e->ready, NULL);
    PyObject *capsule = PyCapsule_New(e, OWNER_CAPSULE, owner_destruct);
    if (!capsule) {
        pthread_cond_destroy(&e->ready); pthread_mutex_destroy(&e->mutex);
        free(e); return NULL;
    }
    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    if (!gpu_ok(cudaSetDevice(device), "set shared reader device", &failure) ||
        !open_driver(&failure, &e->files.version)) goto done;
    size_t capacity = OWNER_SLOTS * OWNER_SLOT_BYTES;
    if (!gpu_ok(cudaMalloc(&e->allocation, capacity + OWNER_ALIGNMENT), "allocate shared read arena", &failure)) goto done;
    e->arena = (void *)(((uintptr_t)e->allocation + OWNER_ALIGNMENT - 1) & ~(uintptr_t)(OWNER_ALIGNMENT - 1));
    if (!gds_ok(cuFileBufRegister(e->arena, capacity, 0), "register shared read arena", &failure)) goto done;
    e->registered = true;
    for (unsigned i = 0; i < OWNER_SLOTS; i++) {
        if (!gpu_ok(cudaStreamCreateWithFlags(&e->slots[i].stream, cudaStreamNonBlocking),
                    "create shared read stream", &failure) ||
            !gpu_ok(cudaEventCreateWithFlags(&e->slots[i].event, cudaEventDisableTiming),
                    "create shared read event", &failure)) goto done;
    }
    for (unsigned i = 0; i < threads; i++) {
        e->threads[i].owner = e; e->threads[i].index = i;
        int status = pthread_create(&e->threads[i].thread, NULL, owner_worker, &e->threads[i]);
        if (status) { snprintf(failure.message, sizeof(failure.message), "create shared read worker: %s", strerror(status)); goto done; }
        e->running++;
    }
done:;
    Py_END_ALLOW_THREADS
    if (failure.message[0]) { Py_DECREF(capsule); return PyErr_Format(PyExc_RuntimeError, "%s", failure.message); }
    return capsule;
}

static bool owner_destination(owner_executor_t *e, uint64_t pointer, uint64_t bytes) {
    if (device_range(pointer, bytes, e->files.device)) return true;
    for (size_t i = 0; i < e->import_count; i++) {
        uintptr_t base = (uintptr_t)e->imports[i].base;
        if (base && pointer >= base && pointer - base <= e->imports[i].bytes &&
            bytes <= e->imports[i].bytes - (pointer - base)) return true;
    }
    return false;
}

static PyObject *owner_execute(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule; Py_buffer chunks, fragments;
    unsigned long long stream;
    int validate_only = 0;
    if (!PyArg_ParseTuple(args, "Oy*y*K|p", &capsule, &chunks, &fragments, &stream, &validate_only)) return NULL;
    owner_executor_t *e = PyCapsule_GetPointer(capsule, OWNER_CAPSULE);
    if (!e) { PyBuffer_Release(&chunks); PyBuffer_Release(&fragments); return NULL; }
    failure_t failure = {{0}};
    owner_chunk_t *jobs = NULL;
    owner_fragment_t *copies = NULL;
    size_t count = chunks.len / 40, fragment_count = fragments.len / sizeof(*copies);
    uint64_t total_reads = 0, total_bytes = 0;
    uint64_t destination_bytes = 0, peer_bytes = 0;
    double read_seconds = 0, idle_seconds = 0;
    double started = seconds();
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&e->mutex);
    if (e->closed || e->poisoned || e->active || e->stopping) {
        snprintf(failure.message, sizeof(failure.message), "shared reader is closed, failed, or active");
        goto unlock;
    }
    if (e->failure.message[0]) { failure = e->failure; e->poisoned = true; goto unlock; }
    e->active = true;
    if (chunks.len % 40 || fragments.len % sizeof(*copies) || count > SIZE_MAX / sizeof(*jobs)) {
        snprintf(failure.message, sizeof(failure.message), "invalid shared read descriptor sizes"); goto done;
    }
    jobs = calloc(count ? count : 1, sizeof(*jobs));
    copies = malloc(fragments.len ? (size_t)fragments.len : 1);
    if (!jobs || !copies) { snprintf(failure.message, sizeof(failure.message), "shared read metadata allocation failed"); goto done; }
    memcpy(copies, fragments.buf, fragments.len);
    for (size_t i = 0; i < count; i++) {
        uint64_t r[5]; memcpy(r, (char *)chunks.buf + i * sizeof(r), sizeof(r));
        if (r[0] > INT_MAX || r[1] > INT64_MAX || !r[2] || r[2] > OWNER_SLOT_BYTES ||
            r[1] % 4096 || r[2] % 4096 || r[2] > (uint64_t)INT64_MAX - r[1] ||
            r[3] > fragment_count || r[4] > fragment_count - r[3]) {
            snprintf(failure.message, sizeof(failure.message), "invalid shared read chunk"); goto done;
        }
        checkpoint_file_t *file = checkpoint_file(&e->files, r[0], &failure);
        if (!file) goto done;
        if (r[1] >= (uint64_t)file->size || r[1] + r[2] - 1 > ((uint64_t)file->size - 1) / 4096 * 4096 + 4095) {
            snprintf(failure.message, sizeof(failure.message), "shared read exceeds aligned file extent"); goto done;
        }
        jobs[i] = (owner_chunk_t){file->handle, file->size, r[1], r[2], r[3], r[4]};
        for (size_t j = r[3]; j < r[3] + r[4]; j++) {
            owner_fragment_t *f = &copies[j];
            if (!f->bytes || !f->rows || f->expand > 1 || f->source > r[2] ||
                f->bytes > r[2] - f->source ||
                (f->source_stride && f->rows - 1 > (r[2] - f->source - f->bytes) / f->source_stride) ||
                f->bytes > (uint64_t)INT64_MAX / (1 + f->expand) ||
                f->destination_stride > INT64_MAX ||
                (f->destination_stride && f->rows - 1 > ((uint64_t)INT64_MAX - f->bytes * (1 + f->expand)) / f->destination_stride) ||
                (f->rows > 1 && f->destination_stride < f->bytes * (1 + f->expand)) ||
                (f->expand && (f->bytes % 2 || f->source_stride % 2 || f->destination_stride % 4)) ||
                f->rows > (uint64_t)INT_MAX * 1024 / f->bytes ||
                r[1] + f->source + (f->rows - 1) * f->source_stride + f->bytes > (uint64_t)file->size ||
                !owner_destination(e, f->destination, (f->rows - 1) * f->destination_stride + f->bytes * (1 + f->expand))) {
                snprintf(failure.message, sizeof(failure.message), "invalid shared scatter fragment or invalid destination"); goto done;
            }
            uint64_t written = f->rows * f->bytes * (1 + f->expand);
            destination_bytes += written;
            if (!device_range(f->destination,
                    (f->rows - 1) * f->destination_stride + f->bytes * (1 + f->expand), e->files.device))
                peer_bytes += written;
        }
    }
    if (validate_only) goto done;
    if (!gpu_ok(cudaSetDevice(e->files.device), "set shared execute device", &failure) ||
        !gpu_ok(cudaStreamSynchronize((cudaStream_t)(uintptr_t)stream), "fence shared destination producers", &failure)) goto done;
    for (unsigned i = 0; i < e->workers; i++) {
        total_reads -= e->threads[i].reads; total_bytes -= e->threads[i].bytes;
        read_seconds -= e->threads[i].read_seconds;
    }
    size_t next = 0, completed = 0;
    while (completed < count && !failure.message[0]) {
        bool progress = false;
        if (e->failure.message[0]) { failure = e->failure; break; }
        for (unsigned i = 0; i < OWNER_SLOTS && !failure.message[0]; i++) {
            owner_slot_t *slot = &e->slots[i];
            if (slot->state == 2) {
                owner_chunk_t *chunk = slot->chunk;
                for (size_t j = chunk->first; j < chunk->first + chunk->count; j++) {
                    owner_fragment_t *f = &copies[j];
                    void *source = (char *)e->arena + i * OWNER_SLOT_BYTES + f->source;
                    void *destination = (void *)(uintptr_t)f->destination, *scratch = NULL;
                    void *parameters[] = {&source, &destination, &f->bytes, &f->rows,
                        &f->source_stride, &f->destination_stride, &scratch, &scratch};
                    uint64_t elements = f->rows * f->bytes / (1 + f->expand);
                    if (!driver_ok(cuLaunchKernel(f->expand ? e->files.expand : e->files.copy,
                        (unsigned)((elements + 1023) / 1024), 1, 1, 128, 1, 1, 0,
                        (CUstream)slot->stream, parameters, NULL), "launch shared weight scatter", &failure)) break;
                }
                if (!failure.message[0]) gpu_ok(cudaEventRecord(slot->event, slot->stream), "record shared scatter completion", &failure);
                slot->state = 3; progress = true;
            }
            if (slot->state == 3 && !failure.message[0]) {
                cudaError_t status = cudaEventQuery(slot->event);
                if (status == cudaSuccess) { slot->state = 0; slot->chunk = NULL; completed++; progress = true; }
                else if (status != cudaErrorNotReady) gpu_ok(status, "query shared scatter completion", &failure);
            }
            if (!slot->state && next < count && !failure.message[0]) {
                slot->chunk = &jobs[next++]; slot->pending = e->parts;
                slot->state = 1; slot->generation++;
                pthread_cond_broadcast(&e->ready); progress = true;
            }
        }
        if (!progress && !failure.message[0]) {
            struct timespec deadline;
            clock_gettime(CLOCK_REALTIME, &deadline);
            deadline.tv_nsec += 10000;
            if (deadline.tv_nsec >= 1000000000) { deadline.tv_sec++; deadline.tv_nsec -= 1000000000; }
            double idle_started = seconds();
            pthread_cond_timedwait(&e->ready, &e->mutex, &deadline);
            idle_seconds += seconds() - idle_started;
        } else {
            pthread_mutex_unlock(&e->mutex);
            pthread_mutex_lock(&e->mutex);
        }
    }
    for (;;) {
        bool pending = false;
        for (unsigned i = 0; i < OWNER_SLOTS; i++) pending |= e->slots[i].pending != 0;
        if (!pending) break;
        pthread_cond_wait(&e->ready, &e->mutex);
    }
    for (unsigned i = 0; i < OWNER_SLOTS; i++) {
        if (!gpu_ok(cudaStreamSynchronize(e->slots[i].stream), "drain shared weight scatter", &failure)) e->unsafe = true;
        e->slots[i].state = 0; e->slots[i].chunk = NULL;
    }
    for (unsigned i = 0; i < e->workers; i++) {
        total_reads += e->threads[i].reads; total_bytes += e->threads[i].bytes;
        read_seconds += e->threads[i].read_seconds;
    }
done:
    if (failure.message[0]) e->poisoned = true;
    e->active = false;
unlock:
    pthread_mutex_unlock(&e->mutex);
    free(jobs); free(copies);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&chunks); PyBuffer_Release(&fragments);
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    return Py_BuildValue("{s:K,s:K,s:K,s:K,s:K,s:K,s:d,s:d,s:d,s:i}", "reads", total_reads, "physical_bytes", total_bytes,
                        "chunks", (uint64_t)count, "fragments", (uint64_t)fragment_count,
                        "destination_bytes", destination_bytes, "peer_bytes", peer_bytes,
                        "read_task_seconds_sum", read_seconds, "idle_seconds", idle_seconds,
                        "execution_seconds", seconds() - started, "gds_version", e->files.version);
}

static PyObject *owner_export(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule; unsigned long long pointer, extent;
    if (!PyArg_ParseTuple(args, "OKK", &capsule, &pointer, &extent)) return NULL;
    owner_executor_t *e = PyCapsule_GetPointer(capsule, OWNER_CAPSULE);
    if (!e) return NULL;
    failure_t failure = {{0}};
    CUdeviceptr base = 0; size_t bytes = 0;
    cudaIpcMemHandle_t handle;
    pthread_mutex_lock(&e->mutex);
    if (e->active || e->closed || e->poisoned || !extent ||
        !device_range(pointer, extent, e->files.device))
        snprintf(failure.message, sizeof(failure.message), "shared IPC export needs an idle reader and valid CUDA device range");
    if (!failure.message[0] &&
        gpu_ok(cudaSetDevice(e->files.device), "set shared export device", &failure) &&
        driver_ok(cuMemGetAddressRange(&base, &bytes, pointer), "query shared allocation extent", &failure)) {
        if (!device_range(base, bytes, e->files.device))
            snprintf(failure.message, sizeof(failure.message), "IPC allocation extent is outside its CUDA device allocation");
        else gpu_ok(cudaIpcGetMemHandle(&handle, (void *)(uintptr_t)base), "export shared weight allocation", &failure);
    }
    pthread_mutex_unlock(&e->mutex);
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    return Py_BuildValue("(KKy#)", (uint64_t)base, (uint64_t)bytes, &handle, (Py_ssize_t)sizeof(handle));
}

static PyObject *owner_import(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule; Py_buffer data; unsigned long long bytes;
    if (!PyArg_ParseTuple(args, "Oy*K", &capsule, &data, &bytes)) return NULL;
    owner_executor_t *e = PyCapsule_GetPointer(capsule, OWNER_CAPSULE);
    if (!e) { PyBuffer_Release(&data); return NULL; }
    if (data.len != sizeof(cudaIpcMemHandle_t) || !bytes || bytes > INT64_MAX) {
        PyBuffer_Release(&data); return PyErr_Format(PyExc_ValueError, "invalid shared CUDA IPC handle or extent");
    }
    cudaIpcMemHandle_t handle; memcpy(&handle, data.buf, sizeof(handle));
    PyBuffer_Release(&data);
    failure_t failure = {{0}};
    void *base = NULL;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&e->mutex);
    if (e->active || e->closed || e->poisoned)
        snprintf(failure.message, sizeof(failure.message), "shared IPC import needs an idle reader");
    if (!failure.message[0]) {
        owner_import_t *imports = realloc(e->imports, (e->import_count + 1) * sizeof(*imports));
        if (!imports) snprintf(failure.message, sizeof(failure.message), "shared IPC metadata allocation failed");
        else {
            e->imports = imports;
            if (gpu_ok(cudaSetDevice(e->files.device), "set shared import device", &failure) &&
                gpu_ok(cudaIpcOpenMemHandle(&base, handle, cudaIpcMemLazyEnablePeerAccess), "import shared weight allocation", &failure))
                e->imports[e->import_count++] = (owner_import_t){base, bytes};
        }
    }
    pthread_mutex_unlock(&e->mutex);
    Py_END_ALLOW_THREADS
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    return PyLong_FromUnsignedLongLong((uintptr_t)base);
}

static PyObject *owner_unmap(PyObject *self, PyObject *capsule) {
    (void)self;
    owner_executor_t *e = PyCapsule_GetPointer(capsule, OWNER_CAPSULE);
    if (!e) return NULL;
    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&e->mutex);
    if (e->active) snprintf(failure.message, sizeof(failure.message), "shared reader is active");
    else if (gpu_ok(cudaSetDevice(e->files.device), "set shared unmap device", &failure)) owner_unmap_all(e, &failure);
    pthread_mutex_unlock(&e->mutex);
    Py_END_ALLOW_THREADS
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    Py_RETURN_NONE;
}

static PyObject *owner_close(PyObject *self, PyObject *capsule) {
    (void)self;
    owner_executor_t *e = PyCapsule_GetPointer(capsule, OWNER_CAPSULE);
    if (!e) return NULL;
    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&e->mutex);
    bool active = e->active;
    pthread_mutex_unlock(&e->mutex);
    if (active) snprintf(failure.message, sizeof(failure.message), "shared reader is active");
    else owner_release(e, &failure);
    Py_END_ALLOW_THREADS
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    Py_RETURN_NONE;
}
