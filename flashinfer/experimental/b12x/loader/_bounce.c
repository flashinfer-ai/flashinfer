/* Included by _batch.c. Fixed read slots stay owned until their CUDA copy finishes. */
#define BOUNCE_SLOTS 16
#define BOUNCE_SLOT_BYTES (1 << 20)
#define BOUNCE_BYTES (BOUNCE_SLOTS * BOUNCE_SLOT_BYTES)
#ifdef B12X_HAVE_LIBURING
#include <liburing.h>
#endif

typedef struct {
    read_job_t job;
    size_t delta, required;
} bounce_slot_t;

typedef struct {
    char *buffers;
    int device;
    cudaStream_t stream;
    bool registered;
    failure_t copy_failure;
    bounce_slot_t slots[BOUNCE_SLOTS];
    pthread_mutex_t mutex;
    pthread_cond_t ready, available, drained;
    pthread_t copier;
    bool copier_started, stopping, poisoned;
    unsigned free_slots[BOUNCE_SLOTS], free_count;
    unsigned ready_slots[BOUNCE_SLOTS], head, tail, ready_count, copying;
    unsigned depth, max_inflight;
    uint64_t reads, physical_bytes, bounced_bytes, strided_bytes, submits;
#ifdef B12X_HAVE_LIBURING
    struct io_uring ring;
    bool ring_ready;
#endif
} bounce_ring_t;

static void bounce_release(bounce_ring_t *ring) {
    if (!ring) return;
    if (ring->copier_started) {
        pthread_mutex_lock(&ring->mutex);
        ring->stopping = true;
        pthread_cond_signal(&ring->ready);
        pthread_mutex_unlock(&ring->mutex);
        pthread_join(ring->copier, NULL);
    }
#ifdef B12X_HAVE_LIBURING
    if (ring->ring_ready) io_uring_queue_exit(&ring->ring);
#endif
    int previous = ring->device;
    cudaGetDevice(&previous);
    cudaSetDevice(ring->device);
    if (ring->stream) { cudaStreamSynchronize(ring->stream); cudaStreamDestroy(ring->stream); }
    if (ring->registered) cudaHostUnregister(ring->buffers);
    free(ring->buffers);
    cudaSetDevice(previous);
    pthread_cond_destroy(&ring->ready);
    pthread_cond_destroy(&ring->available);
    pthread_cond_destroy(&ring->drained);
    pthread_mutex_destroy(&ring->mutex);
    free(ring);
}

#ifdef B12X_HAVE_LIBURING
static void expand_bf16_inplace(char *destination, int64_t bytes) {
    for (int64_t i = bytes / 2; i > 0;) {
        --i;
        uint16_t bf16;
        memcpy(&bf16, destination + 2 * i, sizeof(bf16));
        uint32_t fp32 = (uint32_t)bf16 << 16;
        memcpy(destination + 4 * i, &fp32, sizeof(fp32));
    }
}

static void *bounce_copy_worker(void *opaque) {
    bounce_ring_t *ring = opaque;
    failure_t failure = {{0}};
    cuda_ok(cudaSetDevice(ring->device), "select ring copy device", &failure);
    pthread_mutex_lock(&ring->mutex);
    for (;;) {
        while (!ring->ready_count && !ring->stopping)
            pthread_cond_wait(&ring->ready, &ring->mutex);
        if (!ring->ready_count) break;
        unsigned index = ring->ready_slots[ring->head++ % BOUNCE_SLOTS];
        ring->ready_count--;
        bounce_slot_t slot = ring->slots[index];
        pthread_mutex_unlock(&ring->mutex);
        char *buffer = ring->buffers + (size_t)index * BOUNCE_SLOT_BYTES;
        char *source = buffer + slot.delta;
        size_t width = slot.job.bytes;
        size_t pitch = slot.job.rows > 1 ? (size_t)slot.job.source_stride : width;
        if (slot.job.expand_bf16) {
            for (int64_t row = 0; row < slot.job.rows; row++)
                memmove(buffer + row * width, source + row * pitch, width);
            expand_bf16_inplace(buffer, slot.job.rows * width);
            source = buffer;
            width *= 2;
            pitch = width;
        }
        if (!failure.message[0])
            cuda_ok(cudaMemcpy2DAsync(slot.job.destination,
                slot.job.rows > 1 ? (size_t)slot.job.destination_stride : width,
                source, pitch, width, slot.job.rows, cudaMemcpyHostToDevice,
                ring->stream), "copy checkpoint ring to CUDA weights", &failure);
        /* Reads into other slots remain in flight while this slot is on the GPU. */
        cuda_ok(cudaStreamSynchronize(ring->stream), "complete checkpoint ring copy", &failure);
        pthread_mutex_lock(&ring->mutex);
        if (failure.message[0]) ring->copy_failure = failure;
        ring->bounced_bytes += slot.job.rows * slot.job.bytes;
        if (slot.job.rows > 1) ring->strided_bytes += slot.job.rows * slot.job.bytes;
        ring->free_slots[ring->free_count++] = index;
        ring->copying--;
        pthread_cond_signal(&ring->available);
        if (!ring->copying) pthread_cond_signal(&ring->drained);
    }
    pthread_mutex_unlock(&ring->mutex);
    return NULL;
}
#endif

static bounce_ring_t *bounce_create(int device, unsigned depth, failure_t *failure) {
#ifndef B12X_HAVE_LIBURING
    (void)device; (void)depth;
    snprintf(failure->message, sizeof(failure->message),
             "io_uring bounce support is unavailable: install liburing development headers and pkg-config");
    return NULL;
#else
    bounce_ring_t *ring = calloc(1, sizeof(*ring));
    if (!ring) { snprintf(failure->message, sizeof(failure->message), "bounce ring allocation failed"); return NULL; }
    pthread_mutex_init(&ring->mutex, NULL);
    pthread_cond_init(&ring->ready, NULL);
    pthread_cond_init(&ring->available, NULL);
    pthread_cond_init(&ring->drained, NULL);
    ring->device = device;
    ring->depth = depth;
    int error = posix_memalign((void **)&ring->buffers, IO_ALIGNMENT, BOUNCE_BYTES);
    if (error) goto failed;
    if (!cuda_ok(cudaSetDevice(device), "select ring device", failure) ||
        !cuda_ok(cudaHostRegister(ring->buffers, BOUNCE_BYTES, cudaHostRegisterDefault),
                 "pin checkpoint ring for CUDA copies", failure)) goto failed;
    ring->registered = true;
    if (!cuda_ok(cudaStreamCreateWithFlags(&ring->stream, cudaStreamNonBlocking),
                 "create checkpoint copy stream", failure)) goto failed;
    error = io_uring_queue_init(BOUNCE_SLOTS, &ring->ring, 0);
    if (error < 0) { error = -error; goto failed; }
    ring->ring_ready = true;
    struct iovec buffers[BOUNCE_SLOTS];
    for (unsigned i = 0; i < BOUNCE_SLOTS; i++) {
        buffers[i] = (struct iovec){ring->buffers + (size_t)i * BOUNCE_SLOT_BYTES, BOUNCE_SLOT_BYTES};
        ring->free_slots[ring->free_count++] = i;
    }
    error = io_uring_register_buffers(&ring->ring, buffers, BOUNCE_SLOTS);
    if (error < 0) { error = -error; goto failed; }
    error = pthread_create(&ring->copier, NULL, bounce_copy_worker, ring);
    if (error) goto failed;
    ring->copier_started = true;
    return ring;
failed:
    if (!failure->message[0]) snprintf(failure->message, sizeof(failure->message),
             "io_uring bounce initialization failed: %s; check liburing, kernel/container policy and locked-memory limits",
             strerror(error));
    bounce_release(ring);
    return NULL;
#endif
}

static bool bounce_execute(bounce_ring_t *ring, read_job_t *jobs, size_t count,
                            failure_t *failure) {
#ifndef B12X_HAVE_LIBURING
    (void)ring; (void)jobs; (void)count;
    snprintf(failure->message, sizeof(failure->message), "io_uring bounce support is unavailable");
    return false;
#else
    if (ring->poisoned) {
        snprintf(failure->message, sizeof(failure->message), "io_uring bounce executor is unusable after a submission/completion failure");
        return false;
    }
    for (size_t i = 0; i < count; i++) {
        read_job_t *job = &jobs[i];
        if (!job->host_copy && !validate_direct_range(job->fd, job->offset,
                (job->rows - 1) * job->source_stride + job->bytes, failure)) return false;
    }
    size_t next = 0;
    int64_t row = 0, within = 0;
    unsigned outstanding = 0;
    while ((!failure->message[0] && next < count) || outstanding) {
        unsigned pending = 0;
        while (!failure->message[0] && next < count && outstanding + pending < ring->depth) {
            read_job_t *job = &jobs[next];
            if (job->host_copy) {
                cuda_ok(cudaMemcpyAsync(job->destination, (void *)(uintptr_t)job->offset,
                    job->bytes, cudaMemcpyHostToDevice, ring->stream),
                    "copy checkpoint metadata", failure);
                next++;
                continue;
            }
            pthread_mutex_lock(&ring->mutex);
            if (ring->copy_failure.message[0]) {
                *failure = ring->copy_failure;
                pthread_mutex_unlock(&ring->mutex);
                break;
            }
            if (!ring->free_count) { pthread_mutex_unlock(&ring->mutex); break; }
            unsigned index = ring->free_slots[--ring->free_count];
            pthread_mutex_unlock(&ring->mutex);
            bounce_slot_t *slot = &ring->slots[index];
            slot->job = *job;
            int64_t offset = job->offset + row * job->source_stride + within;
            int64_t aligned = offset & ~(int64_t)(IO_ALIGNMENT - 1);
            slot->delta = offset - aligned;
            slot->job.destination += row * job->destination_stride + within * (1 + job->expand_bf16);
            slot->job.bytes = job->bytes - within;
            if (slot->job.bytes > BOUNCE_SLOT_BYTES - (int64_t)slot->delta)
                slot->job.bytes = BOUNCE_SLOT_BYTES - slot->delta;
            if (job->expand_bf16) {
                if (slot->job.bytes > BOUNCE_SLOT_BYTES / 2)
                    slot->job.bytes = BOUNCE_SLOT_BYTES / 2;
                slot->job.bytes &= ~(int64_t)1;
            }
            slot->job.rows = 1;
            if (!within && job->bytes < IO_ALIGNMENT && job->source_stride >= job->bytes) {
                slot->job.rows += (BOUNCE_SLOT_BYTES - (int64_t)slot->delta - job->bytes) / job->source_stride;
                if (slot->job.rows > job->rows - row) slot->job.rows = job->rows - row;
            }
            if (job->expand_bf16 && slot->job.rows > BOUNCE_SLOT_BYTES / (2 * slot->job.bytes))
                slot->job.rows = BOUNCE_SLOT_BYTES / (2 * slot->job.bytes);
            slot->required = slot->delta + (slot->job.rows - 1) * job->source_stride + slot->job.bytes;
            unsigned length = (slot->required + IO_ALIGNMENT - 1) & ~(size_t)(IO_ALIGNMENT - 1);
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring->ring);
            if (!sqe) {
                snprintf(failure->message, sizeof(failure->message), "io_uring bounce SQ capacity exhausted");
                ring->poisoned = true;
                break;
            }
            io_uring_prep_read_fixed(sqe, job->fd,
                ring->buffers + (size_t)index * BOUNCE_SLOT_BYTES, length, aligned, index);
            io_uring_sqe_set_data64(sqe, index);
            pending++;
            within += slot->job.bytes;
            if (within == job->bytes) { row += slot->job.rows; within = 0; }
            if (row == job->rows) { next++; row = 0; }
        }
        while (pending && !failure->message[0]) {
            ring->submits++;
            int submitted = io_uring_submit(&ring->ring);
            if (submitted == -EINTR) continue;
            if (submitted <= 0) {
                snprintf(failure->message, sizeof(failure->message), "io_uring bounce submit: %s", strerror(submitted < 0 ? -submitted : EIO));
                ring->poisoned = true;
                break;
            }
            outstanding += submitted;
            pending -= submitted;
            if (outstanding > ring->max_inflight) ring->max_inflight = outstanding;
        }
        if (outstanding) {
            struct io_uring_cqe *cqe = NULL;
            int status = io_uring_wait_cqe(&ring->ring, &cqe);
            if (status < 0) {
                if (status != -EINTR && status != -EAGAIN) {
                    if (!failure->message[0]) snprintf(failure->message, sizeof(failure->message), "io_uring bounce wait: %s", strerror(-status));
                    ring->poisoned = true;
                }
                continue;
            }
            unsigned index = (unsigned)io_uring_cqe_get_data64(cqe);
            status = cqe->res;
            io_uring_cqe_seen(&ring->ring, cqe);
            outstanding--;
            ring->reads++;
            if (status > 0) ring->physical_bytes += status;
            bool success = status >= 0 && (size_t)status >= ring->slots[index].required;
            if (!success && !failure->message[0])
                snprintf(failure->message, sizeof(failure->message), "io_uring bounce read: %s", status < 0 ? strerror(-status) : "short O_DIRECT read");
            pthread_mutex_lock(&ring->mutex);
            if (success) {
                ring->ready_slots[ring->tail++ % BOUNCE_SLOTS] = index;
                ring->ready_count++;
                ring->copying++;
                pthread_cond_signal(&ring->ready);
            } else ring->free_slots[ring->free_count++] = index;
            pthread_mutex_unlock(&ring->mutex);
        } else if (!failure->message[0] && next < count) {
            pthread_mutex_lock(&ring->mutex);
            while (!ring->free_count) pthread_cond_wait(&ring->available, &ring->mutex);
            pthread_mutex_unlock(&ring->mutex);
        }
    }
    pthread_mutex_lock(&ring->mutex);
    while (ring->copying) pthread_cond_wait(&ring->drained, &ring->mutex);
    if (ring->copy_failure.message[0]) {
        if (!failure->message[0]) *failure = ring->copy_failure;
        ring->poisoned = true;
    }
    pthread_mutex_unlock(&ring->mutex);
    if (!cuda_ok(cudaStreamSynchronize(ring->stream), "complete checkpoint metadata copies", failure))
        ring->poisoned = true;
    if (ring->poisoned) {
        /* Drain submitted reads above; never submit SQEs left by a failed submit. */
        io_uring_queue_exit(&ring->ring);
        ring->ring_ready = false;
    }
    return !failure->message[0];
#endif
}
