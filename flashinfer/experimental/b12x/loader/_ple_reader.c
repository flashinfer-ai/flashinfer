/* Included by _storage.c. File payloads are never mapped or retained between runs. */
#include <fcntl.h>
#include <time.h>
#ifdef B12X_HAVE_LIBURING
#include <liburing.h>
#include <sys/resource.h>
#endif
/* Older liburing/kernel headers may not spell the task-run bits yet; the
 * values are the UAPI constants and the ladder below drops them on any kernel
 * that rejects the setup request. */
#ifndef IORING_SETUP_COOP_TASKRUN
#define IORING_SETUP_COOP_TASKRUN (1U << 8)
#endif
#ifndef IORING_SETUP_TASKRUN_FLAG
#define IORING_SETUP_TASKRUN_FLAG (1U << 9)
#endif

#include "_row_plan.h"
#define PLE_CAPSULE "b12x.ple_reader"
/* miss_io.md B1: the shipped 64 slots leave the pipe ~1/6 full (derived mean
 * in-flight ~11 at 249k IOPS), so this is the floor the reader raises a
 * caller's queue_depth to; larger explicit requests are honoured as-is. */
#define PLE_DEFAULT_DEPTH 256u
/* A ring needs at least two slots to overlap a submission with a completion,
 * and it is the depth the clamp loop stops halving at. */
#define PLE_MIN_SLOTS 2u

typedef struct ple_reader ple_reader_t;
typedef struct {
  char* buffer;
  size_t job;
} ple_slot_t;

struct ple_reader {
  ple_plan_t plan;
  bool poisoned;
  pthread_mutex_t api_mutex;
  unsigned slots_count;
  unsigned depth_requested; /* queue_depth as passed, before the default/clamp */
  unsigned ring_flags;      /* IORING_SETUP_* the ring actually came up with */
  bool ring_flag_fallback;  /* the top rung of the flag ladder was rejected */
  ple_slot_t* slots;
  char* buffers;
  char *weights, *scales;
#ifdef B12X_HAVE_LIBURING
  struct io_uring ring;
  bool ring_ready;
  bool files_registered;
  int* registered_fds;
  struct iovec* iovecs;
  unsigned* free_slots;
#endif
};

#ifdef B12X_HAVE_LIBURING
static void ple_error(ple_reader_t* reader, const char* operation, int error) {
  if (!reader->plan.failure.message[0])
    snprintf(reader->plan.failure.message, sizeof(reader->plan.failure.message), "%s: %s",
             operation, strerror(error));
}

static bool ple_scatter(ple_reader_t* reader, size_t index, char* buffer, int result) {
  ple_job_t* job = &reader->plan.jobs[index];
  reader->plan.read_calls++;
  if (result > 0) reader->plan.read_bytes += (unsigned)result;
  if (result < 0) {
    ple_error(reader, "io_uring READ_FIXED", -result);
    return false;
  }
  if ((unsigned)result != job->expected) {
    if (!reader->plan.failure.message[0])
      snprintf(reader->plan.failure.message, sizeof(reader->plan.failure.message),
               "short PLE read at offset %lld: expected %u bytes, received %d; source changed or "
               "was truncated",
               (long long)job->offset, job->expected, result);
    return false;
  }
  for (size_t i = job->begin; i < job->end; i++) {
    ple_fragment_t* fragment = &reader->plan.fragments[i];
    memcpy((fragment->scale ? reader->scales : reader->weights) + fragment->destination,
           buffer + (fragment->offset - job->offset), fragment->length);
  }
  return true;
}

static void ple_release(ple_reader_t* reader) {
  if (!reader) return;
  if (reader->ring_ready) io_uring_queue_exit(&reader->ring);
  free(reader->iovecs);
  free(reader->free_slots);
  free(reader->registered_fds);
  for (size_t i = 0; i < reader->plan.file_count; i++) close(reader->plan.files[i].fd);
  free(reader->plan.files);
  free(reader->plan.sources);
  free(reader->buffers);
  free(reader->slots);
  free(reader->plan.fragments);
  free(reader->plan.jobs);
  free(reader->plan.file_counts);
  pthread_mutex_destroy(&reader->api_mutex);
  free(reader);
}

static void ple_delete(PyObject* capsule) {
  ple_reader_t* reader = PyCapsule_GetPointer(capsule, PLE_CAPSULE);
  if (reader) {
    Py_BEGIN_ALLOW_THREADS ple_release(reader);
    Py_END_ALLOW_THREADS
  }
}
#endif

#ifdef B12X_HAVE_LIBURING
/* The ring is driven by whichever thread happens to be running the row cache,
 * never by one dedicated task: py_ple_reader_run serialises its callers on
 * api_mutex, but the synchronous read path (disk_table.py:856, :892 through
 * _fetch at disk_table.py:862-865) and the speculative prefetch worker
 * "b12x-ple-prefetch" (disk_table.py:663-675, whose _run_prefetch_job reaches
 * the same _fetch at disk_table.py:715 without holding the disk_table.py:440
 * lock) are different tasks, and the mutex only moves ownership between them.
 * So IORING_SETUP_SINGLE_ISSUER is out: the kernel binds the ring to the first
 * submitting task and answers a second one with -EEXIST, which would permanently
 * poison a working reader, and IORING_SETUP_DEFER_TASKRUN cannot be requested
 * without it. COOP_TASKRUN needs no single issuer and still keeps completion
 * task work off the IPI path; TASKRUN_FLAG only advertises pending task work.
 * Both are tried first and dropped on any kernel that rejects them. */
static const unsigned ple_flag_ladder[] = {
    IORING_SETUP_COOP_TASKRUN | IORING_SETUP_TASKRUN_FLAG,
    IORING_SETUP_COOP_TASKRUN,
    0u,
};

/* Bring the ring up at the best flag set the kernel accepts. Only a rejected
 * setup request may step down a rung: a resource error is the same error for
 * every rung, and the caller still has to report it. */
static int ple_ring_init(ple_reader_t* reader, unsigned entries) {
  int error = -EINVAL;
  for (size_t rung = 0; rung < sizeof(ple_flag_ladder) / sizeof(ple_flag_ladder[0]); rung++) {
    struct io_uring_params params;
    memset(&params, 0, sizeof(params));
    params.flags = ple_flag_ladder[rung];
    error = io_uring_queue_init_params(entries, &reader->ring, &params);
    if (error >= 0) {
      reader->ring_ready = true;
      reader->ring_flags = ple_flag_ladder[rung];
      reader->ring_flag_fallback = rung != 0;
      return 0;
    }
  }
  return error;
}

/* How many 64 KiB staging slots the locked-memory budget can cover
 * (io_uring/rsrc.c charges registered buffers against RLIMIT_MEMLOCK). This is
 * only an estimate to avoid a doomed first attempt: the limit is per-process
 * and shared with every other ring, so ple_configure()'s registration remains
 * authoritative and still halves the queue on refusal. */
static unsigned ple_slot_budget(unsigned requested) {
  struct rlimit limit;
  if (getrlimit(RLIMIT_MEMLOCK, &limit) != 0) return requested;
  if (limit.rlim_cur == RLIM_INFINITY) return requested;
  unsigned fits = (unsigned)(limit.rlim_cur / PLE_READ_MAX);
  return fits < requested ? fits : requested;
}

/* Allocate and register everything one queue depth needs. Nothing is published
 * to the reader unless every step succeeds, so the caller can simply retry at
 * half the slots: a failed attempt leaves no ring, no pinned buffers and no
 * dangling pointer behind. */
static int ple_configure(ple_reader_t* reader, unsigned slots, bool* buffers_failed) {
  char* buffers = NULL;
  ple_slot_t* slot_array = NULL;
  struct iovec* iovecs = NULL;
  unsigned* free_slots = NULL;
  *buffers_failed = false;
  int error = posix_memalign((void**)&buffers, PLE_BLOCK, (size_t)slots * PLE_READ_MAX);
  if (error) {
    error = -error;
    goto cleanup;
  }
  slot_array = calloc(slots, sizeof(*slot_array));
  iovecs = calloc(slots, sizeof(*iovecs));
  free_slots = calloc(slots, sizeof(*free_slots));
  if (!slot_array || !iovecs || !free_slots) {
    error = -ENOMEM;
    goto cleanup;
  }
  for (unsigned i = 0; i < slots; i++) {
    slot_array[i].buffer = buffers + (size_t)i * PLE_READ_MAX;
    iovecs[i].iov_base = slot_array[i].buffer;
    iovecs[i].iov_len = PLE_READ_MAX;
  }
  error = ple_ring_init(reader, slots);
  if (error) goto cleanup;
  error = io_uring_register_buffers(&reader->ring, iovecs, slots);
  if (error < 0) {
    *buffers_failed = true;
    goto cleanup;
  }
  reader->buffers = buffers;
  reader->slots = slot_array;
  reader->iovecs = iovecs;
  reader->free_slots = free_slots;
  reader->slots_count = slots;
  return 0;
cleanup:
  if (reader->ring_ready) {
    io_uring_queue_exit(&reader->ring);
    reader->ring_ready = false;
  }
  free(buffers);
  free(slot_array);
  free(iovecs);
  free(free_slots);
  return error;
}
#endif

static PyObject* py_ple_reader(PyObject* self, PyObject* args) {
  (void)self;
  long long shard_rows, padded_rows, tp_start, tp_end, weight_bytes, scale_bytes, max_lookups;
  int depth;
  if (!PyArg_ParseTuple(args, "LLLLLLLi", &shard_rows, &padded_rows, &tp_start, &tp_end,
                        &weight_bytes, &scale_bytes, &max_lookups, &depth))
    return NULL;
#ifndef B12X_HAVE_LIBURING
  return PyErr_Format(PyExc_RuntimeError,
                      "io_uring PLE support is unavailable: install liburing development headers "
                      "and pkg-config, then rebuild the b12x native loader");
#else
  if (shard_rows <= 0 || padded_rows <= 0 || tp_start < 0 || tp_end < tp_start ||
      tp_end > padded_rows || weight_bytes <= 0 || scale_bytes < 0 || max_lookups <= 0 ||
      depth <= 0 || depth > 32768 || weight_bytes > PY_SSIZE_T_MAX / max_lookups ||
      scale_bytes > PY_SSIZE_T_MAX / max_lookups || max_lookups > PY_SSIZE_T_MAX / 8)
    return PyErr_Format(PyExc_ValueError, "invalid PLE reader geometry or I/O capacity");
  /* At most floor(row_bytes / gran) + 2 fragments per plane, sized for the
   * smallest granularity ple_dio_gran() can report: a 512 B floor multiplies
   * the worst-case fragment count of a wide row by PLE_BLOCK / PLE_GRAN_MIN,
   * so the capacity bound must not assume PLE_BLOCK (_row_plan.h:6-7). */
  size_t per_row = (size_t)weight_bytes / PLE_GRAN_MIN + 2;
  if (scale_bytes) per_row += (size_t)scale_bytes / PLE_GRAN_MIN + 2;
  if ((size_t)max_lookups > SIZE_MAX / per_row ||
      (size_t)max_lookups * per_row > SIZE_MAX / sizeof(ple_fragment_t) ||
      (size_t)max_lookups * per_row > SIZE_MAX / sizeof(ple_job_t))
    return PyErr_Format(PyExc_OverflowError, "PLE batch metadata is too large");
  ple_reader_t* reader = calloc(1, sizeof(*reader));
  if (!reader) return PyErr_NoMemory();
  /* Only an initialized API mutex reaches ple_release. */
  int error = pthread_mutex_init(&reader->api_mutex, NULL);
  if (error) {
    free(reader);
    return PyErr_Format(PyExc_RuntimeError, "PLE mutex: %s", strerror(error));
  }
  reader->plan.shard_rows = shard_rows;
  reader->plan.padded_rows = padded_rows;
  reader->plan.tp_start = tp_start;
  reader->plan.tp_end = tp_end;
  reader->plan.weight_bytes = weight_bytes;
  reader->plan.scale_bytes = scale_bytes;
  reader->plan.max_lookups = max_lookups;
  reader->plan.capacity = (size_t)max_lookups * per_row;
  /* Opt this plan into the miss-IO pick: per-file statx DIOALIGN fragment
   * floors and the density-gated fragment sort. The GDS planner shares this
   * header and leaves the flag clear, so it keeps 4 KiB cuFile fragments. */
  reader->plan.io_uring_pick = true;
  reader->plan.fragments = calloc(reader->plan.capacity, sizeof(*reader->plan.fragments));
  reader->plan.jobs = calloc(reader->plan.capacity, sizeof(*reader->plan.jobs));
  if (!reader->plan.fragments || !reader->plan.jobs) {
    ple_release(reader);
    return PyErr_NoMemory();
  }
  reader->depth_requested = (unsigned)depth;
  unsigned slots =
      reader->depth_requested < PLE_DEFAULT_DEPTH ? PLE_DEFAULT_DEPTH : reader->depth_requested;
  unsigned budget = ple_slot_budget(slots);
  if (budget < slots) slots = budget;
  if (slots < PLE_MIN_SLOTS) slots = PLE_MIN_SLOTS;
  /* Registered buffers are pinned and charged against the locked-memory limit,
   * which is a per-process budget shared with every other ring, so the only
   * authoritative answer is the registration itself: on refusal, keep halving
   * the queue rather than failing the reader. A shallower ring is slower but
   * correct, and stats report the depth that actually landed. */
  for (;;) {
    bool buffers_failed = false;
    error = ple_configure(reader, slots, &buffers_failed);
    if (!error) break;
    if (slots <= PLE_MIN_SLOTS) {
      ple_release(reader);
      if (buffers_failed)
        return PyErr_Format(PyExc_RuntimeError,
                            "io_uring buffer registration failed: %s; raise the locked-memory "
                            "limit or reduce queue_depth",
                            strerror(-error));
      return PyErr_Format(
          PyExc_RuntimeError,
          "io_uring initialization failed: %s; enable io_uring in the kernel/container policy",
          strerror(-error));
    }
    unsigned smaller = slots / 2 < PLE_MIN_SLOTS ? PLE_MIN_SLOTS : slots / 2;
    slots = smaller;
  }
  PyObject* capsule = PyCapsule_New(reader, PLE_CAPSULE, ple_delete);
  if (!capsule) ple_release(reader);
  return capsule;
#endif
}

static int ple_register_files(ple_plan_t* plan, void* owner) {
  (void)plan;
  ple_reader_t* reader = owner;
#ifdef B12X_HAVE_LIBURING
  int* fds = realloc(reader->registered_fds, reader->plan.file_count * sizeof(*fds));
  if (!fds) return -ENOMEM;
  reader->registered_fds = fds;
  for (size_t i = 0; i < reader->plan.file_count; i++) fds[i] = reader->plan.files[i].fd;
  if (reader->files_registered) {
    int error = io_uring_unregister_files(&reader->ring);
    if (error < 0) {
      reader->poisoned = true;
      return error;
    }
    reader->files_registered = false;
  }
  int error = io_uring_register_files(&reader->ring, fds, (unsigned)reader->plan.file_count);
  if (!error)
    reader->files_registered = true;
  else
    reader->poisoned = true;
  return error;
#else
  (void)reader;
  return 0;
#endif
}

static PyObject* py_ple_reader_add(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* capsule;
  const char* path;
  long long shard, offset;
  int scale;
  if (!PyArg_ParseTuple(args, "OLsLp", &capsule, &shard, &path, &offset, &scale)) return NULL;
  ple_reader_t* reader = PyCapsule_GetPointer(capsule, PLE_CAPSULE);
  if (!reader) return NULL;
  failure_t failure = {{0}};
  int error;
  Py_BEGIN_ALLOW_THREADS pthread_mutex_lock(&reader->api_mutex);
  reader->plan.failure.message[0] = 0;
  error =
      ple_add_source(&reader->plan, shard, path, offset, scale != 0, ple_register_files, reader);
  failure = reader->plan.failure;
  pthread_mutex_unlock(&reader->api_mutex);
  Py_END_ALLOW_THREADS if (error == EINVAL) return PyErr_Format(
      PyExc_ValueError, "invalid PLE source shard, offset, or scale plane");
  if (error == EOVERFLOW)
    return PyErr_Format(PyExc_OverflowError, "PLE source byte range overflows int64");
  if (error)
    return PyErr_Format(
        PyExc_RuntimeError, "%s",
        failure.message[0] ? failure.message : "invalid PLE source geometry or byte range");
  Py_RETURN_NONE;
}

#ifdef B12X_HAVE_LIBURING
/* Arm jobs into the SQ ring, popping slots off the back of the free list, and
 * return how many were armed (free_count is handed the reduced count, so a
 * slot is either on the free list or in flight, never both). A short positive
 * submit accounts for exactly that many SQEs; only the remainder is re-issued.
 * A failed submit may leave pending SQEs, so stop issuing, drain every
 * submitted CQE, and retire the ring without submitting those pending entries. */
static unsigned ple_prepare_wave(ple_reader_t* reader, unsigned* free_count, size_t* next) {
  size_t remaining = reader->plan.job_count - *next;
  unsigned count = remaining < *free_count ? (unsigned)remaining : *free_count;
  for (unsigned i = 0; i < count; i++) {
    unsigned slot = reader->free_slots[--*free_count];
    struct io_uring_sqe* sqe = io_uring_get_sqe(&reader->ring);
    if (!sqe) {
      ple_error(reader, "io_uring SQ capacity", ENOSPC);
      reader->poisoned = true;
      return i;
    }
    ple_job_t* job = &reader->plan.jobs[*next];
    reader->slots[slot].job = (*next)++;
    io_uring_prep_read_fixed(sqe, (int)job->file, reader->slots[slot].buffer, job->length,
                             job->offset, (int)slot);
    sqe->flags |= IOSQE_FIXED_FILE;
    io_uring_sqe_set_data64(sqe, slot);
  }
  return count;
}

/* One enter that submits every armed SQE and blocks until wait_nr completions
 * are visible, replacing the shipped wait-one-CQE granularity: with 64 slots the
 * wave kept only ~11 requests in flight (miss_io.md §1), so the device idled
 * between reaps. The caller waits for half the ring, which is also the point at
 * which refilling is worthwhile (t/io_uring batches 32 of 128 the same way). */
static void ple_flush(ple_reader_t* reader, unsigned prepared, unsigned* outstanding,
                      unsigned wait_nr) {
  unsigned pending = prepared;
  bool entered = false;
  while (pending || !entered) {
    reader->plan.submit_calls++;
    int result = io_uring_submit_and_wait(&reader->ring, wait_nr);
    entered = true;
    if (result == -EINTR) continue;
    if (result < 0) {
      ple_error(reader, "io_uring submit", -result);
      reader->poisoned = true;
      return;
    }
    *outstanding += (unsigned)result;
    pending = result < (int)pending ? pending - (unsigned)result : 0;
  }
}

static void ple_uring_run(ple_reader_t* reader) {
  const unsigned batch = reader->slots_count / 2 ? reader->slots_count / 2 : 1;
  unsigned outstanding = 0, free_count = reader->slots_count;
  size_t next = 0;
  for (unsigned i = 0; i < reader->slots_count; i++) reader->free_slots[i] = i;
  while (outstanding || next < reader->plan.job_count) {
    /* Refill from every slot the previous reap handed back, so the queue stays
     * saturated across the wave instead of one completion at a time. Do not arm
     * refill SQEs until every collected completion has been checked: an I/O
     * error therefore leaves no speculative refill entries. */
    unsigned prepared = 0;
    if (!reader->poisoned && free_count && next < reader->plan.job_count &&
        !reader->plan.failure.message[0])
      prepared = ple_prepare_wave(reader, &free_count, &next);
    if (reader->poisoned) {
      /* Drain only: entering here would submit the abandoned pending SQEs. */
      if (!outstanding) break;
      struct io_uring_cqe* cqe = NULL;
      int result = io_uring_wait_cqe(&reader->ring, &cqe);
      if (result < 0 && result != -EINTR && result != -EAGAIN) {
        ple_error(reader, "io_uring completion wait", -result);
        reader->poisoned = true;
      }
    } else {
      unsigned in_flight = outstanding + prepared;
      if (!in_flight) break; /* nothing submitted, nothing left to arm */
      ple_flush(reader, prepared, &outstanding, in_flight < batch ? in_flight : batch);
    }
    if (!outstanding) continue;
    unsigned head, reaped = 0;
    struct io_uring_cqe* cqe = NULL;
    io_uring_for_each_cqe(&reader->ring, head, cqe) {
      unsigned slot = (unsigned)io_uring_cqe_get_data64(cqe);
      int status = cqe->res;
      outstanding--;
      ple_scatter(reader, reader->slots[slot].job, reader->slots[slot].buffer, status);
      reader->free_slots[free_count++] = slot;
      reaped++;
      if (!outstanding) break;
    }
    io_uring_cq_advance(&reader->ring, reaped);
  }
  if (reader->poisoned) {
    io_uring_queue_exit(&reader->ring);
    reader->ring_ready = false;
  }
}
#endif

static bool ple_overlap(const Py_buffer* a, size_t a_bytes, const Py_buffer* b, size_t b_bytes) {
  uintptr_t x = (uintptr_t)a->buf, y = (uintptr_t)b->buf;
  return a_bytes && b_bytes && (x <= y ? y - x < a_bytes : x - y < b_bytes);
}

static PyObject* py_ple_reader_run(PyObject* self, PyObject* args) {
  (void)self;
  PyObject *capsule, *ids_object, *weights_object, *scales_object;
  long long count;
  if (!PyArg_ParseTuple(args, "OOOOL", &capsule, &ids_object, &weights_object, &scales_object,
                        &count))
    return NULL;
  ple_reader_t* reader = PyCapsule_GetPointer(capsule, PLE_CAPSULE);
  if (!reader) return NULL;
  if (count < 0 || (uint64_t)count > reader->plan.max_lookups)
    return PyErr_Format(PyExc_ValueError, "PLE lookup count exceeds batch capacity");
  Py_buffer ids = {0}, weights = {0}, scales = {0};
  if (PyObject_GetBuffer(ids_object, &ids, PyBUF_CONTIG_RO) < 0) return NULL;
  if (PyObject_GetBuffer(weights_object, &weights, PyBUF_CONTIG) < 0) goto failed;
  if (reader->plan.scale_bytes && PyObject_GetBuffer(scales_object, &scales, PyBUF_CONTIG) < 0)
    goto failed;
  if (!reader->plan.scale_bytes && scales_object != Py_None) {
    PyErr_SetString(PyExc_ValueError, "PLE reader has no scale plane; pass None");
    goto failed;
  }
  size_t id_bytes = (size_t)count * 8;
  size_t weight_bytes = (size_t)count * reader->plan.weight_bytes;
  size_t scale_bytes = (size_t)count * reader->plan.scale_bytes;
  if ((size_t)ids.len < id_bytes || (size_t)weights.len < weight_bytes ||
      (size_t)scales.len < scale_bytes || !PyBuffer_IsContiguous(&ids, 'C') ||
      !PyBuffer_IsContiguous(&weights, 'C') ||
      (reader->plan.scale_bytes && !PyBuffer_IsContiguous(&scales, 'C'))) {
    PyErr_SetString(PyExc_ValueError,
                    "PLE buffers must be C-contiguous and cover count rows (IDs are native signed "
                    "int64 bytes)");
    goto failed;
  }
  if (ple_overlap(&ids, id_bytes, &weights, weight_bytes) ||
      ple_overlap(&ids, id_bytes, &scales, scale_bytes) ||
      ple_overlap(&weights, weight_bytes, &scales, scale_bytes)) {
    PyErr_SetString(PyExc_ValueError, "PLE IDs and destination byte ranges must not overlap");
    goto failed;
  }
  failure_t failure = {{0}};
  Py_BEGIN_ALLOW_THREADS pthread_mutex_lock(&reader->api_mutex);
  struct timespec start, planned_at, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  reader->plan.failure.message[0] = 0;
  reader->plan.lookups = count;
  reader->plan.requested_bytes = reader->plan.read_bytes = reader->plan.read_calls = 0;
  reader->plan.unique_blocks = reader->plan.coalesced_reads = reader->plan.submit_calls = 0;
  reader->plan.planning_seconds = 0;
  if (reader->poisoned)
    snprintf(reader->plan.failure.message, sizeof(reader->plan.failure.message),
             "PLE io_uring reader is unusable after a submission/completion failure; create a new "
             "reader");
  else {
    bool planned = ple_plan(&reader->plan, ids.buf, weights.buf, scales.buf, count);
    clock_gettime(CLOCK_MONOTONIC, &planned_at);
    reader->plan.planning_seconds = (double)(planned_at.tv_sec - start.tv_sec) +
                                    (double)(planned_at.tv_nsec - start.tv_nsec) * 1e-9;
#ifdef B12X_HAVE_LIBURING
    reader->weights = weights.buf;
    reader->scales = scales.buf;
    if (planned && reader->plan.job_count) ple_uring_run(reader);
#else
    (void)planned;
#endif
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  reader->plan.execution_seconds =
      (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) * 1e-9;
  failure = reader->plan.failure;
  pthread_mutex_unlock(&reader->api_mutex);
  Py_END_ALLOW_THREADS PyBuffer_Release(&ids);
  PyBuffer_Release(&weights);
  if (scales.obj) PyBuffer_Release(&scales);
  if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
  Py_RETURN_NONE;
failed:
  PyBuffer_Release(&ids);
  if (weights.obj) PyBuffer_Release(&weights);
  if (scales.obj) PyBuffer_Release(&scales);
  return NULL;
}

static PyObject* py_ple_reader_stats(PyObject* self, PyObject* capsule) {
  (void)self;
  ple_reader_t* reader = PyCapsule_GetPointer(capsule, PLE_CAPSULE);
  if (!reader) return NULL;
  uint64_t lookups, requested, bytes, calls, blocks, coalesced, submits, staging, metadata;
  double seconds, planning;
  /* The queue depth that actually landed, the depth the caller asked for, the
   * smallest per-file fragment floor in use (0 with no sources), and the
   * IORING_SETUP_* bits the kernel accepted: the three knobs this stage can
   * silently fall back on (clamp, statx, flag ladder) stay observable. */
  unsigned depth_effective, depth_requested, dio_align, uring_flags;
  Py_BEGIN_ALLOW_THREADS pthread_mutex_lock(&reader->api_mutex);
  lookups = reader->plan.lookups;
  requested = reader->plan.requested_bytes;
  bytes = reader->plan.read_bytes;
  calls = reader->plan.read_calls;
  blocks = reader->plan.unique_blocks;
  coalesced = reader->plan.coalesced_reads;
  submits = reader->plan.submit_calls;
  seconds = reader->plan.execution_seconds;
  planning = reader->plan.planning_seconds;
  depth_effective = reader->slots_count;
  depth_requested = reader->depth_requested;
  dio_align = reader->plan.min_gran;
  uring_flags = reader->ring_flags;
  staging = (uint64_t)reader->slots_count * PLE_READ_MAX;
  metadata =
      sizeof(*reader) +
      reader->plan.capacity * (sizeof(*reader->plan.fragments) + sizeof(*reader->plan.jobs)) +
      reader->slots_count * sizeof(*reader->slots) +
      reader->plan.source_count * sizeof(*reader->plan.sources) +
      reader->plan.file_count * sizeof(*reader->plan.files) +
      reader->plan.file_counts_capacity * sizeof(*reader->plan.file_counts);
#ifdef B12X_HAVE_LIBURING
  metadata += reader->slots_count * (sizeof(*reader->iovecs) + sizeof(*reader->free_slots));
  metadata += reader->plan.file_count * sizeof(*reader->registered_fds);
  if (reader->ring_ready) {
    metadata += reader->ring.sq.ring_sz;
    if (reader->ring.cq.ring_ptr != reader->ring.sq.ring_ptr) metadata += reader->ring.cq.ring_sz;
    metadata += reader->ring.sq.ring_entries * sizeof(struct io_uring_sqe);
  }
#endif
  pthread_mutex_unlock(&reader->api_mutex);
  Py_END_ALLOW_THREADS return Py_BuildValue(
      "{s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:d,s:d,s:K,s:K,s:K,s:K}", "lookups",
      (unsigned long long)lookups, "requested_bytes", (unsigned long long)requested, "read_bytes",
      (unsigned long long)bytes, "read_calls", (unsigned long long)calls, "unique_blocks",
      (unsigned long long)blocks, "coalesced_reads", (unsigned long long)coalesced, "submit_calls",
      (unsigned long long)submits, "staging_bytes", (unsigned long long)staging, "metadata_bytes",
      (unsigned long long)metadata, "execution_seconds", seconds, "planning_seconds", planning,
      "depth_effective", (unsigned long long)depth_effective, "depth_requested",
      (unsigned long long)depth_requested, "dio_align", (unsigned long long)dio_align,
      "uring_flags", (unsigned long long)uring_flags);
}
