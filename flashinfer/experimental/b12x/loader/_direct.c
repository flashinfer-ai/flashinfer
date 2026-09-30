/* Small O_DIRECT reads for safetensors headers and CPU metadata. */
#include <fcntl.h>
#define IO_ALIGNMENT 4096
#define IO_SCRATCH_BYTES (64 << 10)
typedef struct {
    void *scratch;
    int scratch_bytes;
    pthread_mutex_t mutex;
    uint64_t physical_bytes, reads;
} direct_reader_t;

static void delete_reader(PyObject *capsule) {
    direct_reader_t *reader = PyCapsule_GetPointer(capsule, "b12x.direct_reader");
    if (!reader) return;
    free(reader->scratch);
    pthread_mutex_destroy(&reader->mutex);
    free(reader);
}
static PyObject *py_direct_reader(PyObject *self, PyObject *args) {
    (void)self;
    int device;
    if (!PyArg_ParseTuple(args, "i", &device)) return NULL;
    direct_reader_t *reader = calloc(1, sizeof(*reader));
    if (!reader) return PyErr_NoMemory();
    reader->scratch_bytes = IO_SCRATCH_BYTES;
    int error = posix_memalign(&reader->scratch, IO_ALIGNMENT, IO_SCRATCH_BYTES);
    if (error) { free(reader); return PyErr_Format(PyExc_RuntimeError, "metadata scratch: %s", strerror(error)); }
    pthread_mutex_init(&reader->mutex, NULL);
    PyObject *capsule = PyCapsule_New(reader, "b12x.direct_reader", delete_reader);
    if (!capsule) { free(reader->scratch); pthread_mutex_destroy(&reader->mutex); free(reader); }
    return capsule;
}

static bool validate_direct_range(int fd, int64_t offset, int64_t bytes,
                                  failure_t *failure) {
    struct stat status;
    int flags = fcntl(fd, F_GETFL);
    if (flags < 0 || !(flags & O_DIRECT)) {
        snprintf(failure->message, sizeof(failure->message), "direct reader requires O_DIRECT");
        return false;
    }
    if (fstat(fd, &status) != 0) {
        system_error(failure, "fstat direct input");
        return false;
    }
    if (offset < 0 || bytes < 0 || offset > INT64_MAX - bytes ||
        !S_ISREG(status.st_mode) || offset + bytes > status.st_size) {
        snprintf(failure->message, sizeof(failure->message), "invalid direct input file range");
        return false;
    }
    return true;
}

static bool direct_read_range(direct_reader_t *reader, int fd, int64_t offset,
                              int64_t bytes, char *destination, bool allow_direct,
                              failure_t *failure) {
    if (!validate_direct_range(fd, offset, bytes, failure)) return false;
    while (bytes) {
        int64_t aligned_offset = offset & ~(int64_t)(IO_ALIGNMENT - 1);
        size_t delta = offset - aligned_offset;
        (void)allow_direct;
        size_t payload = bytes < reader->scratch_bytes - (int64_t)delta ?
                         (size_t)bytes : reader->scratch_bytes - delta;
        size_t length = (payload + delta + IO_ALIGNMENT - 1) & ~(size_t)(IO_ALIGNMENT - 1);
        void *buffer = reader->scratch;
        ssize_t count;
        do {
            count = pread(fd, buffer, length, aligned_offset);
        } while (count < 0 && errno == EINTR);
        if (count < 0) {
            system_error(failure, "O_DIRECT pread (no buffered fallback)");
            return false;
        }
        if ((size_t)count < delta + payload) {
            snprintf(failure->message, sizeof(failure->message), "short O_DIRECT read");
            return false;
        }
        reader->physical_bytes += count;
        reader->reads++;
        memcpy(destination, (char *)buffer + delta, payload);
        destination += payload;
        offset += payload;
        bytes -= payload;
    }
    return true;
}

static PyObject *py_direct_bytes(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    int fd;
    long long offset, bytes;
    if (!PyArg_ParseTuple(args, "OiLL", &capsule, &fd, &offset, &bytes)) return NULL;
    direct_reader_t *reader = PyCapsule_GetPointer(capsule, "b12x.direct_reader");
    if (!reader) return NULL;
    if (bytes < 0 || bytes > 100 * (1 << 20))
        return PyErr_Format(PyExc_ValueError, "header/metadata read exceeds 100 MiB");
    PyObject *result = PyBytes_FromStringAndSize(NULL, bytes);
    if (!result) return NULL;
    bool success;
    failure_t failure = {{0}};
    char *destination = PyBytes_AS_STRING(result);
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&reader->mutex);
    success = direct_read_range(reader, fd, offset, bytes, destination, false, &failure);
    pthread_mutex_unlock(&reader->mutex);
    Py_END_ALLOW_THREADS
    if (!success) {
        Py_DECREF(result);
        return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    }
    return result;
}

static PyObject *py_direct_stats(PyObject *self, PyObject *capsule) {
    (void)self;
    direct_reader_t *reader = PyCapsule_GetPointer(capsule, "b12x.direct_reader");
    if (!reader) return NULL;
    return Py_BuildValue("{s:K,s:K,s:i}",
        "physical_bytes", (unsigned long long)reader->physical_bytes,
        "reads", (unsigned long long)reader->reads, "scratch_bytes", reader->scratch_bytes);
}
