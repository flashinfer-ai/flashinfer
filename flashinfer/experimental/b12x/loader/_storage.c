#define _GNU_SOURCE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime_api.h>

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef char require_64_bit_offsets[(sizeof(off_t) == 8 && sizeof(size_t) == 8) ? 1 : -1];

typedef struct {
    char message[512];
} failure_t;

static bool cuda_ok(cudaError_t status, const char *operation, failure_t *failure) {
    if (status == cudaSuccess) return true;
    snprintf(failure->message, sizeof(failure->message), "%s: %s",
             operation, cudaGetErrorString(status));
    return false;
}

static void system_error(failure_t *failure, const char *operation) {
    snprintf(failure->message, sizeof(failure->message), "%s: %s",
             operation, strerror(errno));
}

static PyObject *py_capabilities(PyObject *self, PyObject *args) {
    (void)self;
    int device;
    if (!PyArg_ParseTuple(args, "i", &device)) return NULL;
    static const struct {
        const char *name;
        enum cudaDeviceAttr attribute;
    } keys[] = {
        {"integrated", cudaDevAttrIntegrated},
        {"can_map_host_memory", cudaDevAttrCanMapHostMemory},
        {"managed_memory", cudaDevAttrManagedMemory},
        {"concurrent_managed_access", cudaDevAttrConcurrentManagedAccess},
        {"pageable_memory_access", cudaDevAttrPageableMemoryAccess},
        {"host_page_tables", cudaDevAttrPageableMemoryAccessUsesHostPageTables},
        {"host_register_supported", cudaDevAttrHostRegisterSupported},
        {"registered_host_pointer", cudaDevAttrCanUseHostPointerForRegisteredMem},
    };
    PyObject *result = PyDict_New();
    if (!result) return NULL;
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
        int value;
        failure_t failure;
        if (!cuda_ok(cudaDeviceGetAttribute(&value, keys[i].attribute, device),
                     "cudaDeviceGetAttribute", &failure)) {
            Py_DECREF(result);
            return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
        }
        PyObject *number = PyLong_FromLong(value);
        if (!number || PyDict_SetItemString(result, keys[i].name, number) != 0) {
            Py_XDECREF(number);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(number);
    }
    return result;
}

#include "_direct.c"
#include "_batch.c"
#include "_ple_reader.c"

static PyMethodDef methods[] = {
    {"ple_reader", py_ple_reader, METH_VARARGS, NULL},
    {"ple_reader_add", py_ple_reader_add, METH_VARARGS, NULL},
    {"ple_reader_run", py_ple_reader_run, METH_VARARGS, NULL},
    {"ple_reader_stats", py_ple_reader_stats, METH_O,
     "Last-call counters; staging_bytes and metadata_bytes describe persistent allocations."},
    {"batch_executor", py_batch_executor, METH_VARARGS, NULL},
    {"batch_execute", py_batch_execute, METH_VARARGS, NULL},
    {"batch_stats", py_batch_stats, METH_O, NULL},
    {"direct_reader", py_direct_reader, METH_VARARGS, NULL},
    {"direct_bytes", py_direct_bytes, METH_VARARGS, NULL},
    {"direct_stats", py_direct_stats, METH_O, NULL},
    {"capabilities", py_capabilities, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL},
};
static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, .m_name = "_b12x_loader_storage", .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit__b12x_loader_storage(void) {
    PyObject *result = PyModule_Create(&module);
    if (result && PyModule_AddIntConstant(result, "ABI_VERSION", 1) < 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}
