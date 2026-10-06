#ifndef B12X_LOADER_CUDA_RANGE_H
#define B12X_LOADER_CUDA_RANGE_H
#include <cuda.h>

/* Expandable tensors can span several adjacent CUDA mappings. */
static bool device_range(uintptr_t pointer, uint64_t bytes, int device) {
    if (bytes > UINTPTR_MAX - pointer) return false;
    do {
        CUmemorytype kind;
        int ordinal;
        CUdeviceptr base;
        size_t size;
        if (cuPointerGetAttribute(&kind, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer) != CUDA_SUCCESS ||
            kind != CU_MEMORYTYPE_DEVICE ||
            cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, pointer) != CUDA_SUCCESS ||
            ordinal != device ||
            cuMemGetAddressRange(&base, &size, pointer) != CUDA_SUCCESS ||
            pointer < base || pointer - base >= size) return false;
        size_t available = size - (pointer - base);
        if (bytes <= available) return true;
        pointer += available;
        bytes -= available;
    } while (bytes);
    return true;
}
#endif
