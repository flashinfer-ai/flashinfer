typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "A 64-bit CUDA host ABI is required");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) Dsv4TensorMap { uint64_t opaque[16]; };
struct __align__(64) Dsv4TensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(Dsv4TensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(Dsv4TensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) Dsv4TensorMapPack { Dsv4TensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(Dsv4TensorMap) >= alignof(CUtensorMap), "Dsv4TensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define DSV4_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100(__nv_bfloat16* __restrict__ route_outputs, __nv_bfloat16* __restrict__ route_weights, int* __restrict__ route_slots, __nv_bfloat16* __restrict__ output, int num_tokens, int route_stride, int M)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int feature = blockIdx.x * 256 + tid * 2;
    int token = blockIdx.y;
    float preloaded_weights[6];
    float preload_pair[2];
    if (feature + 1 < M && token < num_tokens) {
        {
            uint32_t _bf16x2_bits_0;
            _bf16x2_bits_0 = *reinterpret_cast<const uint32_t*>(route_weights + token * 6);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&preload_pair[0])[0]), "=f"((&preload_pair[0])[1])
                : "r"(_bf16x2_bits_0));
        }
        preloaded_weights[0] = preload_pair[0];
        preloaded_weights[1] = preload_pair[1];
        {
            uint32_t _bf16x2_bits_1;
            _bf16x2_bits_1 = *reinterpret_cast<const uint32_t*>(route_weights + token * 6 + 2);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&preload_pair[0])[0]), "=f"((&preload_pair[0])[1])
                : "r"(_bf16x2_bits_1));
        }
        preloaded_weights[2] = preload_pair[0];
        preloaded_weights[3] = preload_pair[1];
        {
            uint32_t _bf16x2_bits_2;
            _bf16x2_bits_2 = *reinterpret_cast<const uint32_t*>(route_weights + token * 6 + 4);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&preload_pair[0])[0]), "=f"((&preload_pair[0])[1])
                : "r"(_bf16x2_bits_2));
        }
        preloaded_weights[4] = preload_pair[0];
        preloaded_weights[5] = preload_pair[1];
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (feature + 1 < M && token < num_tokens) {
        float value[2] = {0};
        float partial[2];
        int route_base = token * 6;
        int slot_pair[2];
        int route_index = route_base;
        {
            const int2* _ivptr_3 = reinterpret_cast<const int2*>(route_slots + route_index);
            int2 _ivld_3;
            _ivld_3 = *_ivptr_3;
            slot_pair[0 + 0] = _ivld_3.x;
            slot_pair[0 + 1] = _ivld_3.y;
        }
        int output_row = slot_pair[0];
        {
            uint32_t _bf16x2_bits_4;
            _bf16x2_bits_4 = *reinterpret_cast<const uint32_t*>(route_outputs + output_row * route_stride + feature);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&partial[0])[0]), "=f"((&partial[0])[1])
                : "r"(_bf16x2_bits_4));
        }
        float weight = preloaded_weights[0];
        float _fma_0 = __fmaf_rn(partial[0], weight, value[0]);
        value[0] = _fma_0;
        float _fma_1 = __fmaf_rn(partial[1], weight, value[1]);
        value[1] = _fma_1;
        int output_row_0 = slot_pair[1];
        {
            uint32_t _bf16x2_bits_5;
            _bf16x2_bits_5 = *reinterpret_cast<const uint32_t*>(route_outputs + output_row_0 * route_stride + feature);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&partial[0])[0]), "=f"((&partial[0])[1])
                : "r"(_bf16x2_bits_5));
        }
        float weight_1 = preloaded_weights[1];
        float _fma_2 = __fmaf_rn(partial[0], weight_1, value[0]);
        value[0] = _fma_2;
        float _fma_3 = __fmaf_rn(partial[1], weight_1, value[1]);
        value[1] = _fma_3;
        int route_index_2 = route_base + 2;
        {
            const int2* _ivptr_6 = reinterpret_cast<const int2*>(route_slots + route_index_2);
            int2 _ivld_6;
            _ivld_6 = *_ivptr_6;
            slot_pair[0 + 0] = _ivld_6.x;
            slot_pair[0 + 1] = _ivld_6.y;
        }
        int output_row_3 = slot_pair[0];
        {
            uint32_t _bf16x2_bits_7;
            _bf16x2_bits_7 = *reinterpret_cast<const uint32_t*>(route_outputs + output_row_3 * route_stride + feature);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&partial[0])[0]), "=f"((&partial[0])[1])
                : "r"(_bf16x2_bits_7));
        }
        float weight_4 = preloaded_weights[2];
        float _fma_4 = __fmaf_rn(partial[0], weight_4, value[0]);
        value[0] = _fma_4;
        float _fma_5 = __fmaf_rn(partial[1], weight_4, value[1]);
        value[1] = _fma_5;
        int output_row_5 = slot_pair[1];
        {
            uint32_t _bf16x2_bits_8;
            _bf16x2_bits_8 = *reinterpret_cast<const uint32_t*>(route_outputs + output_row_5 * route_stride + feature);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&partial[0])[0]), "=f"((&partial[0])[1])
                : "r"(_bf16x2_bits_8));
        }
        float weight_6 = preloaded_weights[3];
        float _fma_6 = __fmaf_rn(partial[0], weight_6, value[0]);
        value[0] = _fma_6;
        float _fma_7 = __fmaf_rn(partial[1], weight_6, value[1]);
        value[1] = _fma_7;
        int route_index_7 = route_base + 4;
        {
            const int2* _ivptr_9 = reinterpret_cast<const int2*>(route_slots + route_index_7);
            int2 _ivld_9;
            _ivld_9 = *_ivptr_9;
            slot_pair[0 + 0] = _ivld_9.x;
            slot_pair[0 + 1] = _ivld_9.y;
        }
        int output_row_8 = slot_pair[0];
        {
            uint32_t _bf16x2_bits_10;
            _bf16x2_bits_10 = *reinterpret_cast<const uint32_t*>(route_outputs + output_row_8 * route_stride + feature);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&partial[0])[0]), "=f"((&partial[0])[1])
                : "r"(_bf16x2_bits_10));
        }
        float weight_9 = preloaded_weights[4];
        float _fma_8 = __fmaf_rn(partial[0], weight_9, value[0]);
        value[0] = _fma_8;
        float _fma_9 = __fmaf_rn(partial[1], weight_9, value[1]);
        value[1] = _fma_9;
        int output_row_10 = slot_pair[1];
        {
            uint32_t _bf16x2_bits_11;
            _bf16x2_bits_11 = *reinterpret_cast<const uint32_t*>(route_outputs + output_row_10 * route_stride + feature);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&partial[0])[0]), "=f"((&partial[0])[1])
                : "r"(_bf16x2_bits_11));
        }
        float weight_11 = preloaded_weights[5];
        float _fma_10 = __fmaf_rn(partial[0], weight_11, value[0]);
        value[0] = _fma_10;
        float _fma_11 = __fmaf_rn(partial[1], weight_11, value[1]);
        value[1] = _fma_11;
        {
            __nv_bfloat162 _pk = __floats2bfloat162_rn(value[0 + 0], value[0 + 1]);
            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(output + (token * M + feature)))[0]) = _pk;
        }
    }
}

} // extern "C"

