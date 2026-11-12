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
kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100(__nv_bfloat16* __restrict__ route_outputs, __nv_bfloat16* __restrict__ route_weights, __nv_bfloat16* __restrict__ output, int num_tokens, int route_stride, int M)
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
    if (feature + 1 < M && token < num_tokens) {
        preloaded_weights[0] = (float)route_weights[token * 6];
        preloaded_weights[1] = (float)route_weights[token * 6 + 1];
        preloaded_weights[2] = (float)route_weights[token * 6 + 2];
        preloaded_weights[3] = (float)route_weights[token * 6 + 3];
        preloaded_weights[4] = (float)route_weights[token * 6 + 4];
        preloaded_weights[5] = (float)route_weights[token * 6 + 5];
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (feature + 1 < M && token < num_tokens) {
        float value[2] = {0};
        float partial[2];
        int route_base = token * 6;
        int route_index = route_base;
        int output_row = route_index;
        {
            uint32_t _bf16x2_bits_0;
            _bf16x2_bits_0 = *reinterpret_cast<const uint32_t*>(route_outputs + output_row * route_stride + feature);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&partial[0])[0]), "=f"((&partial[0])[1])
                : "r"(_bf16x2_bits_0));
        }
        float weight = preloaded_weights[0];
        float _fma_0 = __fmaf_rn(partial[0], weight, value[0]);
        value[0] = _fma_0;
        float _fma_1 = __fmaf_rn(partial[1], weight, value[1]);
        value[1] = _fma_1;
        int route_index_0 = route_base + 1;
        int output_row_1 = route_index_0;
        {
            uint32_t _bf16x2_bits_1;
            _bf16x2_bits_1 = *reinterpret_cast<const uint32_t*>(route_outputs + output_row_1 * route_stride + feature);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&partial[0])[0]), "=f"((&partial[0])[1])
                : "r"(_bf16x2_bits_1));
        }
        float weight_2 = preloaded_weights[1];
        float _fma_2 = __fmaf_rn(partial[0], weight_2, value[0]);
        value[0] = _fma_2;
        float _fma_3 = __fmaf_rn(partial[1], weight_2, value[1]);
        value[1] = _fma_3;
        int route_index_3 = route_base + 2;
        int output_row_4 = route_index_3;
        {
            uint32_t _bf16x2_bits_2;
            _bf16x2_bits_2 = *reinterpret_cast<const uint32_t*>(route_outputs + output_row_4 * route_stride + feature);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&partial[0])[0]), "=f"((&partial[0])[1])
                : "r"(_bf16x2_bits_2));
        }
        float weight_5 = preloaded_weights[2];
        float _fma_4 = __fmaf_rn(partial[0], weight_5, value[0]);
        value[0] = _fma_4;
        float _fma_5 = __fmaf_rn(partial[1], weight_5, value[1]);
        value[1] = _fma_5;
        int route_index_6 = route_base + 3;
        int output_row_7 = route_index_6;
        {
            uint32_t _bf16x2_bits_3;
            _bf16x2_bits_3 = *reinterpret_cast<const uint32_t*>(route_outputs + output_row_7 * route_stride + feature);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&partial[0])[0]), "=f"((&partial[0])[1])
                : "r"(_bf16x2_bits_3));
        }
        float weight_8 = preloaded_weights[3];
        float _fma_6 = __fmaf_rn(partial[0], weight_8, value[0]);
        value[0] = _fma_6;
        float _fma_7 = __fmaf_rn(partial[1], weight_8, value[1]);
        value[1] = _fma_7;
        int route_index_9 = route_base + 4;
        int output_row_10 = route_index_9;
        {
            uint32_t _bf16x2_bits_4;
            _bf16x2_bits_4 = *reinterpret_cast<const uint32_t*>(route_outputs + output_row_10 * route_stride + feature);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&partial[0])[0]), "=f"((&partial[0])[1])
                : "r"(_bf16x2_bits_4));
        }
        float weight_11 = preloaded_weights[4];
        float _fma_8 = __fmaf_rn(partial[0], weight_11, value[0]);
        value[0] = _fma_8;
        float _fma_9 = __fmaf_rn(partial[1], weight_11, value[1]);
        value[1] = _fma_9;
        int route_index_12 = route_base + 5;
        int output_row_13 = route_index_12;
        {
            uint32_t _bf16x2_bits_5;
            _bf16x2_bits_5 = *reinterpret_cast<const uint32_t*>(route_outputs + output_row_13 * route_stride + feature);
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&partial[0])[0]), "=f"((&partial[0])[1])
                : "r"(_bf16x2_bits_5));
        }
        float weight_14 = preloaded_weights[5];
        float _fma_10 = __fmaf_rn(partial[0], weight_14, value[0]);
        value[0] = _fma_10;
        float _fma_11 = __fmaf_rn(partial[1], weight_14, value[1]);
        value[1] = _fma_11;
        {
            __nv_bfloat162 _pk = __floats2bfloat162_rn(value[0 + 0], value[0 + 1]);
            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(output + (token * M + feature)))[0]) = _pk;
        }
    }
}

} // extern "C"

