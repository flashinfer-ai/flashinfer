typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128
#define IS_BF16 0

#include <math_constants.h>

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max_noftz(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}


__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}


__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ void fma_scale_x32(
    float* sv, const float2* scale2, const float2* neg_max2)
{
    float2* sv_2 = reinterpret_cast<float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++)
        fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_grouped_mxfp8_quantize_row2d_f16(__nv_bfloat16* __restrict__ x, int* __restrict__ mask, uint8_t* __restrict__ quantized, uint8_t* __restrict__ scales, int M, int K, int PADDED_K, int PM_TILES, int PK_TILES, int BLOCKS_PER_ROW, unsigned long long TOTAL_TASKS)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    unsigned int group = blockIdx.z;
    unsigned int row = blockIdx.y;
    unsigned int block_col = blockIdx.x * 128 + tid;
    if (block_col < (unsigned int)BLOCKS_PER_ROW) {
        int _vec_load_0[1];
        {
            _vec_load_0[0] = *reinterpret_cast<const int*>(mask + group);
        }
        int valid_rows = _vec_load_0[0];
        if (row < (unsigned int)valid_rows) {
            unsigned long long global_row = (unsigned long long)group * (unsigned long long)M + (unsigned long long)row;
            unsigned int k0 = block_col * 32;
            float values[32];
            if (k0 < (unsigned int)K) {
                unsigned long long x_offset = global_row * (unsigned long long)K + (unsigned long long)k0;
                {
                    {
                        const void* _v8p_0 = (const void*)(x + (x_offset));
                        uint32_t _v8_0_0[8];
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_v8_0_0[0]), "=r"(_v8_0_0[1]), "=r"(_v8_0_0[2]), "=r"(_v8_0_0[3]), "=r"(_v8_0_0[4]), "=r"(_v8_0_0[5]), "=r"(_v8_0_0[6]), "=r"(_v8_0_0[7]) : "l"((const char*)_v8p_0 + 0) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 0]))
                            : "r"(_v8_0_0[0]));
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 2]))
                            : "r"(_v8_0_0[1]));
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 4]))
                            : "r"(_v8_0_0[2]));
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 6]))
                            : "r"(_v8_0_0[3]));
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 8]))
                            : "r"(_v8_0_0[4]));
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 10]))
                            : "r"(_v8_0_0[5]));
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 12]))
                            : "r"(_v8_0_0[6]));
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 14]))
                            : "r"(_v8_0_0[7]));
                        uint32_t _v8_0_1[8];
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_v8_0_1[0]), "=r"(_v8_0_1[1]), "=r"(_v8_0_1[2]), "=r"(_v8_0_1[3]), "=r"(_v8_0_1[4]), "=r"(_v8_0_1[5]), "=r"(_v8_0_1[6]), "=r"(_v8_0_1[7]) : "l"((const char*)_v8p_0 + 32) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 16]))
                            : "r"(_v8_0_1[0]));
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 18]))
                            : "r"(_v8_0_1[1]));
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 20]))
                            : "r"(_v8_0_1[2]));
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 22]))
                            : "r"(_v8_0_1[3]));
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 24]))
                            : "r"(_v8_0_1[4]));
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 26]))
                            : "r"(_v8_0_1[5]));
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 28]))
                            : "r"(_v8_0_1[6]));
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 30]))
                            : "r"(_v8_0_1[7]));
                    }
                }
            } else {
                #pragma unroll
                for (int element = 0; element < 32; element++) {
                    values[element] = 0.0f;
                }
            }
            float magnitudes[32];
            #pragma unroll
            for (int element_1 = 0; element_1 < 32; element_1++) {
                magnitudes[element_1] = values[element_1];
            }
            float _fabs_0 = fabsf(magnitudes[0]);
            magnitudes[0] = _fabs_0;
            float _fabs_1 = fabsf(magnitudes[1]);
            magnitudes[1] = _fabs_1;
            float _fabs_2 = fabsf(magnitudes[2]);
            magnitudes[2] = _fabs_2;
            float _fabs_3 = fabsf(magnitudes[3]);
            magnitudes[3] = _fabs_3;
            float _fabs_4 = fabsf(magnitudes[4]);
            magnitudes[4] = _fabs_4;
            float _fabs_5 = fabsf(magnitudes[5]);
            magnitudes[5] = _fabs_5;
            float _fabs_6 = fabsf(magnitudes[6]);
            magnitudes[6] = _fabs_6;
            float _fabs_7 = fabsf(magnitudes[7]);
            magnitudes[7] = _fabs_7;
            float _fabs_8 = fabsf(magnitudes[8]);
            magnitudes[8] = _fabs_8;
            float _fabs_9 = fabsf(magnitudes[9]);
            magnitudes[9] = _fabs_9;
            float _fabs_10 = fabsf(magnitudes[10]);
            magnitudes[10] = _fabs_10;
            float _fabs_11 = fabsf(magnitudes[11]);
            magnitudes[11] = _fabs_11;
            float _fabs_12 = fabsf(magnitudes[12]);
            magnitudes[12] = _fabs_12;
            float _fabs_13 = fabsf(magnitudes[13]);
            magnitudes[13] = _fabs_13;
            float _fabs_14 = fabsf(magnitudes[14]);
            magnitudes[14] = _fabs_14;
            float _fabs_15 = fabsf(magnitudes[15]);
            magnitudes[15] = _fabs_15;
            float _fabs_16 = fabsf(magnitudes[16]);
            magnitudes[16] = _fabs_16;
            float _fabs_17 = fabsf(magnitudes[17]);
            magnitudes[17] = _fabs_17;
            float _fabs_18 = fabsf(magnitudes[18]);
            magnitudes[18] = _fabs_18;
            float _fabs_19 = fabsf(magnitudes[19]);
            magnitudes[19] = _fabs_19;
            float _fabs_20 = fabsf(magnitudes[20]);
            magnitudes[20] = _fabs_20;
            float _fabs_21 = fabsf(magnitudes[21]);
            magnitudes[21] = _fabs_21;
            float _fabs_22 = fabsf(magnitudes[22]);
            magnitudes[22] = _fabs_22;
            float _fabs_23 = fabsf(magnitudes[23]);
            magnitudes[23] = _fabs_23;
            float _fabs_24 = fabsf(magnitudes[24]);
            magnitudes[24] = _fabs_24;
            float _fabs_25 = fabsf(magnitudes[25]);
            magnitudes[25] = _fabs_25;
            float _fabs_26 = fabsf(magnitudes[26]);
            magnitudes[26] = _fabs_26;
            float _fabs_27 = fabsf(magnitudes[27]);
            magnitudes[27] = _fabs_27;
            float _fabs_28 = fabsf(magnitudes[28]);
            magnitudes[28] = _fabs_28;
            float _fabs_29 = fabsf(magnitudes[29]);
            magnitudes[29] = _fabs_29;
            float _fabs_30 = fabsf(magnitudes[30]);
            magnitudes[30] = _fabs_30;
            float _fabs_31 = fabsf(magnitudes[31]);
            magnitudes[31] = _fabs_31;
            float2 _reg_reduce_max2_1 = {-CAKE_INF, -CAKE_INF};
            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[0], magnitudes[1]));
            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[2], magnitudes[3]));
            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[4], magnitudes[5]));
            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[6], magnitudes[7]));
            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[8], magnitudes[9]));
            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[10], magnitudes[11]));
            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[12], magnitudes[13]));
            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[14], magnitudes[15]));
            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[16], magnitudes[17]));
            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[18], magnitudes[19]));
            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[20], magnitudes[21]));
            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[22], magnitudes[23]));
            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[24], magnitudes[25]));
            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[26], magnitudes[27]));
            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[28], magnitudes[29]));
            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[30], magnitudes[31]));
            float magnitudes_max = row_max_reduce(_reg_reduce_max2_1);
            float absmax = magnitudes_max;
            float scale = absmax / 448.0f;
            unsigned int scale_bits = __as_u32(scale);
            unsigned int exponent = scale_bits >> 23 & 255;
            unsigned int mantissa = scale_bits & 8388607;
            unsigned int has_mantissa = ((mantissa != 0) ? 1 : 0);
            unsigned int normal = ((exponent != 0) ? 1 : 0);
            unsigned int large_subnormal = ((mantissa > 4194304) ? 1 : 0);
            unsigned int _min_0 = ((exponent + (has_mantissa & (normal | large_subnormal))) < (254) ? (exponent + (has_mantissa & (normal | large_subnormal))) : (254));
            unsigned int scale_byte = _min_0;
            unsigned int inverse_nonzero_bits = 254 - scale_byte << 23;
            unsigned int zero_bits = 0;
            unsigned int inverse_bits = ((scale_byte == 0) ? zero_bits : inverse_nonzero_bits);
            float inverse = 0.0f;
            inverse = reinterpret_cast<float*>(&inverse_bits)[0];
            const float2 _scale2_2 = {inverse, inverse};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(values)[_ls], _scale2_2);
            unsigned long long task = global_row * (unsigned long long)BLOCKS_PER_ROW + (unsigned long long)block_col;
            unsigned long long q_offset = task * 32;
            {
                unsigned int _fp8_pk[8];
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[0]) : "f"(values[0 + 0]), "f"(values[0 + 1]), "f"(values[0 + 2]), "f"(values[0 + 3]));
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[1]) : "f"(values[0 + 4]), "f"(values[0 + 5]), "f"(values[0 + 6]), "f"(values[0 + 7]));
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[2]) : "f"(values[0 + 8]), "f"(values[0 + 9]), "f"(values[0 + 10]), "f"(values[0 + 11]));
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[3]) : "f"(values[0 + 12]), "f"(values[0 + 13]), "f"(values[0 + 14]), "f"(values[0 + 15]));
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[4]) : "f"(values[0 + 16]), "f"(values[0 + 17]), "f"(values[0 + 18]), "f"(values[0 + 19]));
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[5]) : "f"(values[0 + 20]), "f"(values[0 + 21]), "f"(values[0 + 22]), "f"(values[0 + 23]));
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[6]) : "f"(values[0 + 24]), "f"(values[0 + 25]), "f"(values[0 + 26]), "f"(values[0 + 27]));
                asm("{\n\t"
                    ".reg .b16 _lo, _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}\n"
                    : "=r"(_fp8_pk[7]) : "f"(values[0 + 28]), "f"(values[0 + 29]), "f"(values[0 + 30]), "f"(values[0 + 31]));
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"(reinterpret_cast<unsigned char*>(quantized + q_offset) + (0)), "r"(_fp8_pk[0]), "r"(_fp8_pk[1]), "r"(_fp8_pk[2]), "r"(_fp8_pk[3]), "r"(_fp8_pk[4]), "r"(_fp8_pk[5]), "r"(_fp8_pk[6]), "r"(_fp8_pk[7]) : "memory");
            }
            unsigned int m_tile = row / 128;
            unsigned int row_in_tile = row - m_tile * 128;
            unsigned int k_tile = block_col / 4;
            unsigned int scale_col = block_col - k_tile * 4;
            unsigned long long tile_linear = ((unsigned long long)group * (unsigned long long)PM_TILES + (unsigned long long)m_tile) * (unsigned long long)PK_TILES + (unsigned long long)k_tile;
            unsigned long long scale_offset = tile_linear * 512 + (unsigned long long)(row_in_tile & 31) * 16 + (unsigned long long)(row_in_tile >> 5) * 4 + (unsigned long long)scale_col;
            *(reinterpret_cast<unsigned char*>(scales + scale_offset) + (0)) = (unsigned char)(scale_byte);
        }
    }
}

} // extern "C"
