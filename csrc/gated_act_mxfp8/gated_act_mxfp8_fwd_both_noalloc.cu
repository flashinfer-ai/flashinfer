typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_ROW_ACT_OFF 0
#define SMEM_ROW_ACT_STAGE_BYTES 2048
#define SMEM_ROW_ACT_STRIDE 2048
#define SMEM_COL_ACT_OFF 2048
#define SMEM_COL_ACT_STAGE_BYTES 2048
#define SMEM_COL_ACT_STRIDE 2048
#define SMEM_PAD_OFF 4096
#define SMEM_PAD_STAGE_BYTES 4608
#define SMEM_PAD_STRIDE 4608
#define SMEM_TOTAL 8704
#define THREADS 128
#define NO_ALLOCATE_ 1

#include <math_constants.h>

__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_gated_act_mxfp8_fwd_both_direct_64x64_noalloc(__nv_bfloat16* __restrict__ gated_input, const __grid_constant__ CUtensorMap row_act_tma, const __grid_constant__ CUtensorMap col_act_tma, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* row_act = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int row_act_addr = smem + 0;
    uint8_t* col_act = reinterpret_cast<uint8_t*>(smem_raw + 2048);
    const int col_act_addr = smem + 2048;
    unsigned int* pad = reinterpret_cast<unsigned int*>(smem_raw + 4096);
    const int pad_addr = smem + 4096;

    // === Task calls (dependency order) ===
    int tid_0 = tid;
    int half = tid_0 & 1;
    int blk = tid_0 >> 1 & 1;
    int row = tid_0 >> 2;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int wchk = row >> 2 ^ (blk * 2 + half) * 2;
    int wrow = row & 3;
    int tq = tid_0 & 3;
    int cpr = tid_0 >> 2 & 31;
    int arr = tid_0 >> 7;
    int rsw = (cpr >> 3 & 3) * 2;
    unsigned int gate_words[16];
    unsigned int up_words[16];
    unsigned int grad_words[16];
    float pair_values[2];
    unsigned int packed_act[8];
    unsigned int packed_gate[8];
    float row_scaled_quad[4];
    unsigned int row_fp8_act[4];
    unsigned int row_fp8_gate[4];
    unsigned int col_values[8];
    unsigned int col_a01[2];
    unsigned int col_a23[2];
    unsigned int col_even[2];
    unsigned int col_odd[2];
    float col_scaled_quad[4];
    float x0 = 0.0f;
    float x1 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;
    float grad0 = 0.0f;
    float grad1 = 0.0f;
    unsigned int bits0 = 0;
    unsigned int bits1 = 0;
    unsigned int bits2 = 0;
    unsigned int bits3 = 0;
    float value0 = 0.0f;
    float value1 = 0.0f;
    float value2 = 0.0f;
    float value3 = 0.0f;
    float2 _f2_0 = make_float2(0.0f, 0.0f);
    float2 out_act = _f2_0;
    float2 _f2_1 = make_float2(0.0f, 0.0f);
    float2 out_gate = _f2_1;
    unsigned int gate_scale = 0;
    unsigned int inv_gate = 0;
    int grow0 = by * 32 + row;
    int col0 = bx * 64 + blk * 32 + half * 16;
    int gate_index0 = grow0 * (2 * K) + col0;
    {
        {
            const void* _v8p_0 = (const void*)(gated_input + (gate_index0));
            uint32_t _v8_0_0[8];
            asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_v8_0_0[0]), "=r"(_v8_0_0[1]), "=r"(_v8_0_0[2]), "=r"(_v8_0_0[3]), "=r"(_v8_0_0[4]), "=r"(_v8_0_0[5]), "=r"(_v8_0_0[6]), "=r"(_v8_0_0[7]) : "l"((const void*)((const char*)_v8p_0 + 0)) : "memory");
            *(&gate_words[0 + 0]) = _v8_0_0[0];
            *(&gate_words[0 + 1]) = _v8_0_0[1];
            *(&gate_words[0 + 2]) = _v8_0_0[2];
            *(&gate_words[0 + 3]) = _v8_0_0[3];
            *(&gate_words[0 + 4]) = _v8_0_0[4];
            *(&gate_words[0 + 5]) = _v8_0_0[5];
            *(&gate_words[0 + 6]) = _v8_0_0[6];
            *(&gate_words[0 + 7]) = _v8_0_0[7];
        }
    }
    {
        {
            const void* _v8p_1 = (const void*)(gated_input + (gate_index0 + K));
            uint32_t _v8_1_0[8];
            asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_v8_1_0[0]), "=r"(_v8_1_0[1]), "=r"(_v8_1_0[2]), "=r"(_v8_1_0[3]), "=r"(_v8_1_0[4]), "=r"(_v8_1_0[5]), "=r"(_v8_1_0[6]), "=r"(_v8_1_0[7]) : "l"((const void*)((const char*)_v8p_1 + 0)) : "memory");
            *(&up_words[0 + 0]) = _v8_1_0[0];
            *(&up_words[0 + 1]) = _v8_1_0[1];
            *(&up_words[0 + 2]) = _v8_1_0[2];
            *(&up_words[0 + 3]) = _v8_1_0[3];
            *(&up_words[0 + 4]) = _v8_1_0[4];
            *(&up_words[0 + 5]) = _v8_1_0[5];
            *(&up_words[0 + 6]) = _v8_1_0[6];
            *(&up_words[0 + 7]) = _v8_1_0[7];
        }
    }
    #pragma unroll
    for (int stage = 0; stage < 1; stage++) {
        int grow = by * 32 + stage * 32 + row;
        unsigned int amax_act = 0;
        unsigned int amax_gate = 0;
        #pragma unroll
        for (int pair = 0; pair < 8; pair++) {
            unsigned int gate_word = gate_words[pair];
            unsigned int up_word = up_words[pair];
            bits0 = (gate_word & 65535) << 16;
            bits1 = gate_word & 4294901760;
            x0 = reinterpret_cast<float*>(&bits0)[0];
            x1 = reinterpret_cast<float*>(&bits1)[0];
            bits0 = (up_word & 65535) << 16;
            bits1 = up_word & 4294901760;
            up0 = reinterpret_cast<float*>(&bits0)[0];
            up1 = reinterpret_cast<float*>(&bits1)[0];
            float _exp2_noftz_0;
            asm volatile("ex2.approx.f32 %0, %1;" : "=f"(_exp2_noftz_0) : "f"((-x0) * 1.4426950408889634f));
            float _rcp_rn_0;
            asm volatile("rcp.rn.f32 %0, %1;" : "=f"(_rcp_rn_0) : "f"(1.0f + _exp2_noftz_0));
            float _exp2_noftz_1;
            asm volatile("ex2.approx.f32 %0, %1;" : "=f"(_exp2_noftz_1) : "f"((-x1) * 1.4426950408889634f));
            float _rcp_rn_1;
            asm volatile("rcp.rn.f32 %0, %1;" : "=f"(_rcp_rn_1) : "f"(1.0f + _exp2_noftz_1));
            float2 _f2_2 = make_float2(_rcp_rn_0, _rcp_rn_1);
            float2 sigmoid = _f2_2;
            float2 _f2_3 = make_float2(x0, x1);
            float2 _f32x2_mul_rn_0;
            asm volatile("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_f32x2_mul_rn_0) : "l"(*(const unsigned long long*)&_f2_3), "l"(*(const unsigned long long*)&sigmoid));
            float2 act = _f32x2_mul_rn_0;
            {
                float2 _f2_7 = make_float2(up0, up1);
                float2 _f32x2_mul_rn_4;
                asm volatile("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_f32x2_mul_rn_4) : "l"(*(const unsigned long long*)&act), "l"(*(const unsigned long long*)&_f2_7));
                out_act = _f32x2_mul_rn_4;
            }
            pair_values[0] = out_act.x;
            pair_values[1] = out_act.y;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_values[_lp*2 + 0], pair_values[_lp*2+1 + 0]));
                packed_act[(pair) + _lp] = *(uint32_t*)&_bf2;
            }
            uint32_t _bf16x2_abs_max_nan_0;
            asm volatile("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_abs_max_nan_0) : "r"(amax_act), "r"(packed_act[pair]));
            amax_act = _bf16x2_abs_max_nan_0;
            int pad_pair = blk * 16 + half * 8 + pair;
            int pad_word = pad_pair * 36 + wchk * 4 + wrow;
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(pad_addr + (unsigned int)(pad_word * 4)), "r"((packed_act[pair])));
        }
        __syncthreads();
        #pragma unroll
        for (int chunk = 0; chunk < 2; chunk++) {
            int logical_chunk = tq * 2 + chunk;
            int physical_chunk = logical_chunk ^ rsw;
            int pad_read_word = cpr * 36 + physical_chunk * 4;
            {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&col_values[chunk * 4])), "=r"(*reinterpret_cast<uint32_t*>(&col_values[(chunk * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&col_values[(chunk * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&col_values[(chunk * 4) + 3]))
                    : "r"(pad_addr + (unsigned int)(pad_read_word * 4)));
            }
        }
        amax_act = amax_act & 2147450879;
        unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, amax_act, 1);
        unsigned int peer_act = _shfl_xor_0;
        uint32_t _bf16x2_max_nan_0;
        asm volatile("max.NaN.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_nan_0) : "r"(amax_act), "r"(peer_act));
        amax_act = _bf16x2_max_nan_0;
        uint32_t _bf16x2_max_nan_1;
        asm volatile("max.NaN.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_nan_1) : "r"(amax_act), "r"(amax_act >> 16));
        amax_act = _bf16x2_max_nan_1;
        unsigned int act_bits = (amax_act & 65535) << 16;
        int act_scale_i32 = (int)(act_bits + 2031616 >> 23) - 8;
        if (act_scale_i32 < 0) {
            act_scale_i32 = 0;
        }
        unsigned int act_exponent = act_bits & 2139095040;
        if (act_exponent == 2139095040) {
            act_scale_i32 = 255;
        }
        unsigned int act_scale = (unsigned int)act_scale_i32;
        int scale_col = bx * 2 + blk;
        int row_num_scale_blocks = K / 128;
        int row_scale_index = ((grow >> 7) * row_num_scale_blocks + (scale_col >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (scale_col & 3);
        {
            if (half == 0) {
                *(reinterpret_cast<unsigned char*>(row_scales + row_scale_index) + (0)) = (unsigned char)(act_scale);
            }
        }
        unsigned int inv_act = 254 - act_scale << 7;
        if (act_scale == 255) {
            inv_act = 32704;
        }
        inv_act = inv_act | inv_act << 16;
        unsigned int gate_scale_0 = act_scale;
        int gate_row_scale_index = row_scale_index;
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            uint32_t _bf16x2_mul_0;
            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(packed_act[2 * q]), "r"(inv_act));
            unsigned int scaled_act0 = _bf16x2_mul_0;
            uint32_t _bf16x2_mul_1;
            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(packed_act[2 * q + 1]), "r"(inv_act));
            unsigned int scaled_act1 = _bf16x2_mul_1;
            bits0 = (scaled_act0 & 65535) << 16;
            bits1 = scaled_act0 & 4294901760;
            bits2 = (scaled_act1 & 65535) << 16;
            bits3 = scaled_act1 & 4294901760;
            value0 = reinterpret_cast<float*>(&bits0)[0];
            value1 = reinterpret_cast<float*>(&bits1)[0];
            value2 = reinterpret_cast<float*>(&bits2)[0];
            value3 = reinterpret_cast<float*>(&bits3)[0];
            row_scaled_quad[0] = value0;
            row_scaled_quad[1] = value1;
            row_scaled_quad[2] = value2;
            row_scaled_quad[3] = value3;
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(row_scaled_quad[0]), "f"(row_scaled_quad[1]),
                                       "f"(row_scaled_quad[2]), "f"(row_scaled_quad[3]));
                row_fp8_act[(q) + 0] = _packed;
            }
        }
        int word_quad = blk * 2 + half;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((row_act_addr + (unsigned int)(stage * 2048) + (unsigned int)(row * 64 + word_quad * 16))), "r"(row_fp8_act[0]), "r"(row_fp8_act[1]), "r"(row_fp8_act[2]), "r"(row_fp8_act[3]) : "memory");
        if (stage + 1 < 1) {
            int next_grow = by * 32 + (stage + 1) * 32 + row;
            int next_col = bx * 64 + blk * 32 + half * 16;
            int next_gate_index = next_grow * (2 * K) + next_col;
            {
                {
                    const void* _v8p_2 = (const void*)(gated_input + (next_gate_index));
                    uint32_t _v8_2_0[8];
                    asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(_v8_2_0[0]), "=r"(_v8_2_0[1]), "=r"(_v8_2_0[2]), "=r"(_v8_2_0[3]), "=r"(_v8_2_0[4]), "=r"(_v8_2_0[5]), "=r"(_v8_2_0[6]), "=r"(_v8_2_0[7]) : "l"((const void*)((const char*)_v8p_2 + 0)) : "memory");
                    *(&gate_words[0 + 0]) = _v8_2_0[0];
                    *(&gate_words[0 + 1]) = _v8_2_0[1];
                    *(&gate_words[0 + 2]) = _v8_2_0[2];
                    *(&gate_words[0 + 3]) = _v8_2_0[3];
                    *(&gate_words[0 + 4]) = _v8_2_0[4];
                    *(&gate_words[0 + 5]) = _v8_2_0[5];
                    *(&gate_words[0 + 6]) = _v8_2_0[6];
                    *(&gate_words[0 + 7]) = _v8_2_0[7];
                }
            }
            {
                {
                    const void* _v8p_3 = (const void*)(gated_input + (next_gate_index + K));
                    uint32_t _v8_3_0[8];
                    asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(_v8_3_0[0]), "=r"(_v8_3_0[1]), "=r"(_v8_3_0[2]), "=r"(_v8_3_0[3]), "=r"(_v8_3_0[4]), "=r"(_v8_3_0[5]), "=r"(_v8_3_0[6]), "=r"(_v8_3_0[7]) : "l"((const void*)((const char*)_v8p_3 + 0)) : "memory");
                    *(&up_words[0 + 0]) = _v8_3_0[0];
                    *(&up_words[0 + 1]) = _v8_3_0[1];
                    *(&up_words[0 + 2]) = _v8_3_0[2];
                    *(&up_words[0 + 3]) = _v8_3_0[3];
                    *(&up_words[0 + 4]) = _v8_3_0[4];
                    *(&up_words[0 + 5]) = _v8_3_0[5];
                    *(&up_words[0 + 6]) = _v8_3_0[6];
                    *(&up_words[0 + 7]) = _v8_3_0[7];
                }
            }
        }
        unsigned int col_amax = 0;
        #pragma unroll
        for (int chunk_1 = 0; chunk_1 < 2; chunk_1++) {
            #pragma unroll
            for (int word = 0; word < 4; word++) {
                uint32_t _bf16x2_abs_max_nan_2;
                asm volatile("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_abs_max_nan_2) : "r"(col_amax), "r"(col_values[chunk_1 * 4 + word]));
                col_amax = _bf16x2_abs_max_nan_2;
            }
        }
        col_amax = col_amax & 2147450879;
        unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, col_amax, 1);
        unsigned int col_peer = _shfl_xor_2;
        uint32_t _bf16x2_max_nan_4;
        asm volatile("max.NaN.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_nan_4) : "r"(col_amax), "r"(col_peer));
        col_amax = _bf16x2_max_nan_4;
        unsigned int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, col_amax, 2);
        unsigned int col_peer_1 = _shfl_xor_3;
        uint32_t _bf16x2_max_nan_5;
        asm volatile("max.NaN.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_nan_5) : "r"(col_amax), "r"(col_peer_1));
        col_amax = _bf16x2_max_nan_5;
        unsigned int col_bits0 = (col_amax & 65535) << 16;
        unsigned int col_bits1 = col_amax & 4294901760;
        int col_scale0_i32 = (int)(col_bits0 + 2031616 >> 23) - 8;
        int col_scale1_i32 = (int)(col_bits1 + 2031616 >> 23) - 8;
        if (col_scale0_i32 < 0) {
            col_scale0_i32 = 0;
        }
        if (col_scale1_i32 < 0) {
            col_scale1_i32 = 0;
        }
        unsigned int col_exponent0 = col_bits0 & 2139095040;
        unsigned int col_exponent1 = col_bits1 & 2139095040;
        if (col_exponent0 == 2139095040) {
            col_scale0_i32 = 255;
        }
        if (col_exponent1 == 2139095040) {
            col_scale1_i32 = 255;
        }
        unsigned int col_scale0 = (unsigned int)col_scale0_i32;
        unsigned int col_scale1 = (unsigned int)col_scale1_i32;
        int row_tile = by + stage;
        int col_out_col = bx * 64 + cpr * 2;
        int col_num_scale_blocks = M / 128;
        int col_scale_index0 = ((col_out_col >> 7) * col_num_scale_blocks + (row_tile >> 2)) * 512 + (col_out_col & 31) * 16 + (col_out_col >> 5 & 3) * 4 + (row_tile & 3);
        int col_out_col1 = col_out_col + 1;
        int col_scale_index1 = ((col_out_col1 >> 7) * col_num_scale_blocks + (row_tile >> 2)) * 512 + (col_out_col1 & 31) * 16 + (col_out_col1 >> 5 & 3) * 4 + (row_tile & 3);
        if (tq == 0) {
            *(reinterpret_cast<unsigned char*>(col_scales + col_scale_index0) + (0)) = (unsigned char)(col_scale0);
            *(reinterpret_cast<unsigned char*>(col_scales + col_scale_index1) + (0)) = (unsigned char)(col_scale1);
        }
        unsigned int col_inv0 = 254 - col_scale0 << 7;
        unsigned int col_inv1 = 254 - col_scale1 << 7;
        if (col_scale0 == 255) {
            col_inv0 = 32704;
        }
        if (col_scale1 == 255) {
            col_inv1 = 32704;
        }
        unsigned int col_inv_pair = col_inv0 | col_inv1 << 16;
        #pragma unroll
        for (int chunk_2 = 0; chunk_2 < 2; chunk_2++) {
            uint32_t _bf16x2_mul_4;
            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(col_values[chunk_2 * 4]), "r"(col_inv_pair));
            unsigned int col_scaled0 = _bf16x2_mul_4;
            uint32_t _bf16x2_mul_5;
            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(col_values[chunk_2 * 4 + 1]), "r"(col_inv_pair));
            unsigned int col_scaled1 = _bf16x2_mul_5;
            bits0 = (col_scaled0 & 65535) << 16;
            bits1 = col_scaled0 & 4294901760;
            bits2 = (col_scaled1 & 65535) << 16;
            bits3 = col_scaled1 & 4294901760;
            value0 = reinterpret_cast<float*>(&bits0)[0];
            value1 = reinterpret_cast<float*>(&bits1)[0];
            value2 = reinterpret_cast<float*>(&bits2)[0];
            value3 = reinterpret_cast<float*>(&bits3)[0];
            col_scaled_quad[0] = value0;
            col_scaled_quad[1] = value1;
            col_scaled_quad[2] = value2;
            col_scaled_quad[3] = value3;
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(col_scaled_quad[0]), "f"(col_scaled_quad[1]),
                                       "f"(col_scaled_quad[2]), "f"(col_scaled_quad[3]));
                col_a01[(chunk_2) + 0] = _packed;
            }
            uint32_t _bf16x2_mul_6;
            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(col_values[chunk_2 * 4 + 2]), "r"(col_inv_pair));
            unsigned int col_scaled2 = _bf16x2_mul_6;
            uint32_t _bf16x2_mul_7;
            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(col_values[chunk_2 * 4 + 3]), "r"(col_inv_pair));
            unsigned int col_scaled3 = _bf16x2_mul_7;
            bits0 = (col_scaled2 & 65535) << 16;
            bits1 = col_scaled2 & 4294901760;
            bits2 = (col_scaled3 & 65535) << 16;
            bits3 = col_scaled3 & 4294901760;
            value0 = reinterpret_cast<float*>(&bits0)[0];
            value1 = reinterpret_cast<float*>(&bits1)[0];
            value2 = reinterpret_cast<float*>(&bits2)[0];
            value3 = reinterpret_cast<float*>(&bits3)[0];
            col_scaled_quad[0] = value0;
            col_scaled_quad[1] = value1;
            col_scaled_quad[2] = value2;
            col_scaled_quad[3] = value3;
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(col_scaled_quad[0]), "f"(col_scaled_quad[1]),
                                       "f"(col_scaled_quad[2]), "f"(col_scaled_quad[3]));
                col_a23[(chunk_2) + 0] = _packed;
            }
            col_even[chunk_2] = __byte_perm(col_a01[chunk_2], col_a23[chunk_2], 0x6420);
            col_odd[chunk_2] = __byte_perm(col_a01[chunk_2], col_a23[chunk_2], 0x7531);
        }
        int col_local = cpr * 2;
        int col_byte = tq * 2 * 4;
        {
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((col_act_addr + (unsigned int)(stage * 2048) + (unsigned int)(col_local * 32 + col_byte))), "r"(col_even[0]), "r"(col_even[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((col_act_addr + (unsigned int)(stage * 2048) + (unsigned int)((col_local + 1) * 32 + col_byte))), "r"(col_odd[0]), "r"(col_odd[1]) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (warp == 0) {
            {
                tma_store_2d((&row_act_tma), bx * 64, by * 32 + stage * 32, row_act_addr + (unsigned int)(stage * 2048));
            }
            tma_store_2d((&col_act_tma), by * 32 + stage * 32, bx * 64, col_act_addr + (unsigned int)(stage * 2048));
            asm volatile("cp.async.bulk.commit_group;");
        }
    }
    if (warp == 0) {
        asm volatile("cp.async.bulk.wait_group.read 0;");
    }
}

} // extern "C"
