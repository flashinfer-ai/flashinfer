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
#define NUM_RAW_PIPE_STAGES 2
#define SMEM_SMEM_QD_OFF 1024
#define SMEM_SMEM_QD_STAGE_BYTES 4096
#define SMEM_SMEM_QD_STRIDE 4096
#define SMEM_SMEM_Q_RAW_OFF 27648
#define SMEM_SMEM_Q_RAW_STAGE_BYTES 4096
#define SMEM_SMEM_Q_RAW_STRIDE 4096
#define SMEM_SMEM_KD_OFF 5120
#define SMEM_SMEM_KD_STAGE_BYTES 4096
#define SMEM_SMEM_KD_STRIDE 4096
#define SMEM_SMEM_K_RAW_OFF 31744
#define SMEM_SMEM_K_RAW_STAGE_BYTES 4096
#define SMEM_SMEM_K_RAW_STRIDE 4096
#define SMEM_SMEM_KI_OFF 9216
#define SMEM_SMEM_KI_STAGE_BYTES 4096
#define SMEM_SMEM_KI_STRIDE 4096
#define SMEM_SMEM_GATE_RAW_OFF 35840
#define SMEM_SMEM_GATE_RAW_STAGE_BYTES 4096
#define SMEM_SMEM_GATE_RAW_STRIDE 4096
#define SMEM_SMEM_W_OUT_OFF 39936
#define SMEM_SMEM_W_OUT_STAGE_BYTES 4096
#define SMEM_SMEM_W_OUT_STRIDE 4096
#define SMEM_SMEM_KR_OFF 13312
#define SMEM_SMEM_KR_STAGE_BYTES 4096
#define SMEM_SMEM_KR_STRIDE 4096
#define SMEM_SMEM_QK_PLAIN_OFF 17408
#define SMEM_SMEM_QK_PLAIN_STAGE_BYTES 512
#define SMEM_SMEM_QK_PLAIN_STRIDE 512
#define SMEM_SMEM_BETA_RAW_OFF 26752
#define SMEM_SMEM_BETA_RAW_STAGE_BYTES 256
#define SMEM_SMEM_BETA_RAW_STRIDE 256
#define SMEM_SMEM_GATE_OFF 17920
#define SMEM_SMEM_GATE_STAGE_BYTES 8192
#define SMEM_SMEM_GATE_STRIDE 8192
#define SMEM_SMEM_BETA_OFF 26112
#define SMEM_SMEM_BETA_STAGE_BYTES 64
#define SMEM_SMEM_BETA_STRIDE 64
#define SMEM_SMEM_ABT_OFF 26176
#define SMEM_SMEM_ABT_STAGE_BYTES 512
#define SMEM_SMEM_ABT_STRIDE 512
#define SMEM_TOTAL 44032
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


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


__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(128, 5) void
kernel_flashkda_bf16_bt16_prepare(__nv_bfloat16* __restrict__ q, CakeTensorMap const* q_tma, __nv_bfloat16* __restrict__ k, CakeTensorMap const* k_tma, __nv_bfloat16* __restrict__ raw_gate, CakeTensorMap const* raw_gate_tma, __nv_bfloat16* __restrict__ beta_logits, CakeTensorMap const* beta_logits_tma, float* __restrict__ a_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, int* __restrict__ cu_chunks, int* __restrict__ chunk_to_seq, __nv_bfloat16* __restrict__ ws_qd, CakeTensorMap const* ws_qd_tma, __nv_bfloat16* __restrict__ ws_kd, CakeTensorMap const* ws_kd_tma, __nv_bfloat16* __restrict__ ws_w, CakeTensorMap const* ws_w_tma, __nv_bfloat16* __restrict__ ws_qk_t, float* __restrict__ ws_diag, int total_chunks, int num_heads, float gate_lower_bound)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (warp == 3) {
        uint64_t __cake_tensormap_acquire_addr = (uint64_t)(q_tma);
        if (lane == 1) __cake_tensormap_acquire_addr = (uint64_t)(k_tma);
        if (lane == 2) __cake_tensormap_acquire_addr = (uint64_t)(raw_gate_tma);
        if (lane == 3) __cake_tensormap_acquire_addr = (uint64_t)(beta_logits_tma);
        if (lane == 4) __cake_tensormap_acquire_addr = (uint64_t)(ws_qd_tma);
        if (lane == 5) __cake_tensormap_acquire_addr = (uint64_t)(ws_kd_tma);
        if (lane == 6) __cake_tensormap_acquire_addr = (uint64_t)(ws_w_tma);
        if (lane < 7) {
            asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"(__cake_tensormap_acquire_addr) : "memory");
        }
    }


    // Kernel setup ops
    __nv_bfloat16* smem_qd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_qd_addr = smem + 1024;
    __nv_bfloat16* smem_q_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 27648);
    const int smem_q_raw_addr = smem + 27648;
    __nv_bfloat16* smem_kd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 5120);
    const int smem_kd_addr = smem + 5120;
    __nv_bfloat16* smem_k_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 31744);
    const int smem_k_raw_addr = smem + 31744;
    __nv_bfloat16* smem_ki = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_ki_addr = smem + 9216;
    __nv_bfloat16* smem_gate_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 35840);
    const int smem_gate_raw_addr = smem + 35840;
    __nv_bfloat16* smem_w_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 39936);
    const int smem_w_out_addr = smem + 39936;
    __nv_bfloat16* smem_kr = reinterpret_cast<__nv_bfloat16*>(smem_raw + 13312);
    const int smem_kr_addr = smem + 13312;
    __nv_bfloat16* smem_qk_plain = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_qk_plain_addr = smem + 17408;
    __nv_bfloat16* smem_beta_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 26752);
    const int smem_beta_raw_addr = smem + 26752;
    float* smem_gate = reinterpret_cast<float*>(smem_raw + 17920);
    const int smem_gate_addr = smem + 17920;
    float* smem_beta = reinterpret_cast<float*>(smem_raw + 26112);
    const int smem_beta_addr = smem + 26112;
    __nv_bfloat16* smem_abt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 26176);
    const int smem_abt_addr = smem + 26176;

    // Mbarrier init (5 groups, 7 barriers)
    // Mbarriers at smem_raw[0..56)

    if (warp == 0) {
        // --- pipeline 'raw_pipe' ---
        // gate_raw_full: 2 barriers, init_count=1
        // qk_raw_full: 2 barriers, init_count=1
        // k_half_ready: 1 barriers, init_count=4
        // k_full_ready: 1 barriers, init_count=4
        // pairwise_ready: 1 barriers, init_count=2
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 4; _bar += 32) {
            mbarrier_init(smem + 0 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 2; _bar += 32) {
            mbarrier_init(smem + 32 + _bar * 8, 4);
        }
        for (int _bar = lane; _bar < 1; _bar += 32) {
            mbarrier_init(smem + 48 + _bar * 8, 2);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    const int mbar_base = smem;
    #define gate_raw_full_addr (mbar_base + 0)
    #define qk_raw_full_addr (mbar_base + 16)
    #define k_half_ready_addr (mbar_base + 32)
    #define k_full_ready_addr (mbar_base + 40)
    #define pairwise_ready_addr (mbar_base + 48)

    // === Task calls (dependency order) ===
    int linear_cta = blockIdx.x;
    int base_ctas_per_head = gridDim.x / num_heads;
    int extra_heads = gridDim.x % num_heads;
    int extra_span = (base_ctas_per_head + 1) * extra_heads;
    int head_idx = 0;
    int cta_rank_in_head = 0;
    int ctas_for_head = base_ctas_per_head;
    if (linear_cta < extra_span) {
        ctas_for_head = base_ctas_per_head + 1;
        head_idx = linear_cta / ctas_for_head;
        cta_rank_in_head = linear_cta % ctas_for_head;
    } else {
        head_idx = extra_heads + (linear_cta - extra_span) / ctas_for_head;
        cta_rank_in_head = (linear_cta - extra_span) % ctas_for_head;
    }
    int chunk_lo = cta_rank_in_head * total_chunks / ctas_for_head;
    int chunk_hi = (cta_rank_in_head + 1) * total_chunks / ctas_for_head;
    int my_chunks = chunk_hi - chunk_lo;
    int col = tid;
    float _exp2_0 = approx_exp2(a_log[head_idx] * 1.4426950408889634f);
    float gate_rate = _exp2_0;
    float gate_rate_half = gate_rate * 0.5f;
    float gate_bias = dt_bias[head_idx * 128 + col];
    float gate_half_scale = gate_lower_bound * 0.7213475204444817f;
    int raw_epoch = 0;
    unsigned int k_half_ready_phase = 0;
    unsigned int k_full_ready_phase = 0;
    unsigned int pairwise_ready_phase = 0;
    long long current_token_base = 0;
    long long current_eos = 0;
    if (my_chunks > 0) {
        int first_seq = chunk_to_seq[chunk_lo];
        int first_local_chunk = chunk_lo - cu_chunks[first_seq];
        long long first_bos = cu_seqlens[first_seq];
        current_eos = cu_seqlens[first_seq + 1];
        current_token_base = first_bos + (long long)(first_local_chunk * 16);
    }
    #pragma unroll 1
    for (int cta_chunk = 0; cta_chunk < my_chunks; cta_chunk++) {
        int gchunk = chunk_lo + cta_chunk;
        long long token_base = current_token_base;
        long long eos = current_eos;
        int chunk_is_full = ((eos >= token_base + 16) ? 1 : 0);
        unsigned int raw_stage = (unsigned int)raw_epoch & 1;
        unsigned int raw_phase = (unsigned int)raw_epoch / 2 & 1;
        if (cta_chunk == 0) {
            if (warp == 0) {
                if (elect_sync()) {
                    int gate_tx_bytes = 4096;
                    mbarrier_arrive_expect_tx(gate_raw_full_addr + (raw_stage) * 8, gate_tx_bytes);
                    tma_3d_gmem2smem(smem_gate_raw_addr, raw_gate_tma, 0, head_idx, (int)token_base, gate_raw_full_addr + (raw_stage) * 8);
                    mbarrier_arrive_expect_tx(qk_raw_full_addr + (raw_stage) * 8, 8192);
                    tma_4d_gmem2smem(smem_q_raw_addr, q_tma, 0, (int)token_base, head_idx, 0, qk_raw_full_addr + (raw_stage) * 8);
                    tma_4d_gmem2smem(smem_k_raw_addr, k_tma, 0, (int)token_base, head_idx, 0, qk_raw_full_addr + (raw_stage) * 8);
                }
            }
        }
        float beta_value = 0.0f;
        if (tid < 16) {
            {
                long long beta_token = token_base + (long long)tid;
                if (beta_token < eos) {
                    long long beta_index = beta_token * (long long)num_heads + (long long)head_idx;
                    float _tanh_approx_1;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"((float)beta_logits[beta_index] * 0.5f));
                    beta_value = _tanh_approx_1 * 0.5f + 0.5f;
                }
            }
        }
        mbarrier_wait(gate_raw_full_addr + (raw_stage) * 8, raw_phase);
        if (tid < 16) {
            smem_beta[tid] = beta_value;
        }
        if (chunk_is_full == 0) {
            mbarrier_wait(qk_raw_full_addr + (raw_stage) * 8, raw_phase);
            int tail_row = warp * 4 + lane / 8;
            int tail_lane_in_row = lane % 8;
            if (eos <= token_base + (long long)tail_row) {
                float tail_zero[8];
                tail_zero[0] = 0.0f;
                tail_zero[1] = 0.0f;
                tail_zero[2] = 0.0f;
                tail_zero[3] = 0.0f;
                tail_zero[4] = 0.0f;
                tail_zero[5] = 0.0f;
                tail_zero[6] = 0.0f;
                tail_zero[7] = 0.0f;
                #pragma unroll
                for (int dim_half = 0; dim_half < 2; dim_half++) {
                    int tail_segment = dim_half * 8 + tail_lane_in_row;
                    unsigned int packed[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tail_zero[_lp*2 + 0], tail_zero[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word = 0; word < 4; word++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_q_raw_addr + (unsigned int)(tail_segment * 8 / 64 * 2048 + tail_row * 128 + tail_segment * 8 % 64 * 2 ^ (tail_segment * 8 / 64 * 2048 + tail_row * 128 + tail_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word * 4)), "r"((packed[word])));
                    }
                    unsigned int packed_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tail_zero[_lp*2 + 0], tail_zero[_lp*2+1 + 0]));
                        packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_1 = 0; word_1 < 4; word_1++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_k_raw_addr + (unsigned int)(tail_segment * 8 / 64 * 2048 + tail_row * 128 + tail_segment * 8 % 64 * 2 ^ (tail_segment * 8 / 64 * 2048 + tail_row * 128 + tail_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_1 * 4)), "r"((packed_0[word_1])));
                    }
                }
            }
            #pragma unroll
            for (int tail_gate_row = 0; tail_gate_row < 16; tail_gate_row++) {
                if (eos <= token_base + (long long)tail_gate_row) {
                    smem_gate_raw[tail_gate_row * 128 + col] = 0.0f;
                }
            }
            __syncthreads();
        }
        float prefix_log2 = 0.0f;
        float gate_decay[16];
        if (chunk_is_full != 0) {
            #pragma unroll
            for (int row = 0; row < 16; row++) {
                float _cvt_f32_0 = __bfloat162float(smem_gate_raw[row * 128 + col]);
                float gate_raw_value = _cvt_f32_0;
                float gate_arg = gate_rate_half * (gate_raw_value + gate_bias);
                float _tanh_approx_3;
                asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_3) : "f"(gate_arg));
                float _fma_0 = __fmaf_rn(_tanh_approx_3, gate_half_scale, gate_half_scale);
                float gate_increment = _fma_0;
                prefix_log2 += gate_increment;
                gate_decay[row] = prefix_log2;
            }
        } else {
            #pragma unroll
            for (int row_1 = 0; row_1 < 16; row_1++) {
                float gate_increment_1 = 0.0f;
                if (eos > token_base + (long long)row_1) {
                    float _cvt_f32_1 = __bfloat162float(smem_gate_raw[row_1 * 128 + col]);
                    float gate_raw_value_1 = _cvt_f32_1;
                    float gate_arg_1 = gate_rate_half * (gate_raw_value_1 + gate_bias);
                    float _tanh_approx_4;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_4) : "f"(gate_arg_1));
                    float _fma_1 = __fmaf_rn(_tanh_approx_4, gate_half_scale, gate_half_scale);
                    gate_increment_1 = _fma_1;
                }
                prefix_log2 += gate_increment_1;
                gate_decay[row_1] = prefix_log2;
            }
        }
        #pragma unroll
        for (int row_2 = 0; row_2 < 16; row_2++) {
            float _exp2_1 = approx_exp2(gate_decay[row_2]);
            gate_decay[row_2] = _exp2_1;
            smem_gate[row_2 * 128 + col] = gate_decay[row_2];
        }
        float total_decay = gate_decay[15];
        smem_gate[1920 + col] = total_decay;
        ws_diag[((long long)head_idx * (long long)total_chunks + (long long)gchunk) * 128 + (long long)col] = total_decay;
        __syncthreads();
        if (chunk_is_full != 0) {
            mbarrier_wait(qk_raw_full_addr + (raw_stage) * 8, raw_phase);
        }
        int row_3 = warp * 4 + lane / 8;
        int lane_in_row = lane % 8;
        float q_raw[16];
        float k_raw[16];
        unsigned int q_raw_packed[8];
        unsigned int k_raw_packed[8];
        q_raw[0] = 0.0f;
        q_raw[1] = 0.0f;
        q_raw[2] = 0.0f;
        q_raw[3] = 0.0f;
        q_raw[4] = 0.0f;
        q_raw[5] = 0.0f;
        q_raw[6] = 0.0f;
        q_raw[7] = 0.0f;
        q_raw[8] = 0.0f;
        q_raw[9] = 0.0f;
        q_raw[10] = 0.0f;
        q_raw[11] = 0.0f;
        q_raw[12] = 0.0f;
        q_raw[13] = 0.0f;
        q_raw[14] = 0.0f;
        q_raw[15] = 0.0f;
        k_raw[0] = 0.0f;
        k_raw[1] = 0.0f;
        k_raw[2] = 0.0f;
        k_raw[3] = 0.0f;
        k_raw[4] = 0.0f;
        k_raw[5] = 0.0f;
        k_raw[6] = 0.0f;
        k_raw[7] = 0.0f;
        k_raw[8] = 0.0f;
        k_raw[9] = 0.0f;
        k_raw[10] = 0.0f;
        k_raw[11] = 0.0f;
        k_raw[12] = 0.0f;
        k_raw[13] = 0.0f;
        k_raw[14] = 0.0f;
        k_raw[15] = 0.0f;
        #pragma unroll
        for (int dim_half_1 = 0; dim_half_1 < 2; dim_half_1++) {
            int segment = dim_half_1 * 8 + lane_in_row;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&q_raw_packed[dim_half_1 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&q_raw_packed[(dim_half_1 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_raw_packed[(dim_half_1 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_raw_packed[(dim_half_1 * 4) + 3]))
                : "r"((smem_q_raw_addr + (unsigned int)(segment * 8 / 64 * 2048 + row_3 * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row_3 * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&k_raw_packed[dim_half_1 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&k_raw_packed[(dim_half_1 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_raw_packed[(dim_half_1 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_raw_packed[(dim_half_1 * 4) + 3]))
                : "r"((smem_k_raw_addr + (unsigned int)(segment * 8 / 64 * 2048 + row_3 * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row_3 * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
        }
        float q_raw_packed_f32[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&q_raw_packed_f32[_pair * 2])[0]), "=f"((&q_raw_packed_f32[_pair * 2])[1])
                : "r"(q_raw_packed[_pair]));
        }
        float k_raw_packed_f32[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&k_raw_packed_f32[_pair * 2])[0]), "=f"((&k_raw_packed_f32[_pair * 2])[1])
                : "r"(k_raw_packed[_pair]));
        }
        #pragma unroll
        for (int elem = 0; elem < 16; elem++) {
            q_raw[elem] = q_raw_packed_f32[elem];
            k_raw[elem] = k_raw_packed_f32[elem];
        }
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        #pragma unroll
        for (int elem_pair = 0; elem_pair < 8; elem_pair++) {
            float _bf16x2_dot_f32_0;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_bf16x2_dot_f32_0) : "r"(q_raw_packed[elem_pair]), "r"(q_raw_packed[elem_pair]), "f"(q_sum));
            q_sum = _bf16x2_dot_f32_0;
            float _bf16x2_dot_f32_1;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_bf16x2_dot_f32_1) : "r"(k_raw_packed[elem_pair]), "r"(k_raw_packed[elem_pair]), "f"(k_sum));
            k_sum = _bf16x2_dot_f32_1;
        }
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 4);
        q_sum += _shfl_xor_0;
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 4);
        k_sum += _shfl_xor_1;
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 2);
        q_sum += _shfl_xor_2;
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 2);
        k_sum += _shfl_xor_3;
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 1);
        q_sum += _shfl_xor_4;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 1);
        k_sum += _shfl_xor_5;
        float _max_0 = max_noftz(q_sum, 1e-24f);
        float _rsqrt_0 = rsqrtf(_max_0);
        float q_inv = _rsqrt_0;
        float _max_1 = max_noftz(k_sum, 1e-24f);
        float _rsqrt_1 = rsqrtf(_max_1);
        float k_inv_norm = _rsqrt_1;
        __syncthreads();
        raw_epoch += 1;
        if (my_chunks > cta_chunk + 1) {
            int next_gchunk = gchunk + 1;
            int next_seq = chunk_to_seq[next_gchunk];
            int next_local_chunk = next_gchunk - cu_chunks[next_seq];
            long long next_bos = cu_seqlens[next_seq];
            long long next_eos = cu_seqlens[next_seq + 1];
            long long next_token_base = next_bos + (long long)(next_local_chunk * 16);
            unsigned int next_raw_stage = (unsigned int)raw_epoch & 1;
            if (warp == 0) {
                if (elect_sync()) {
                    int gate_tx_bytes_1 = 4096;
                    mbarrier_arrive_expect_tx(gate_raw_full_addr + (next_raw_stage) * 8, gate_tx_bytes_1);
                    tma_3d_gmem2smem(smem_gate_raw_addr, raw_gate_tma, 0, head_idx, (int)next_token_base, gate_raw_full_addr + (next_raw_stage) * 8);
                    mbarrier_arrive_expect_tx(qk_raw_full_addr + (next_raw_stage) * 8, 8192);
                    tma_4d_gmem2smem(smem_q_raw_addr, q_tma, 0, (int)next_token_base, head_idx, 0, qk_raw_full_addr + (next_raw_stage) * 8);
                    tma_4d_gmem2smem(smem_k_raw_addr, k_tma, 0, (int)next_token_base, head_idx, 0, qk_raw_full_addr + (next_raw_stage) * 8);
                }
            }
            current_token_base = next_token_base;
            current_eos = next_eos;
        }
        float kk_acc[8];
        #pragma unroll
        for (int dim_half_2 = 0; dim_half_2 < 2; dim_half_2++) {
            int segment_1 = dim_half_2 * 8 + lane_in_row;
            int reg_base = dim_half_2 * 8;
            float qd_values[8];
            float kd_values[8];
            float ki_values[8];
            float kr_values[8];
            float2 _f2_0 = make_float2(q_inv, q_inv);
            float2 q_norm_pair = _f2_0;
            float2 _f2_1 = make_float2(k_inv_norm, k_inv_norm);
            float2 k_norm_pair = _f2_1;
            #pragma unroll
            for (int elem_pair_1 = 0; elem_pair_1 < 4; elem_pair_1++) {
                int elem0 = elem_pair_1 * 2;
                int elem1 = elem0 + 1;
                int this_col0 = segment_1 * 8 + elem0;
                int this_col1 = this_col0 + 1;
                float decay0 = smem_gate[row_3 * 128 + this_col0];
                float decay1 = smem_gate[row_3 * 128 + this_col1];
                float _rcp_0 = approx_rcp(decay0);
                float inv_decay0 = _rcp_0;
                float _rcp_1 = approx_rcp(decay1);
                float inv_decay1 = _rcp_1;
                float2 _f2_2 = make_float2(q_raw[reg_base + elem0], q_raw[reg_base + elem1]);
                float2 raw_q_pair = _f2_2;
                float2 _f2_3 = make_float2(k_raw[reg_base + elem0], k_raw[reg_base + elem1]);
                float2 raw_k_pair = _f2_3;
                float2 q_value_pair = mul_f32x2(raw_q_pair, q_norm_pair);
                float2 k_value_pair = mul_f32x2(raw_k_pair, k_norm_pair);
                float2 _f2_4 = make_float2(decay0, decay1);
                float2 decay_pair = _f2_4;
                float2 _f2_5 = make_float2(inv_decay0, inv_decay1);
                float2 inv_decay_pair = _f2_5;
                float2 qd_pair = mul_f32x2(q_value_pair, decay_pair);
                float2 kd_pair = mul_f32x2(k_value_pair, decay_pair);
                float2 ki_pair = mul_f32x2(k_value_pair, inv_decay_pair);
                qd_values[elem0] = qd_pair.x;
                qd_values[elem1] = qd_pair.y;
                kd_values[elem0] = kd_pair.x;
                kd_values[elem1] = kd_pair.y;
                ki_values[elem0] = ki_pair.x;
                ki_values[elem1] = ki_pair.y;
                float total_decay0 = smem_gate[1920 + this_col0];
                float total_decay1 = smem_gate[1920 + this_col1];
                float2 _f2_6 = make_float2(total_decay0, total_decay1);
                float2 kr_pair = mul_f32x2(ki_pair, _f2_6);
                kr_values[elem0] = kr_pair.x;
                kr_values[elem1] = kr_pair.y;
            }
            unsigned int packed_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_values[_lp*2 + 0], qd_values[_lp*2+1 + 0]));
                packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int word_2 = 0; word_2 < 4; word_2++) {
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + (unsigned int)(segment_1 * 8 / 64 * 2048 + row_3 * 128 + segment_1 * 8 % 64 * 2 ^ (segment_1 * 8 / 64 * 2048 + row_3 * 128 + segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_2 * 4)), "r"((packed_1[word_2])));
            }
            unsigned int packed_0_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kd_values[_lp*2 + 0], kd_values[_lp*2+1 + 0]));
                packed_0_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int word_3 = 0; word_3 < 4; word_3++) {
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + (unsigned int)(segment_1 * 8 / 64 * 2048 + row_3 * 128 + segment_1 * 8 % 64 * 2 ^ (segment_1 * 8 / 64 * 2048 + row_3 * 128 + segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_3 * 4)), "r"((packed_0_1[word_3])));
            }
            unsigned int packed_1_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_values[_lp*2 + 0], ki_values[_lp*2+1 + 0]));
                packed_1_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int word_4 = 0; word_4 < 4; word_4++) {
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_ki_addr + (unsigned int)(segment_1 * 8 / 64 * 2048 + row_3 * 128 + segment_1 * 8 % 64 * 2 ^ (segment_1 * 8 / 64 * 2048 + row_3 * 128 + segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_4 * 4)), "r"((packed_1_1[word_4])));
            }
            unsigned int packed_2[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kr_values[_lp*2 + 0], kr_values[_lp*2+1 + 0]));
                packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int word_5 = 0; word_5 < 4; word_5++) {
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_addr + (unsigned int)(segment_1 * 8 / 64 * 2048 + row_3 * 128 + segment_1 * 8 % 64 * 2 ^ (segment_1 * 8 / 64 * 2048 + row_3 * 128 + segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_5 * 4)), "r"((packed_2[word_5])));
            }
            if (dim_half_2 == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(k_half_ready_addr);
                }
                if (warp == 0) {
                    mbarrier_wait(k_half_ready_addr, k_half_ready_phase);
                    unsigned int a_frag[4];
                    unsigned int b_frag[4];
                    for (int mma_d = 0; mma_d < 4; mma_d++) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"((smem_kd_addr + (unsigned int)((mma_d * 16 + lane / 16 * 8) / 64 * 2048 + lane % 16 * 128 + (mma_d * 16 + lane / 16 * 8) % 64 * 2 ^ ((mma_d * 16 + lane / 16 * 8) / 64 * 2048 + lane % 16 * 128 + (mma_d * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"((smem_ki_addr + (unsigned int)((mma_d * 16 + lane % 16 / 8 * 8) / 64 * 2048 + (8 * (lane / 16) + lane % 8) * 128 + (mma_d * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((mma_d * 16 + lane % 16 / 8 * 8) / 64 * 2048 + (8 * (lane / 16) + lane % 8) * 128 + (mma_d * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(kk_acc[0]), "=f"(kk_acc[1]), "=f"(kk_acc[2]), "=f"(kk_acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(((((mma_d == 0) ? 1 : 0)) ? 0.0f : kk_acc[0])), "f"(((((mma_d == 0) ? 1 : 0)) ? 0.0f : kk_acc[1])), "f"(((((mma_d == 0) ? 1 : 0)) ? 0.0f : kk_acc[2])), "f"(((((mma_d == 0) ? 1 : 0)) ? 0.0f : kk_acc[3])));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(kk_acc[4]), "=f"(kk_acc[(4) + 1]), "=f"(kk_acc[(4) + 2]), "=f"(kk_acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(((((mma_d == 0) ? 1 : 0)) ? 0.0f : kk_acc[4])), "f"(((((mma_d == 0) ? 1 : 0)) ? 0.0f : kk_acc[(4) + 1])), "f"(((((mma_d == 0) ? 1 : 0)) ? 0.0f : kk_acc[(4) + 2])), "f"(((((mma_d == 0) ? 1 : 0)) ? 0.0f : kk_acc[(4) + 3])));
                    }
                }
            }
        }
        if (elect_sync()) {
            mbarrier_arrive(k_full_ready_addr);
        }
        if (warp == 3) {
            mbarrier_wait(k_full_ready_addr, k_full_ready_phase);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (elect_sync()) {
                tma_store_4d(ws_qd_tma, 0, gchunk * 16, head_idx, 0, smem_qd_addr);
                tma_store_4d(ws_kd_tma, 0, gchunk * 16, head_idx, 0, smem_kd_addr);
            }
            asm volatile("cp.async.bulk.commit_group;");
        }
        if (warp == 0) {
            mbarrier_wait(k_full_ready_addr, k_full_ready_phase);
            unsigned int a_frag_1[4];
            unsigned int b_frag_1[4];
            for (int mma_d_1 = 0; mma_d_1 < 4; mma_d_1++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                    : "r"((smem_kd_addr + (unsigned int)((64 + mma_d_1 * 16 + lane / 16 * 8) / 64 * 2048 + lane % 16 * 128 + (64 + mma_d_1 * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + mma_d_1 * 16 + lane / 16 * 8) / 64 * 2048 + lane % 16 * 128 + (64 + mma_d_1 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_1[0]), "=r"(b_frag_1[1]), "=r"(b_frag_1[2]), "=r"(b_frag_1[3])
                    : "r"((smem_ki_addr + (unsigned int)((64 + mma_d_1 * 16 + lane % 16 / 8 * 8) / 64 * 2048 + (8 * (lane / 16) + lane % 8) * 128 + (64 + mma_d_1 * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((64 + mma_d_1 * 16 + lane % 16 / 8 * 8) / 64 * 2048 + (8 * (lane / 16) + lane % 8) * 128 + (64 + mma_d_1 * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(kk_acc[0]), "=f"(kk_acc[1]), "=f"(kk_acc[2]), "=f"(kk_acc[3])
                    : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "f"(((((mma_d_1 == 0) ? 0 : 0)) ? 0.0f : kk_acc[0])), "f"(((((mma_d_1 == 0) ? 0 : 0)) ? 0.0f : kk_acc[1])), "f"(((((mma_d_1 == 0) ? 0 : 0)) ? 0.0f : kk_acc[2])), "f"(((((mma_d_1 == 0) ? 0 : 0)) ? 0.0f : kk_acc[3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(kk_acc[4]), "=f"(kk_acc[(4) + 1]), "=f"(kk_acc[(4) + 2]), "=f"(kk_acc[(4) + 3])
                    : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"(b_frag_1[2]), "r"(b_frag_1[(2) + 1]), "f"(((((mma_d_1 == 0) ? 0 : 0)) ? 0.0f : kk_acc[4])), "f"(((((mma_d_1 == 0) ? 0 : 0)) ? 0.0f : kk_acc[(4) + 1])), "f"(((((mma_d_1 == 0) ? 0 : 0)) ? 0.0f : kk_acc[(4) + 2])), "f"(((((mma_d_1 == 0) ? 0 : 0)) ? 0.0f : kk_acc[(4) + 3])));
            }
            float inverse_values[8];
            int row0 = lane / 4;
            int row1 = row0 + 8;
            int col0 = lane % 4 * 2;
            float beta0 = smem_beta[row0];
            float beta1 = smem_beta[row1];
            float l_values[8];
            l_values[0] = 0.0f;
            l_values[1] = 0.0f;
            l_values[2] = 0.0f;
            l_values[3] = 0.0f;
            l_values[4] = 0.0f;
            l_values[5] = 0.0f;
            l_values[6] = 0.0f;
            l_values[7] = 0.0f;
            if (row0 > col0) {
                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(kk_acc[0] * beta0);
                float _cvt_f32_2 = __bfloat162float(_cvt_bf16_0);
                l_values[0] = _cvt_f32_2;
            }
            if (row0 > col0 + 1) {
                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(kk_acc[1] * beta0);
                float _cvt_f32_3 = __bfloat162float(_cvt_bf16_1);
                l_values[1] = _cvt_f32_3;
            }
            if (row1 > col0) {
                __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(kk_acc[2] * beta1);
                float _cvt_f32_4 = __bfloat162float(_cvt_bf16_2);
                l_values[2] = _cvt_f32_4;
            }
            if (row1 > col0 + 1) {
                __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(kk_acc[3] * beta1);
                float _cvt_f32_5 = __bfloat162float(_cvt_bf16_3);
                l_values[3] = _cvt_f32_5;
            }
            if (row0 > col0 + 8) {
                __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(kk_acc[4] * beta0);
                float _cvt_f32_6 = __bfloat162float(_cvt_bf16_4);
                l_values[4] = _cvt_f32_6;
            }
            if (row0 > col0 + 9) {
                __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(kk_acc[5] * beta0);
                float _cvt_f32_7 = __bfloat162float(_cvt_bf16_5);
                l_values[5] = _cvt_f32_7;
            }
            if (row1 > col0 + 8) {
                __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(kk_acc[6] * beta1);
                float _cvt_f32_8 = __bfloat162float(_cvt_bf16_6);
                l_values[6] = _cvt_f32_8;
            }
            if (row1 > col0 + 9) {
                __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(kk_acc[7] * beta1);
                float _cvt_f32_9 = __bfloat162float(_cvt_bf16_7);
                l_values[7] = _cvt_f32_9;
            }
            unsigned int l_frag[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(l_values[_lp*2 + 0], l_values[_lp*2+1 + 0]));
                l_frag[_lp] = *(uint32_t*)&_h2;
            }
            unsigned int d_frag[4];
            d_frag[0] = 0;
            d_frag[1] = 0;
            d_frag[2] = 0;
            d_frag[3] = 0;
            d_frag[0] = l_frag[0];
            d_frag[3] = l_frag[3];
            float diagonal_values[4];
            diagonal_values[0] = -l_values[0];
            diagonal_values[1] = -l_values[1];
            diagonal_values[2] = -l_values[6];
            diagonal_values[3] = -l_values[7];
            if (row0 == col0) {
                diagonal_values[0] = diagonal_values[0] + 1.0f;
            }
            if (row0 == col0 + 1) {
                diagonal_values[1] = diagonal_values[1] + 1.0f;
            }
            if (row1 == col0 + 8) {
                diagonal_values[2] = diagonal_values[2] + 1.0f;
            }
            if (row1 == col0 + 9) {
                diagonal_values[3] = diagonal_values[3] + 1.0f;
            }
            unsigned int diagonal_frag[4];
            unsigned int low_word[1];
            unsigned int high_word[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(diagonal_values[_lp*2 + 0], diagonal_values[_lp*2+1 + 0]));
                low_word[_lp] = *(uint32_t*)&_h2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(diagonal_values[_lp*2 + 2], diagonal_values[_lp*2+1 + 2]));
                high_word[_lp] = *(uint32_t*)&_h2;
            }
            diagonal_frag[0] = 0;
            diagonal_frag[1] = 0;
            diagonal_frag[2] = 0;
            diagonal_frag[3] = 0;
            diagonal_frag[0] = low_word[0];
            diagonal_frag[3] = high_word[0];
            float diagonal_product[4];
            unsigned int rhs_trans_frag[2];
            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                : "=r"(rhs_trans_frag[0])
                : "r"(d_frag[0]));
            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                : "=r"(rhs_trans_frag[1])
                : "r"(d_frag[3]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(diagonal_product[0]), "=f"(diagonal_product[1]), "=f"(diagonal_product[2]), "=f"(diagonal_product[3])
                : "r"(d_frag[0]), "r"(d_frag[1]), "r"(d_frag[2]), "r"(d_frag[3]), "r"(rhs_trans_frag[0]), "r"(rhs_trans_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            unsigned int d2_frag[4];
            unsigned int low_word_0[1];
            unsigned int high_word_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(diagonal_product[_lp*2 + 0], diagonal_product[_lp*2+1 + 0]));
                low_word_0[_lp] = *(uint32_t*)&_h2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(diagonal_product[_lp*2 + 2], diagonal_product[_lp*2+1 + 2]));
                high_word_1[_lp] = *(uint32_t*)&_h2;
            }
            d2_frag[0] = 0;
            d2_frag[1] = 0;
            d2_frag[2] = 0;
            d2_frag[3] = 0;
            d2_frag[0] = low_word_0[0];
            d2_frag[3] = high_word_1[0];
            unsigned int rhs_trans_frag_2[2];
            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                : "=r"(rhs_trans_frag_2[0])
                : "r"(d2_frag[0]));
            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                : "=r"(rhs_trans_frag_2[1])
                : "r"(d2_frag[3]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(diagonal_product[0]), "=f"(diagonal_product[1]), "=f"(diagonal_product[2]), "=f"(diagonal_product[3])
                : "r"(diagonal_frag[0]), "r"(diagonal_frag[1]), "r"(diagonal_frag[2]), "r"(diagonal_frag[3]), "r"(rhs_trans_frag_2[0]), "r"(rhs_trans_frag_2[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            #pragma unroll
            for (int value_idx = 0; value_idx < 4; value_idx++) {
                diagonal_values[value_idx] = diagonal_values[value_idx] + diagonal_product[value_idx];
            }
            unsigned int low_word_3[1];
            unsigned int high_word_4[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(diagonal_values[_lp*2 + 0], diagonal_values[_lp*2+1 + 0]));
                low_word_3[_lp] = *(uint32_t*)&_h2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(diagonal_values[_lp*2 + 2], diagonal_values[_lp*2+1 + 2]));
                high_word_4[_lp] = *(uint32_t*)&_h2;
            }
            diagonal_frag[0] = 0;
            diagonal_frag[1] = 0;
            diagonal_frag[2] = 0;
            diagonal_frag[3] = 0;
            diagonal_frag[0] = low_word_3[0];
            diagonal_frag[3] = high_word_4[0];
            unsigned int rhs_trans_frag_5[2];
            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                : "=r"(rhs_trans_frag_5[0])
                : "r"(d2_frag[0]));
            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                : "=r"(rhs_trans_frag_5[1])
                : "r"(d2_frag[3]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(diagonal_product[0]), "=f"(diagonal_product[1]), "=f"(diagonal_product[2]), "=f"(diagonal_product[3])
                : "r"(d2_frag[0]), "r"(d2_frag[1]), "r"(d2_frag[2]), "r"(d2_frag[3]), "r"(rhs_trans_frag_5[0]), "r"(rhs_trans_frag_5[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            unsigned int d4_frag[4];
            unsigned int low_word_6[1];
            unsigned int high_word_7[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(diagonal_product[_lp*2 + 0], diagonal_product[_lp*2+1 + 0]));
                low_word_6[_lp] = *(uint32_t*)&_h2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(diagonal_product[_lp*2 + 2], diagonal_product[_lp*2+1 + 2]));
                high_word_7[_lp] = *(uint32_t*)&_h2;
            }
            d4_frag[0] = 0;
            d4_frag[1] = 0;
            d4_frag[2] = 0;
            d4_frag[3] = 0;
            d4_frag[0] = low_word_6[0];
            d4_frag[3] = high_word_7[0];
            unsigned int rhs_trans_frag_8[2];
            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                : "=r"(rhs_trans_frag_8[0])
                : "r"(d4_frag[0]));
            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                : "=r"(rhs_trans_frag_8[1])
                : "r"(d4_frag[3]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(diagonal_product[0]), "=f"(diagonal_product[1]), "=f"(diagonal_product[2]), "=f"(diagonal_product[3])
                : "r"(diagonal_frag[0]), "r"(diagonal_frag[1]), "r"(diagonal_frag[2]), "r"(diagonal_frag[3]), "r"(rhs_trans_frag_8[0]), "r"(rhs_trans_frag_8[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            #pragma unroll
            for (int value_idx_1 = 0; value_idx_1 < 4; value_idx_1++) {
                diagonal_values[value_idx_1] = diagonal_values[value_idx_1] + diagonal_product[value_idx_1];
            }
            unsigned int binv_frag[4];
            binv_frag[0] = 0;
            binv_frag[1] = 0;
            binv_frag[2] = 0;
            binv_frag[3] = 0;
            binv_frag[0] = diagonal_frag[0];
            binv_frag[3] = diagonal_frag[3];
            unsigned int a21_frag[4];
            a21_frag[0] = 0;
            a21_frag[1] = 0;
            a21_frag[2] = 0;
            a21_frag[3] = 0;
            a21_frag[1] = l_frag[1];
            float correction_product[4];
            unsigned int rhs_trans_frag_9[2];
            rhs_trans_frag_9[0] = 0;
            rhs_trans_frag_9[1] = 0;
            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                : "=r"(rhs_trans_frag_9[1])
                : "r"(a21_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(correction_product[0]), "=f"(correction_product[1]), "=f"(correction_product[2]), "=f"(correction_product[3])
                : "r"(binv_frag[0]), "r"(binv_frag[1]), "r"(binv_frag[2]), "r"(binv_frag[3]), "r"(rhs_trans_frag_9[0]), "r"(rhs_trans_frag_9[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            float correction_values[2];
            correction_values[0] = -correction_product[2];
            correction_values[1] = -correction_product[3];
            unsigned int correction_word[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(correction_values[_lp*2 + 0], correction_values[_lp*2+1 + 0]));
                correction_word[_lp] = *(uint32_t*)&_h2;
            }
            unsigned int correction_frag[4];
            correction_frag[0] = 0;
            correction_frag[1] = 0;
            correction_frag[2] = 0;
            correction_frag[3] = 0;
            correction_frag[1] = correction_word[0];
            unsigned int rhs_trans_frag_10[2];
            rhs_trans_frag_10[0] = 0;
            rhs_trans_frag_10[1] = 0;
            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                : "=r"(rhs_trans_frag_10[0])
                : "r"(binv_frag[0]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(correction_product[0]), "=f"(correction_product[1]), "=f"(correction_product[2]), "=f"(correction_product[3])
                : "r"(correction_frag[0]), "r"(correction_frag[1]), "r"(correction_frag[2]), "r"(correction_frag[3]), "r"(rhs_trans_frag_10[0]), "r"(rhs_trans_frag_10[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            float n_values[8];
            n_values[0] = 0.0f;
            n_values[1] = 0.0f;
            n_values[2] = 0.0f;
            n_values[3] = 0.0f;
            n_values[4] = 0.0f;
            n_values[5] = 0.0f;
            n_values[6] = 0.0f;
            n_values[7] = 0.0f;
            n_values[0] = diagonal_values[0];
            n_values[1] = diagonal_values[1];
            n_values[2] = correction_product[2];
            n_values[3] = correction_product[3];
            n_values[6] = diagonal_values[2];
            n_values[7] = diagonal_values[3];
            #pragma unroll
            for (int value_idx_2 = 0; value_idx_2 < 8; value_idx_2++) {
                inverse_values[value_idx_2] = n_values[value_idx_2];
            }
            unsigned int inverse_frag[4];
            unsigned int inverse_trans_frag[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inverse_values[_lp*2 + 0], inverse_values[_lp*2+1 + 0]));
                inverse_frag[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int inverse_word = 0; inverse_word < 4; inverse_word++) {
                asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                    : "=r"(inverse_trans_frag[inverse_word])
                    : "r"(inverse_frag[inverse_word]));
            }
            float inverse_trans_frag_f32[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&inverse_trans_frag_f32[_pair * 2])[0]), "=f"((&inverse_trans_frag_f32[_pair * 2])[1])
                    : "r"(inverse_trans_frag[_pair]));
            }
            int abt_row0 = lane / 4;
            int abt_row1 = abt_row0 + 8;
            float abt_beta0 = smem_beta[abt_row0];
            float abt_beta1 = smem_beta[abt_row1];
            #pragma unroll
            for (int abt_elem = 0; abt_elem < 8; abt_elem++) {
                float abt_beta = abt_beta0;
                if (abt_elem >= 2 && abt_elem < 4) {
                    abt_beta = abt_beta1;
                }
                if (abt_elem >= 6) {
                    abt_beta = abt_beta1;
                }
                inverse_trans_frag_f32[abt_elem] = inverse_trans_frag_f32[abt_elem] * abt_beta;
            }
            unsigned int packed_3[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inverse_trans_frag_f32[_lp*2 + 0], inverse_trans_frag_f32[_lp*2+1 + 0]));
                packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            int lane_row = lane % 16;
            int lane_col = lane / 16 * 8;
            uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(smem_abt_addr + (unsigned int)(lane_col / 16 * 512 + lane_row * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 512 + lane_row * 32 + lane_col % 16 * 2 >> 7 & 1) << 4)));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[3]))
                : "memory");
            if (elect_sync()) {
                mbarrier_arrive(pairwise_ready_addr);
            }
        }
        if (warp == 2) {
            mbarrier_wait(k_full_ready_addr, k_full_ready_phase);
            unsigned int qk_a_frag[4];
            unsigned int qk_b_frag[4];
            float qk_acc[8];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_a_frag[0]), "=r"(qk_a_frag[1]), "=r"(qk_a_frag[2]), "=r"(qk_a_frag[3])
                : "r"(smem_qd_addr + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_b_frag[0]), "=r"(qk_b_frag[1]), "=r"(qk_b_frag[2]), "=r"(qk_b_frag[3])
                : "r"(smem_ki_addr + (unsigned int)((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[0]), "=f"(qk_acc[1]), "=f"(qk_acc[2]), "=f"(qk_acc[3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[0]), "r"(qk_b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[4]), "=f"(qk_acc[(4) + 1]), "=f"(qk_acc[(4) + 2]), "=f"(qk_acc[(4) + 3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[2]), "r"(qk_b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_a_frag[0]), "=r"(qk_a_frag[1]), "=r"(qk_a_frag[2]), "=r"(qk_a_frag[3])
                : "r"(smem_qd_addr + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_b_frag[0]), "=r"(qk_b_frag[1]), "=r"(qk_b_frag[2]), "=r"(qk_b_frag[3])
                : "r"(smem_ki_addr + (unsigned int)(((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[0]), "r"(qk_b_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[2]), "r"(qk_b_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_a_frag[0]), "=r"(qk_a_frag[1]), "=r"(qk_a_frag[2]), "=r"(qk_a_frag[3])
                : "r"(smem_qd_addr + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_b_frag[0]), "=r"(qk_b_frag[1]), "=r"(qk_b_frag[2]), "=r"(qk_b_frag[3])
                : "r"(smem_ki_addr + (unsigned int)((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[0]), "r"(qk_b_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[2]), "r"(qk_b_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_a_frag[0]), "=r"(qk_a_frag[1]), "=r"(qk_a_frag[2]), "=r"(qk_a_frag[3])
                : "r"(smem_qd_addr + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_b_frag[0]), "=r"(qk_b_frag[1]), "=r"(qk_b_frag[2]), "=r"(qk_b_frag[3])
                : "r"(smem_ki_addr + (unsigned int)(((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[0]), "r"(qk_b_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[2]), "r"(qk_b_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_a_frag[0]), "=r"(qk_a_frag[1]), "=r"(qk_a_frag[2]), "=r"(qk_a_frag[3])
                : "r"(smem_qd_addr + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128) * 16))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_b_frag[0]), "=r"(qk_b_frag[1]), "=r"(qk_b_frag[2]), "=r"(qk_b_frag[3])
                : "r"(smem_ki_addr + (unsigned int)((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[0]), "r"(qk_b_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[2]), "r"(qk_b_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_a_frag[0]), "=r"(qk_a_frag[1]), "=r"(qk_a_frag[2]), "=r"(qk_a_frag[3])
                : "r"(smem_qd_addr + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2) * 16))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_b_frag[0]), "=r"(qk_b_frag[1]), "=r"(qk_b_frag[2]), "=r"(qk_b_frag[3])
                : "r"(smem_ki_addr + (unsigned int)(((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[0]), "r"(qk_b_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[2]), "r"(qk_b_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_a_frag[0]), "=r"(qk_a_frag[1]), "=r"(qk_a_frag[2]), "=r"(qk_a_frag[3])
                : "r"(smem_qd_addr + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2 ^ 6) * 16))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_b_frag[0]), "=r"(qk_b_frag[1]), "=r"(qk_b_frag[2]), "=r"(qk_b_frag[3])
                : "r"(smem_ki_addr + (unsigned int)((((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[0]), "r"(qk_b_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[2]), "r"(qk_b_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_a_frag[0]), "=r"(qk_a_frag[1]), "=r"(qk_a_frag[2]), "=r"(qk_a_frag[3])
                : "r"(smem_qd_addr + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2 ^ 6 ^ 2) * 16))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(qk_b_frag[0]), "=r"(qk_b_frag[1]), "=r"(qk_b_frag[2]), "=r"(qk_b_frag[3])
                : "r"(smem_ki_addr + (unsigned int)(((((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[0]), "r"(qk_b_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(qk_a_frag[0]), "r"(qk_a_frag[1]), "r"(qk_a_frag[2]), "r"(qk_a_frag[3]), "r"(qk_b_frag[2]), "r"(qk_b_frag[(2) + 1]));
            int row0_1 = lane / 4;
            int row1_1 = row0_1 + 8;
            int col0_1 = lane % 4 * 2;
            float qk_values[8];
            qk_values[0] = 0.0f;
            qk_values[1] = 0.0f;
            qk_values[2] = 0.0f;
            qk_values[3] = 0.0f;
            qk_values[4] = 0.0f;
            qk_values[5] = 0.0f;
            qk_values[6] = 0.0f;
            qk_values[7] = 0.0f;
            if (row0_1 >= col0_1) {
                qk_values[0] = qk_acc[0];
            }
            if (row0_1 >= col0_1 + 1) {
                qk_values[1] = qk_acc[1];
            }
            if (row1_1 >= col0_1) {
                qk_values[2] = qk_acc[2];
            }
            if (row1_1 >= col0_1 + 1) {
                qk_values[3] = qk_acc[3];
            }
            if (row0_1 >= col0_1 + 8) {
                qk_values[4] = qk_acc[4];
            }
            if (row0_1 >= col0_1 + 9) {
                qk_values[5] = qk_acc[5];
            }
            if (row1_1 >= col0_1 + 8) {
                qk_values[6] = qk_acc[6];
            }
            if (row1_1 >= col0_1 + 9) {
                qk_values[7] = qk_acc[7];
            }
            unsigned int qk_packed[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qk_values[_lp*2 + 0], qk_values[_lp*2+1 + 0]));
                qk_packed[_lp] = *(uint32_t*)&_bf2;
            }
            int lane_row_1 = lane % 16;
            int lane_col_1 = lane / 16 * 8;
            uint32_t _stmatrix_addr_1 = static_cast<uint32_t>((unsigned long long)(smem_qk_plain_addr + (unsigned int)(lane_col_1 / 16 * 512 + lane_row_1 * 32 + lane_col_1 % 16 * 2 ^ (lane_col_1 / 16 * 512 + lane_row_1 * 32 + lane_col_1 % 16 * 2 >> 7 & 1) << 4)));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&qk_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&qk_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&qk_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&qk_packed[3]))
                : "memory");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(pairwise_ready_addr);
            }
        }
        if (warp == 1 || warp == 2 || warp == 3) {
            mbarrier_wait(pairwise_ready_addr, pairwise_ready_phase);
        }
        long long qk_ws_base = ((long long)head_idx * (long long)total_chunks + (long long)gchunk) * 16 * 16;
        if (warp == 2) {
            float qk_fold_acc[8];
            unsigned int a_frag_2[4];
            unsigned int b_frag_2[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag_2[0]), "=r"(a_frag_2[1]), "=r"(a_frag_2[2]), "=r"(a_frag_2[3])
                : "r"((smem_abt_addr + (unsigned int)(lane / 16 * 8 / 16 * 512 + lane % 16 * 32 + lane / 16 * 8 % 16 * 2 ^ (lane / 16 * 8 / 16 * 512 + lane % 16 * 32 + lane / 16 * 8 % 16 * 2 >> 7 & 1) << 4)))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_2[0]), "=r"(b_frag_2[1]), "=r"(b_frag_2[2]), "=r"(b_frag_2[3])
                : "r"((smem_qk_plain_addr + (unsigned int)(lane % 16 / 8 * 8 / 16 * 512 + (8 * (lane / 16) + lane % 8) * 32 + lane % 16 / 8 * 8 % 16 * 2 ^ (lane % 16 / 8 * 8 / 16 * 512 + (8 * (lane / 16) + lane % 8) * 32 + lane % 16 / 8 * 8 % 16 * 2 >> 7 & 1) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_fold_acc[0]), "=f"(qk_fold_acc[1]), "=f"(qk_fold_acc[2]), "=f"(qk_fold_acc[3])
                : "r"(a_frag_2[0]), "r"(a_frag_2[1]), "r"(a_frag_2[2]), "r"(a_frag_2[3]), "r"(b_frag_2[0]), "r"(b_frag_2[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_fold_acc[4]), "=f"(qk_fold_acc[(4) + 1]), "=f"(qk_fold_acc[(4) + 2]), "=f"(qk_fold_acc[(4) + 3])
                : "r"(a_frag_2[0]), "r"(a_frag_2[1]), "r"(a_frag_2[2]), "r"(a_frag_2[3]), "r"(b_frag_2[2]), "r"(b_frag_2[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            int row0_2 = lane / 4;
            int row1_2 = row0_2 + 8;
            int col0_2 = lane % 4 * 2;
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(qk_fold_acc[0 + 0], qk_fold_acc[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(ws_qk_t))[qk_ws_base + (long long)row0_2 * 16 + (long long)col0_2]) = _pk;
            }
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(qk_fold_acc[2 + 0], qk_fold_acc[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(ws_qk_t))[qk_ws_base + (long long)row1_2 * 16 + (long long)col0_2]) = _pk;
            }
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(qk_fold_acc[4 + 0], qk_fold_acc[4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(ws_qk_t))[qk_ws_base + (long long)row0_2 * 16 + (long long)col0_2 + 8]) = _pk;
            }
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(qk_fold_acc[6 + 0], qk_fold_acc[6 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(ws_qk_t))[qk_ws_base + (long long)row1_2 * 16 + (long long)col0_2 + 8]) = _pk;
            }
        }
        if (warp == 1 || warp == 3) {
            int fold_slot = 0;
            int w_group_base = 0;
            if (warp == 3) {
                fold_slot = 2;
                w_group_base = 4;
            }
            #pragma unroll
            for (int local_group = 0; local_group < 4; local_group++) {
                int w_group = w_group_base + local_group;
                float w_fold_acc[8];
                unsigned int w_a_frag[4];
                unsigned int w_b_trans_frag[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(w_a_frag[0]), "=r"(w_a_frag[1]), "=r"(w_a_frag[2]), "=r"(w_a_frag[3])
                    : "r"((smem_abt_addr + (unsigned int)(lane / 16 * 8 / 16 * 512 + lane % 16 * 32 + lane / 16 * 8 % 16 * 2 ^ (lane / 16 * 8 / 16 * 512 + lane % 16 * 32 + lane / 16 * 8 % 16 * 2 >> 7 & 1) << 4)))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(w_b_trans_frag[0]), "=r"(w_b_trans_frag[1]), "=r"(w_b_trans_frag[2]), "=r"(w_b_trans_frag[3])
                    : "r"((smem_kr_addr + (unsigned int)((w_group * 16 + lane / 16 * 8) / 64 * 2048 + lane % 16 * 128 + (w_group * 16 + lane / 16 * 8) % 64 * 2 ^ ((w_group * 16 + lane / 16 * 8) / 64 * 2048 + lane % 16 * 128 + (w_group * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(w_fold_acc[0]), "=f"(w_fold_acc[1]), "=f"(w_fold_acc[2]), "=f"(w_fold_acc[3])
                    : "r"(w_a_frag[0]), "r"(w_a_frag[1]), "r"(w_a_frag[2]), "r"(w_a_frag[3]), "r"(w_b_trans_frag[0]), "r"(w_b_trans_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(w_fold_acc[4]), "=f"(w_fold_acc[(4) + 1]), "=f"(w_fold_acc[(4) + 2]), "=f"(w_fold_acc[(4) + 3])
                    : "r"(w_a_frag[0]), "r"(w_a_frag[1]), "r"(w_a_frag[2]), "r"(w_a_frag[3]), "r"(w_b_trans_frag[2]), "r"(w_b_trans_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                unsigned int packed_4[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(w_fold_acc[_lp*2 + 0], w_fold_acc[_lp*2+1 + 0]));
                    packed_4[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_2 = static_cast<uint32_t>((unsigned long long)(smem_w_out_addr + (unsigned int)((w_group * 16 + lane / 16 * 8) / 64 * 2048 + lane % 16 * 128 + (w_group * 16 + lane / 16 * 8) % 64 * 2 ^ ((w_group * 16 + lane / 16 * 8) / 64 * 2048 + lane % 16 * 128 + (w_group * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[3]))
                    : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (elect_sync()) {
                tma_store_4d(ws_w_tma, 0, gchunk * 16, head_idx, fold_slot / 2, smem_w_out_addr + (unsigned int)(fold_slot / 2 * 16 * 64 * 2));
            }
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group.read 0;");
        }
        __syncthreads();
        k_half_ready_phase ^= 1;
        k_full_ready_phase ^= 1;
        pairwise_ready_phase ^= 1;
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"

